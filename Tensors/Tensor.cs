using System;
using System.Text;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Tensors
{
    public enum Padding { Same, Const, }; // TODO Reflect
    public enum Distribution { Uniform, Normal };
    public enum Activation { Linear, ReLU, Sigmoid, Tanh };

    public class Tensor : IEnumerable<double>, IEnumerator<double>
    {
        public static bool debugEnumeration = false;
        public static bool debugGC = false;

        internal static System.Random _rng;
        static Tensor() => _rng = new Random();
        public static void Seed(Int32 seed) { _rng = new Random(seed); }

        private class Dimension
        {
            public int size, stride, index, padLeft, padRight;
            public Padding padType;
            public double padValue;
            public Dimension Clone() => (Dimension)MemberwiseClone();
        }

        private double[] _data;
        private readonly int _start;
        private int _offset;
        private readonly Dimension[] _dims;
        private Stack<double> _paddingValues;

        private int[] _shape;
        public int[] shape => _shape ?? (_shape = _dims.Select(s => s.size).ToArray());
        public string shapeStr => "(" + String.Join(",", shape) + ")";
        public readonly int size;
        public int rank => _dims.Length;
        private int _lastIndexUpdated;
        public int lastIndexUpdated {
            get { return _lastIndexUpdated; }
            private set { _lastIndexUpdated = value; }
        }
        public int[] indices => _dims.Select(s => s.index).ToArray();
        public double Current {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return _paddingValues?.Count() > 0 ? _paddingValues.Peek() : _data[_offset]; }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private set { _data[_offset] = value; }
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetCurrent(double value)
        {
            WarnAboutInplaceModification();
            Current = value;
        }

        public Tensor grad;
        public bool noGrad;
        private bool _requiresGrad;

        private Action<Tensor> _Backpropagate;
        public Action<Tensor> Backpropagate {
            get { return _Backpropagate; }
            set { _Backpropagate = noGrad ? (Action<Tensor>)null : value; }
        }

        object IEnumerator.Current => double.MinValue;//Test to see whether this one is ever used

        public Tensor(double[] data)
        {
            size = data.Length;
            _dims = new Dimension[] { new Dimension { size = size, stride = 1 } };
            _data = data;
            if (debugGC) {
                Console.WriteLine($"Data generation {GC.GetGeneration(_data)}");
                Console.WriteLine($"Tensor generation {GC.GetGeneration(this)}");
            }
        }

        public Tensor(params int[] sizes)
        {
            size = sizes.Aggregate((a, x) => a * x);
            _dims = new Dimension[sizes.Length];
            var currStride = size;
            for (var i = 0; i < sizes.Length; i++) {
                currStride /= sizes[i];
                _dims[i] = new Dimension
                {
                    size = sizes[i],
                    stride = sizes[i] > 1 ? currStride : 0,
                };
            }
            _data = new double[size];
            if (debugGC) {
                Console.WriteLine($"Data generation {GC.GetGeneration(_data)}");
                Console.WriteLine($"Tensor generation {GC.GetGeneration(this)}");
            }
        }

        private Tensor(double[] data, Dimension[] dims, int start, Action<Tensor> Backpropagate, bool noGrad)
        {
            _data = data;
            _dims = dims;
            size = shape.Aggregate((a, x) => a * x);
            _start = start;
            this.Backpropagate = Backpropagate;
            this.noGrad = noGrad;
            foreach (var dim in dims) {
                if (dim.padLeft > 0 || dim.padRight > 0) {
                    _paddingValues = new Stack<double>();
                    break;
                }
            }
            if (debugGC) {
                Console.WriteLine($"Data generation {GC.GetGeneration(_data)}");
                Console.WriteLine($"Tensor generation {GC.GetGeneration(this)}");
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
        public IEnumerator<double> GetEnumerator()
        {
            Reset();
            return this;
        }

        public void Reset()
        {
            _paddingValues?.Clear();
            if (debugEnumeration) Console.Write($"ResetOffset");
            _offset = _start;
            for (var i = 0; i < _dims.Length; i++) {
                _dims[i].index = 0;
                if (_dims[i].padLeft > 0 && _dims[i].padType == Padding.Const) {
                    if (debugEnumeration) Console.Write($" padLeft[{i}] with {_dims[i].padValue}");
                    _paddingValues.Push(_dims[i].padValue);
                }
            }
            _dims[_dims.Length - 1].index = -1;
            if (debugEnumeration) {
                var paddingValues = $"[{String.Join(",", (_paddingValues ?? new Stack<double>()))}]";
                Console.WriteLine($" {paddingValues}");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool MoveNext()
        {
            if (debugEnumeration) {
                string paddingValues = $"[{String.Join(",", (_paddingValues ?? new Stack<double>()))}]";
                Console.Write($"AdvanceOffset {String.Join(",", indices)} {paddingValues}");
            }
            for (var i = _dims.Length - 1; i >= 0; i--) {
                var dim = _dims[i];
                dim.index += 1;
                if (debugEnumeration) Console.Write($" {i}->{dim.index}");
                if (dim.index == 0) {
                    lastIndexUpdated = 0;
                    if (debugEnumeration) Console.WriteLine($" starting at {_offset}");
                    return true;
                } else if (dim.index < dim.padLeft) {
                    lastIndexUpdated = i;
                    if (debugEnumeration) Console.WriteLine($" paddingLeft {_offset}");
                    return true;
                } else if (dim.index == dim.padLeft) {
                    if (dim.padType == Padding.Const)
                        _paddingValues.Pop();
                    lastIndexUpdated = i;
                    if (debugEnumeration) Console.WriteLine($" contents {_offset}");
                    return true;
                } else if (dim.index < dim.size - dim.padRight) {
                    _offset += dim.stride;
                    lastIndexUpdated = i;
                    if (debugEnumeration) Console.WriteLine($" contents {_offset}");
                    return true;
                } else if (dim.padRight > 0 && dim.index == dim.size - dim.padRight) {
                    if (dim.padType == Padding.Const)
                        _paddingValues.Push(dim.padValue);
                    lastIndexUpdated = i;
                    if (debugEnumeration) Console.WriteLine($" paddingRight {_offset}");
                    return true;
                } else if (dim.index < dim.size) {
                    lastIndexUpdated = i;
                    if (debugEnumeration) Console.WriteLine($" paddingRight {_offset}");
                    return true;
                }

                if (dim.padType == Padding.Const) {
                    if (dim.padRight > 0)
                        _paddingValues.Pop();
                    if (dim.padLeft > 0)
                        _paddingValues.Push(dim.padValue);
                }
                _offset -= (dim.size - dim.padLeft - dim.padRight - 1) * dim.stride;
                dim.index = 0;
                if (debugEnumeration) Console.Write($"->{dim.index}");
            }
            lastIndexUpdated = -1;
            if (debugEnumeration) Console.WriteLine($" done {_offset}");
            return false;
        }

        public bool CloseTo(Tensor other, double tolerance = 1e-8)
        {
            if (!shape.SequenceEqual(other.shape))
                throw new ArgumentException($"Attempt to compare incompatible tensors {shapeStr} != {other.shapeStr}");

            Reset();
            other.Reset();
            while (MoveNext() && other.MoveNext())
                if (Math.Abs(Current - other.Current) > tolerance) return false;
            return true;
        }

        public void Fill_(double value)
        {
            WarnAboutInplaceModification();
            foreach (var unused in this)
                Current = value;
        }

        public override string ToString() => $"Tensor of shape {shapeStr}";

        private bool inplaceWarningSent;
        internal void WarnAboutInplaceModification()
        {
            if (Backpropagate != null && Backpropagate != StoreGrads && !inplaceWarningSent) {
                inplaceWarningSent = true;
                Console.WriteLine("WARNING: Inplace modification of tensor contents can mess up the results of automatic differentiation.");
                Console.WriteLine(Environment.StackTrace);
            }
        }

        public void Backward(double errorScale = 1)
        {
            var grads = new Tensor(shape);
            grads.noGrad = true;
            grads.Fill_(errorScale);
            Backpropagate(grads);
        }

        public Tensor RequireGrads(bool value = true)
        {
            if (value && (Backpropagate != null && Backpropagate != StoreGrads))
                throw new Exception("You shouldn't require grads on a calculated Tensor");

            _requiresGrad = value;
            Backpropagate = value ? StoreGrads : (Action<Tensor>)null;
            return this;
        }

        public void StoreGrads(Tensor grad)
        {
            if (!shape.SequenceEqual(grad.shape))
                Console.WriteLine($"Received incompatible grads {grad.shapeStr} for tensor {shapeStr}");

            if (this.grad == null) {
                this.grad = grad;
            } else {
                this.grad.AddM_(grad);
            }
        }

        public void ClearGrads() => grad = null;

        public static void Broadcast(Tensor inA, Tensor inB, out Tensor outA, out Tensor outB)
        {
            var broadcastRank = Math.Max(inA.rank, inB.rank);
            var newShapeA = new Dimension[broadcastRank];
            var newShapeB = new Dimension[broadcastRank];
            var broadcastDimsA = new List<int>();
            var broadcastDimsB = new List<int>();

            for (var i = 0; i < broadcastRank; i++) {
                var idxA = i - broadcastRank + inA.rank;
                var idxB = i - broadcastRank + inB.rank;
                if (i < broadcastRank - inA.rank) {
                    newShapeA[i] = new Dimension { size = inB._dims[idxB].size };
                    newShapeB[i] = inB._dims[idxB];
                    broadcastDimsA.Add(i);
                } else if (i < broadcastRank - inB.rank) {
                    newShapeA[i] = inA._dims[idxA];
                    newShapeB[i] = new Dimension { size = inA._dims[idxA].size };
                    broadcastDimsB.Add(i);
                } else if (inA._dims[idxA].size == inB._dims[idxB].size) {
                    newShapeA[i] = inA._dims[idxA];
                    newShapeB[i] = inB._dims[idxB];
                } else if (inA._dims[idxA].size == 1) {
                    newShapeA[i] = new Dimension { size = inB._dims[idxB].size };
                    newShapeB[i] = inB._dims[idxB];
                    broadcastDimsA.Add(i);
                } else if (inB._dims[idxB].size == 1) {
                    newShapeA[i] = inA._dims[idxA];
                    newShapeB[i] = new Dimension { size = inA._dims[idxA].size };
                    broadcastDimsB.Add(i);
                } else {
                    throw new ArgumentException($"Trying to broadcast incompatible shapes {inA.shapeStr} and {inB.shapeStr}");
                }
            }
            Action<Tensor> unBroadcastA = null, unBroadcastB = null;
            if (inA.Backpropagate != null) {
                unBroadcastA = grad => {
                    foreach (var dim in broadcastDimsA)
                        grad = grad.Mean(dim);

                    inA.Backpropagate(grad);
                };
            }
            if (inB.Backpropagate != null) {
                unBroadcastB = grad => {
                    foreach (var dim in broadcastDimsB)
                        grad = grad.Mean(dim);

                    inB.Backpropagate(grad);
                };
            }
            outA = new Tensor(inA._data, newShapeA, inA._start, unBroadcastA, inA.noGrad);
            outB = new Tensor(inB._data, newShapeB, inB._start, unBroadcastB, inB.noGrad);
        }

        public Tensor Copy()
        {
            var output = new Tensor(shape);
            Reset();
            while (MoveNext() && output.MoveNext())
                output.Current = Current;
            return output;
        }

        public Tensor T()
        {
            var order = new int[rank];
            for (var i = 0; i < rank - 2; i++)
                order[i] = i;
            order[rank - 2] = rank - 1;
            order[rank - 1] = rank - 2;
            return Permute(order);
        }

        public Tensor Permute(params int[] order)
        {
            if (order.Length != rank)
                throw new ArgumentException($"Permute({String.Join(",", order)}) wrong number of args for shape {shapeStr}");

            var newShape = new Dimension[rank];
            for (var i = 0; i < rank; i++)
                newShape[i] = _dims[order[i]].Clone();

            Action<Tensor> permuteGrads = null;
            if (Backpropagate != null)
                permuteGrads = grad => Backpropagate(grad.Permute(UnPermuteOrder(order)));

            return new Tensor(_data, newShape, _start, permuteGrads, noGrad);
        }

        public static int[] UnPermuteOrder(params int[] order)
        {
            var output = new int[order.Length];
            for (var i = 0; i < order.Length; i++)
                output[order[i]] = i;
            return output;
        }

        public Tensor MergeDimWithNext(int dim, int count)
        {
            // 
            throw new NotImplementedException();
        }

        public Tensor ReshapeDim(int dim, int[] shape)
        {
            throw new NotImplementedException();
        }

        public Tensor Reshape(int[] shape)
        {
            throw new NotImplementedException();
        }

        public Tensor Squeeze(int dim)
        {
            if (_dims[dim].size > 1)
                throw new ArgumentException($"Cannot squeeze dim {dim} in shape {shapeStr}");

            var newShape = new Dimension[rank - 1];
            for (var i = 0; i < rank; i++)
                if (i != dim)
                    newShape[i < dim ? i : i - 1] = new Dimension
                    {
                        size = _dims[i].size,
                        stride = _dims[i].stride,
                    };

            Action<Tensor> unsqueezeGrads = null;
            if (Backpropagate != null)
                unsqueezeGrads = grad => Backpropagate(grad.Unsqueeze(dim));

            return new Tensor(_data, newShape, _start, unsqueezeGrads, noGrad);
        }

        public Tensor Unsqueeze(int dim, int newSize = 1)
        {
            if (dim > rank)
                throw new ArgumentException($"Can't unsqueeze dimension {dim} in shape {shapeStr}");

            var newShape = new Dimension[rank - 1];
            newShape[dim] = new Dimension { size = newSize };
            for (var i = 0; i < rank; i++)
                newShape[i < dim ? i : i + 1] = new Dimension
                {
                    size = _dims[i].size,
                    stride = _dims[i].stride,
                };

            Action<Tensor> squeezeGrads = null;
            if (Backpropagate != null)
                squeezeGrads = grad => Backpropagate(grad.Squeeze(dim));

            return new Tensor(_data, newShape, _start, squeezeGrads, noGrad);
        }

        public Tensor Slice(int dim, int start, int length = 1)
        {
            if (start + length > _dims[dim].size)
                throw new ArgumentException($"Slice({dim}, {start}, {length}) incompatible with shape {shapeStr}");

            var newStart = _start + start * _dims[dim].stride;
            var newShape = new Dimension[rank];
            _dims.CopyTo(newShape, 0);
            newShape[dim] = new Dimension
            {
                size = length,
                stride = _dims[dim].stride,
            };

            Action<Tensor> padGrads = null;
            if (Backpropagate != null)
                padGrads = grad => Backpropagate(grad.Pad(dim, start - 1, _dims[dim].size - start - length, Padding.Const));

            return new Tensor(_data, newShape, newStart, padGrads, noGrad);
        }

        public Tensor Pad(int dim, int left, int right, Padding type, double value = 0)
        {
            if (dim >= rank)
                throw new ArgumentException($"Can't pad dimension {dim} in shape {shapeStr}");

            var newShape = new Dimension[rank];
            _dims.CopyTo(newShape, 0);
            newShape[dim] = new Dimension
            {
                size = _dims[dim].size + left + right,
                stride = _dims[dim].stride,
                padLeft = left,
                padRight = right,
                padType = type,
                padValue = value,
            };

            Action<Tensor> sliceGrads = null;
            if (Backpropagate != null)
                sliceGrads = grad => Backpropagate(grad.Slice(dim, left, _dims[dim].size));

            return new Tensor(_data, newShape, _start, sliceGrads, noGrad);
        }

        public static Tensor Elementwise(Tensor a, Func<double, double> calcFn)
        {
            var output = new Tensor(a.shape);
            output.noGrad = a.noGrad;
            a.Reset();
            while (output.MoveNext() && a.MoveNext())
                output.Current = calcFn(a.Current);
            return output;
        }

        public static Tensor Elementwise(Tensor a, Tensor b, Func<double, double, double> calcFn)
        {
            var output = new Tensor(a.shape);
            output.noGrad = a.noGrad || b.noGrad;
            Broadcast(a, b, out Tensor c, out Tensor d);
            while (output.MoveNext() && c.MoveNext() && d.MoveNext())
                output.Current = calcFn(c.Current, d.Current);
            return output;
        }

        public static Tensor Elementwise(Tensor a, Tensor b, Tensor c, Func<double, double, double, double> calcFn)
        {
            var output = new Tensor(a.shape);
            output.noGrad = a.noGrad || b.noGrad || c.noGrad;
            a.Reset();
            b.Reset();
            c.Reset();
            while (output.MoveNext() && a.MoveNext() && b.MoveNext() && c.MoveNext())
                output.Current = calcFn(a.Current, b.Current, c.Current);
            return output;
        }

        public Tensor Plus(double scalar)
        {
            var output = Elementwise(this, x => x + scalar);
            if (Backpropagate != null)
                output.Backpropagate = grad => Backpropagate(grad);
            return output;
        }

        public Tensor PlusM(Tensor other, double multiplier = 1)
        {
            var output = Elementwise(this, other, (x, y) => x + y * multiplier);
            output.Backpropagate = grad => {
                if (Backpropagate != null)
                    Backpropagate(grad);
                if (other.Backpropagate != null)
                    other.Backpropagate(grad.Times(multiplier));
            };
            return output;
        }

        public Tensor Times(double scalar)
        {
            var output = Elementwise(this, x => x * scalar);
            if (Backpropagate != null)
                output.Backpropagate = grad => Backpropagate(Elementwise(grad, g => g * scalar));
            return output;
        }

        public Tensor Times(Tensor other)
        {
            var output = Elementwise(this, other, (x, y) => x * y);
            output.Backpropagate = grad => {
                if (Backpropagate != null)
                    Backpropagate(Elementwise(grad, other, (g, y) => g * y));
                if (other.Backpropagate != null)
                    other.Backpropagate(Elementwise(grad, this, (g, x) => g * x));
            };
            return output;
        }

        public Tensor Power(double exponent)
        {
            var output = Elementwise(this, x => Math.Pow(x, exponent));
            if (Backpropagate != null)
                output.Backpropagate = grad =>
                    Backpropagate(Elementwise(grad, this, (g, x) => g * exponent * Math.Pow(x, exponent - 1)));
            return output;
        }

        public Tensor ReLU()
        {
            var output = Elementwise(this, x => Math.Max(0, x));
            if (Backpropagate != null)
                output.Backpropagate = grad =>
                    Backpropagate(Elementwise(grad, output, (g, o) => (o > 0 ? g : 0)));
            return output;
        }

        public Tensor Tanh()
        {
            var output = Elementwise(this, x => Math.Tanh(x));
            if (Backpropagate != null)
                output.Backpropagate = grad =>
                    Backpropagate(Elementwise(grad, output, (g, o) => (g * (1 - o * o))));
            return output;
        }

        public Tensor Sigmoid()
        {
            var output = Elementwise(this, x => 1 / (1 + Math.Exp(-x)));
            if (Backpropagate != null)
                output.Backpropagate = grad =>
                    Backpropagate(Elementwise(grad, output, (g, o) => (g * o * (1 - o))));
            return output;
        }

        public Tensor Mean(int dim)
        {
            var order = new int[rank];
            var newShape = new int[rank - 1];
            for (var i = 0; i < rank - 1; i++) {
                order[i] = (i < dim ? i : i + 1);
                newShape[i] = _dims[i < dim ? i : i + 1].size;
            }
            order[rank - 1] = dim;
            var permuted = Permute(order);
            var output = new Tensor(newShape);
            while (permuted.MoveNext()) {
                if (permuted.lastIndexUpdated == rank - 1) {
                    output.Current += permuted.Current;
                } else {
                    output.Current /= _dims[dim].size;
                    output.MoveNext();
                    output.Current = permuted.Current;
                }
            }
            output.Current /= _dims[dim].size;

            if (Backpropagate != null)
                output.Backpropagate = grad =>
                    Backpropagate(grad.Unsqueeze(dim, _dims[dim].size));
            return output;
        }

        public Tensor Mean()
        {
            var output = new Tensor(1);
            foreach (var v in this)
                output.Current += v;
            output.Current /= size;

            if (Backpropagate != null)
                output.Backpropagate = grad => {
                    for (var dim = 0; dim < rank; dim++)
                        grad = grad.Unsqueeze(dim, _dims[dim].size);
                    Backpropagate(grad);
                };
            return output;
        }

        public Tensor MatrixMultiply(Tensor other)
        {
            var otherT = other.T();
            Broadcast(this.Unsqueeze(this.rank - 1), otherT.Unsqueeze(other.rank - 2), out Tensor a, out Tensor b);
            var newShape = a.shape.Take(a.rank - 1).ToArray();
            newShape[a.rank - 2] = b.shape[b.rank - 2];
            var output = new Tensor(newShape);
            while (a.MoveNext() && b.MoveNext()) {
                if (a.lastIndexUpdated == a.rank - 1)
                    output.MoveNext();
                output.Current += a.Current * b.Current;
            }

            output.Backpropagate = grad => {
                if (Backpropagate != null)
                    Backpropagate(grad.MatrixMultiply(otherT));

                if (other.Backpropagate != null)
                    other.Backpropagate(T().MatrixMultiply(grad));
            };
            return output;
        }

        public void AddM_(Tensor other, double multiplier = 1)
        {
            Reset();
            other.Reset();
            while (MoveNext() && other.MoveNext())
                Current += other.Current * multiplier;
        }

        // The IEnumerator must be disposable, but in our case the IEnumerator == the 
        // IEnumerable which we want to keep, so we just do nothing.
        void IDisposable.Dispose() { }
    }
}
