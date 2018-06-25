using System;
using System.Linq;
using System.IO;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Diagnostics;

/**
 * TODO
 * - Pad(dim, before, after, value || replicate || reflect)
 * 
 * - Parallelise some ops using Environment.ProcessorCount ??
 * 
 * - Tests
 *   - given an order check
 *     t.Slice(d, ..).Permute(order) == t.Permute(order).Slice(order[d], ..)
 *     t.Permute(order).Slice(d, ..).Permute(UnPermuteOrder(order)) == t.Slice(order[d], ..)
 * 
 *
 * - Question: how to get _data out efficiently without making it public?
 *   - tensor[coords] getter for one element at a time...
 *   - DangerouslyAccessData_() returns the data array by reference
 *
 * - algo strassen
 *   - modern processors do pretty well with the naive approach
 *     according to some accounts, there are no gains under size 500, and
 *     even then the gains are small.
 */

/// <summary>
/// A tensor class that works like a cross between numpy and pytorch.
///
/// A tensor has a one-dimensional storage array, and a multi-dimensional shape.
/// The _strides array describes how to move along each dimensions. The class is
/// designed to be copy-on-write, creating new tensors instead of
/// mutating existing tensors. The underlying storage array is shared between
/// tensors where possible.
///
/// This copy-on-write architecture will allow the addition of math operations
/// with tape-based automatic differentiation.
/// </summary>
/// <example>
/// The idea is to be able to use it as follows...
/// <code>
/// var t1 = Tensor(shape1, data1, requiresGrad=true);
/// var t2 = Tensor(shape2, data2, requiresGrad=false);
///
/// var t3 = t1.plus(t2);
/// var loss = t3.lossFunction(targetValues);
/// loss.Backward(); // this should backpropagate all the way back to the inputs.
///
/// var t1Gradient = t1.grad; // ??
/// </code>
/// </example>
class Tensor
{
    #region Properties

    
    public readonly int[]  shape = new int[] {1};
    public readonly bool   contiguous = true;
    public readonly bool   naturalOrder = true;
    public bool            noGrad = false;
    
    private bool           _requiresGrad;
    public bool            requiresGrad {
        get { return _requiresGrad; }
        set {
            if (value && (Backpropagate != null && Backpropagate != StoreGrads))
                throw new Exception("You shouldn't require grads on a calculated Tensor");
            
            _requiresGrad = value;
            if (value && !noGrad)
            {
                Backpropagate = StoreGrads;
            }
            else
            {
                Backpropagate = null;
            }
        }
    }
    
    public Tensor          grad;

    private readonly int[] _strides = new int[] {1};
    private readonly int   _start = 0;
    private double[]       _data;

    public int size { get { return shape.Aggregate((a, x) => a * x); } }
    public int rank { get { return shape.Length; } }

    public delegate void Backpropagator(Tensor grad);
    private Backpropagator _Backpropagate;
    public Backpropagator Backpropagate {
        get { return _Backpropagate; }
        set { if (!noGrad) _Backpropagate = value; }
    }

    #endregion

    #region Constructors

    public Tensor(double[] data)
    {
        this.shape = new int[] {data.Length};
        _data = data;
    }

    public Tensor(params int[] shape)
    {
        this.shape = shape;
        _data = new double[size];
        _strides = Tensor.MakeStrides(shape, 1);
    }

    public Tensor(double scalar)
    {
        _data = new double[] { scalar };
    }

    private Tensor(double[] data, int[] shape, int[] strides, int start,
        Backpropagator backward, bool noGrad = false)
    {
        this.shape = shape;
        _data = data;
        _strides = strides;
        _start = start;
        naturalOrder = true;
        for (var i = 0; i < rank - 1; i++)
        {
            if (_strides[i] != 0 && _strides[i] < _strides[i + 1])
            {
                naturalOrder = false;
                break;
            }
        }
        int maxStride = 0;
        int maxStrideIndex = -1;
        for (var i = 0; i < rank; i++)
        {
            if (_strides[i] > maxStride)
            {
                maxStride = _strides[i];
                maxStrideIndex = i;
            }
        }
        this.contiguous = _strides[maxStrideIndex] * shape[maxStrideIndex] == size;
        //Console.WriteLine($"New tensor ({String.Join(",", shape)}) ({String.Join(",", _strides)}) from {start}, contig {contiguous}");
        //Console.WriteLine($"naturalOrder {naturalOrder} _strides[0] * shape[0] {_strides[0] * shape[0]} == {size}");
        //PrintImplementationDetails();
        this.noGrad = noGrad;
        Backpropagate = backward;
    }

    private static int[] MakeStrides(int[] shape, int baseStride)
    {
        // This function may be better off in a Utils class.
        var strides = new int[shape.Length];
        strides[shape.Length - 1] = baseStride;
        for (var i = shape.Length - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];

        return strides;
    }

    #endregion

    #region Inplace fill operations

    private void WarnAboutInplaceModification()
    {
        if (Backpropagate != null)
        {
            Console.WriteLine("WARNING inplace modification of tensor contents can mess up the results of automatic differentiation.");
            Console.WriteLine(Environment.StackTrace);
        }
    }

    public void Fill_(double value)
    {
        WarnAboutInplaceModification();
        Array.Fill(_data, value, _start, size);
    }

    public void FillWithRange_(double start = 0, double step = 1)
    {
        WarnAboutInplaceModification();
        for (var i = 0; i < size; i++)
            _data[i + _start] = start + i * step;
    }

    public void FillUniform_(double minval = 0, double maxval = 1)
    {
        WarnAboutInplaceModification();
        var random = new Random();
        for (var i = 0; i < size; i++)
            _data[i + _start] = minval + random.NextDouble() * (maxval - minval);
    }

    public void FillNormal_(double mean = 0, double std = 1)
    {
        WarnAboutInplaceModification();
        var random = new Random();
        
        // adjust the std to account for truncation at 2*std
        // .8796 = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        std /= .8796;
        
        var i = 0;
        while (i < size)
        {
            // Box-Muller transform gives two values each time
            var uniform1 = random.NextDouble();
            var uniform2 = random.NextDouble();
            var distance = Math.Sqrt(-2.0 * Math.Log(uniform1));
            var angle = 2.0 * Math.PI * uniform2;

            // Discard if more than 2*std away from the mean.
            var randomBit = distance * Math.Sin(angle);
            if (Math.Abs(randomBit) > 2)
                continue;
            
            _data[i++] = mean + std * randomBit;

            if (i == size)
                break;
            
            randomBit = distance * Math.Cos(angle);
            if (Math.Abs(randomBit) > 2)
                continue;
            
            _data[(i++) + _start] = mean + std * randomBit;
        }
    }
    
    public enum WeightDistribution {Uniform, Normal};
    public enum Activation {Linear, ReLU, Sigmoid, Tanh};
    public void InitialiseWeights_(WeightDistribution dist, Activation activation, int fanIn, int fanOut)
    {
        // He et al prove that using either fanIn or fanOut rather than their 
        // average would work fine in most network architectures.
        
        // If you use fanOut at all layers, then the variance of the gradients is
        // the same at each layer, but the variance of the output of layer i =
        // Var(input) * fanInOfLayer_1 / fanOutOfLayer_i.
        
        // Using fanIn keeps the output variance the same, but allows the gradient 
        // variance to vary.
        
        double limitOrStd = 1;
        if (dist == WeightDistribution.Uniform)
        {
            if (activation == Activation.Linear || activation == Activation.Tanh)
            {
                limitOrStd = Math.Sqrt(6.0 / (fanIn + fanOut));
            }
            else if (activation == Activation.Sigmoid)
            {
                limitOrStd = 4 * Math.Sqrt(6.0 / (fanIn + fanOut));
            }
            else if (activation == Activation.ReLU)
            {
                limitOrStd = Math.Sqrt(12.0 / (fanIn + fanOut));
            }
            Console.WriteLine($"InitialiseWeights using limit {limitOrStd}");
            FillUniform_(-limitOrStd, limitOrStd);
        }
        else if (dist == WeightDistribution.Normal)
        {
            if (activation == Activation.Linear || activation == Activation.Tanh)
            {
                limitOrStd = Math.Sqrt(2.0 / (fanIn + fanOut));
            }
            else if (activation == Activation.Sigmoid)
            {
                limitOrStd = 4 * Math.Sqrt(2.0 / (fanIn + fanOut));
            }
            else if (activation == Activation.ReLU)
            {
                limitOrStd = Math.Sqrt(4.0 / (fanIn + fanOut));
            }
            Console.WriteLine($"InitialiseWeights using std {limitOrStd}");
            FillNormal_(0, limitOrStd);
        }
    }

    #endregion

    #region Display and debug

    public override string ToString()
    {
        return $"Tensor of shape ({String.Join(",", shape)})" +
            (size == 1 ? $" containing value {_data[_start]}" : "");
    }

    public void PrintImplementationDetails()
    {
        Console.Error.WriteLine(this + $", contiguous {contiguous}, naturalOrder {naturalOrder}");
        Console.Error.WriteLine($"Strides ({String.Join(",", _strides)}), start {_start}, data size {_data.Length}");
    }

    public void PrintContents()
    {
        Console.Error.WriteLine(this);
        Console.Error.WriteLine(ContentsAsString());
    }

    public string ContentsAsString()
    {
        var output = new List<string>();
        var (indices, dataIndex) = StartPosition();
        var matrixRow = new double[shape[rank - 1]];
        var rowHead = 0;
        while (dataIndex < _data.Length)
        {
            if (rank > 2 && indices[rank - 2] == 0 && indices[rank - 1] == 0)
                output.Add($"{String.Join(",", indices.Take(rank - 2))}");

            matrixRow[rowHead++] = _data[dataIndex];

            if (rowHead == shape[rank - 1])
            {
                rowHead = 0;
                output.Add($"  {String.Join(", ", matrixRow)}");
            }
            IncrementPosition(ref indices, ref dataIndex, shape);
        }
        return String.Join("\n", output);
    }

    #endregion

    #region Indexing and broadcasting

    // single element indexing via coordinates
    public double this[params int[] indices] 
    { 
        get { return _data[DataIndex(indices)]; } 
        set { 
            WarnAboutInplaceModification();
            _data[DataIndex(indices)] = value; 
        } 
    }

    public double[] DangerouslyAccessData_()
    {
        return _data;
    }

    public bool BroadcastableTo(int[] desiredShape)
    {
        int[] shapeToImitate;
        if (desiredShape.Length > rank)
        {
            shapeToImitate = new int[rank];
            Array.Copy(desiredShape, desiredShape.Length - rank, shapeToImitate, 0, rank);
        }
        else
        {
            shapeToImitate = desiredShape;
        }

        if (shapeToImitate.Length != rank)
        {
            throw new ArgumentException(
                $"Bad desired shape {String.Join(",", desiredShape)} for broadcasting to from {String.Join(",", shape)}.");
        }

        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] > 1 && shape[i] != shapeToImitate[i])
                return false;
        }
        return true;
    }
    
    public static int[] BroadcastShape(int[] shape1, int[] shape2)
    {
        var outputShape = new int[Math.Max(shape1.Length, shape2.Length)];
        for (var i = 1; i <= outputShape.Length; i++)
        {
            var s1i = shape1.Length - i;
            var s2i = shape2.Length - i;
            var osi = outputShape.Length - i;
            
            if (i <= shape1.Length && i <= shape2.Length 
             && shape1[s1i] == shape2[s2i]) 
            {
                outputShape[osi] = shape1[s1i];
            }
            else if (i > shape1.Length || shape1[s1i] == 1)
            {
                outputShape[osi] = shape2[s2i];
            }
            else if (i > shape2.Length || shape2[s2i] == 1) 
            {
                outputShape[osi] = shape1[s1i];
            }
            else
            {
                throw new ArgumentException($"Incompatible shapes [{String.Join(",", shape1)}] and [{String.Join(",", shape2)}]");
            }
        }
        //Console.WriteLine($"Broadcastshape [{String.Join(",", shape1)}] [{String.Join(",", shape2)}] => [{String.Join(",", outputShape)}]");
        return outputShape;
    }
    
    public static (int[], int[]) MultiplyShapes(int[] shape1, int[] shape2)
    {
        var newRank = Math.Max(shape1.Length, shape2.Length);
        var inputShape = new int[newRank];
        var outputShape = new int[newRank - 1];
        for (var i = 1; i <= newRank; i++)
        {
            var s1i = shape1.Length - i;
            var s2i = shape2.Length - i;
            var isi = inputShape.Length - i;
            
            if (i <= shape1.Length && i <= shape2.Length 
             && shape1[s1i] == shape2[s2i]) 
            {
                inputShape[isi] = shape1[s1i];
            }
            else if (i > shape1.Length || shape1[s1i] == 1)
            {
                inputShape[isi] = shape2[s2i];
            }
            else if (i > shape2.Length || shape2[s2i] == 1) 
            {
                inputShape[isi] = shape1[s1i];
            }
            else
            {
                //Console.WriteLine($"Multiplyshape [{String.Join(",", shape1)}] [{String.Join(",", shape2)}]");
                throw new ArgumentException($"Incompatible shapes for matrix multiply [{String.Join(",", shape1)}] and [{String.Join(",", shape2)}] at ");
            }
            if (isi < outputShape.Length)
                outputShape[isi] = inputShape[isi];
        }
        //Console.WriteLine($"Multiplyshape [{String.Join(",", shape1)}] [{String.Join(",", shape2)}] => [{String.Join(",", inputShape)}] [{String.Join(",", outputShape)}]");
        return (inputShape, outputShape);
    }

    /// <summary>
    /// Use this function for looping over a tensor.
    ///
    /// indices should be initialised to all zeros.
    /// dataIndex should be initialised to _start.
    ///
    /// First check that this tensor is BroadcastableTo(desiredShape).
    ///
    /// Use of ref serves as a reminder that this function mutates those inputs.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void IncrementPosition(ref int[] indices, ref int dataIndex, int[] desiredShape = null)
    {
        if (desiredShape == null)
            desiredShape = shape;
        
        for (var i = 1; i <= desiredShape.Length; i++)
        {
            var desiredDim = desiredShape.Length - i;
            var thisDim = shape.Length - i;
            indices[desiredDim] += 1;
            if (thisDim < 0)
            {
                // broadcasting over extra dims
            }
            else if (indices[desiredDim] < desiredShape[desiredDim])
            {
                if (shape[thisDim] > 1) // Only modify dataIndex if we are not broadcasting along dim.
                    dataIndex += _strides[thisDim];
                
                break;
            }
            else
            {
                indices[desiredDim] = 0;
                if (shape[thisDim] > 1)
                {
                    dataIndex -= _strides[thisDim] * (shape[thisDim] - 1);
                }
                if (desiredDim == 0) // End of indexable data.
                {
                    dataIndex = _data.Length;
                }
            }
        }
    }
    
    public (int[] indices, int head) StartPosition(int[] desiredShape = null)
    {
        if (desiredShape == null)
            desiredShape = shape;
        
        return (new int[desiredShape.Length], _start);
    }

    /// <summary>
    /// Given an array of indices, find the address of the underlying data element.
    /// </summary>
    private int DataIndex(int[] indices)
    {
        var dataIndex = _start;
        for (var i = 0; i < indices.Length; i++)
        {
            var index = indices[i];
            if (index < 0) // Negative indexing.
            {
                index += shape[i];
            }

            if (shape[i] > 1 && index > shape[i])
            {
                throw new ArgumentOutOfRangeException($"Received {i}th index {indices[i]} for shape {shape}");
            }

            dataIndex += index * _strides[i];
        }
        return dataIndex;
    }

    #endregion

    #region Autograd
    
    public void Backward(double errorScale = 1)
    {
        var grads = new Tensor(shape);
        grads.noGrad = true;
        grads.Fill_(errorScale);
        grads.noGrad = true;
        Backpropagate(grads);
    }

    public void StoreGrads(Tensor grad)
    {
        if (!shape.SequenceEqual(grad.shape))
            Console.WriteLine($"Incompatible grads {String.Join(",", grad.shape)} with tensor {String.Join(",", shape)}");
        
        if (this.grad == null)
        {
            this.grad = grad;
        }
        else
        {
            this.grad.Addm_(grad);
        }
    }
    
    public void ClearGrads()
    {
        grad = null;
    }

    public Tensor Detach()
    {
        return new Tensor(_data, shape, _strides, _start, null, noGrad);
    }
    
    public void Detach_()
    {
        Backpropagate = null;
    }

    #endregion

    #region Basic tensor ops, Copy, Reshape, Slice, etc.

    private void ShowWarningAndStackTrace(string warning)
    {
        Console.WriteLine("WARNING: " + warning);
        Console.WriteLine(Environment.StackTrace);
    }

    public Tensor Copy(bool warnIfContiguous = true)
    {
        var newData = new double[size];
        Tensor output;
        if (contiguous && naturalOrder)
        {
            if (warnIfContiguous)
                ShowWarningAndStackTrace("Copying contiguous tensor data, is this necessary?");

            Array.Copy(_data, _start, newData, 0, size);
            output = new Tensor(newData);
            output.noGrad = true;
            return output.Reshape(shape);
        }

        var (indices, readHead) = StartPosition();

        for (var writeHead = 0; writeHead < newData.Length; writeHead++)
        {
            newData[writeHead] = _data[readHead];
            IncrementPosition(ref indices, ref readHead, shape);
        }
        output = new Tensor(newData);
        output.noGrad = true;
        return output.Reshape(shape);
    }

    /// <summary>
    /// Transpose the last two dimensions of the tensor.
    /// </summary>
    public Tensor T()
    {
        int[] order = new int[rank];
        for (var i = 0; i < rank - 2; i++)
            order[i] = i;
        order[rank - 2] = rank - 1;
        order[rank - 1] = rank - 2;

        return Permute(order);
    }

    public Tensor Permute(params int[] order)
    {
        if (order.Length != rank)
            throw new ArgumentException($"Permute({String.Join(",", order)}) wrong number of args for shape ({String.Join(",", shape)})");
        
        var newShape = new int[rank];
        var newStrides = new int[rank];
        for (var i = 0; i < rank; i++)
        {
            newShape[i] = shape[order[i]];
            newStrides[i] = _strides[order[i]];
        }
        Backpropagator permuteGrads = null;
        if (Backpropagate != null)
            permuteGrads = grad => Backpropagate(grad.Permute(UnPermuteOrder(order)));
        
        return new Tensor(_data, newShape, newStrides, _start, permuteGrads, noGrad);
    }
    
    public static int[] UnPermuteOrder(int[] permuteOrder)
    {
        var unPermuteOrder = new int[permuteOrder.Length];
        for (var i = 0; i < permuteOrder.Length; i++)
            unPermuteOrder[permuteOrder[i]] = i;
        return unPermuteOrder;
    }

    public Tensor Reshape(params int[] shape)
    {
        var wildcardIndex = Array.IndexOf(shape, -1);
        if (wildcardIndex != Array.LastIndexOf(shape, -1))
            throw new ArgumentException($"Too many unknowns in shape ({String.Join(",", shape)})");

        var newSize = shape.Aggregate((a, x) => a * x);

        if (wildcardIndex >= 0)
        {
            int missingDivisor = size / -newSize;
            shape[wildcardIndex] = missingDivisor;
            newSize *= -missingDivisor;
        }
        
        if (newSize != size)
            throw new ArgumentException($"New size {newSize} ({String.Join(",", shape)}) does not match old size {size} ({String.Join(",", this.shape)})");

        int[] newStrides;
        
        if (!contiguous || !naturalOrder)
        {
            // each new dimension must either exactly match an existing dimension,
            // or the product of consecutive new dimensions must match a contiguous
            // section of the existing tensor dimensions.
            var newShapeIndex = -1;
            var oldShapeIndex = -1;
            var newDimSize = 0;
            var oldDimSize = 0;
            var newShape = shape;
            var oldShape = this.shape;
            newStrides = new int[newShape.Length];
            
            bool contiguousEnough = true;
            while (newShapeIndex < newShape.Length && oldShapeIndex < oldShape.Length)
            {
                if (newDimSize == oldDimSize)
                {
                    newShapeIndex += 1;
                    oldShapeIndex += 1;
                    if (newShapeIndex == newShape.Length && oldShapeIndex == oldShape.Length)
                        break;

                    newDimSize = newShape[newShapeIndex];
                    oldDimSize = oldShape[oldShapeIndex];
                    newStrides[newShapeIndex] = oldShape[oldShapeIndex] * _strides[oldShapeIndex] / newShape[newShapeIndex];
                }
                else if (newDimSize < oldDimSize && newShapeIndex < newShape.Length - 1)
                {
                    newShapeIndex += 1;
                    newDimSize *= newShape[newShapeIndex];
                    newStrides[newShapeIndex] = newStrides[newShapeIndex - 1] / newShape[newShapeIndex];
                }
                else if (oldDimSize < newDimSize)
                {
                    oldShapeIndex += 1;
                    if (oldShapeIndex >= oldShape.Length) // can this ever happen?
                    {
                        contiguousEnough = false;
                        throw new ArgumentException($"Reshape({String.Join(",", shape)}) tensor is not contiguous enough. Run Copy() first.");
                    }

                    if (_strides[oldShapeIndex - 1] != _strides[oldShapeIndex] * oldShape[oldShapeIndex])
                    {
                        contiguousEnough = false;
                        break;
                    }

                    oldDimSize *= oldShape[oldShapeIndex];
                }
            }
            if (!contiguousEnough)
                //throw new ArgumentException($"Reshape({String.Join(",", shape)}) tensor is not contiguous enough. Run Copy() first.");
                return Copy().Reshape(shape);
        }
        else
        {
            newStrides = Tensor.MakeStrides(shape, _strides[_strides.Length - 1]);
        }

        if (size != newSize)
            throw new ArgumentException($"Reshape({String.Join(",", shape)}) is incompatible with tensor shape ({String.Join(",", this.shape)})");

        Backpropagator reshapeGrads = null;
        if (Backpropagate != null)
            reshapeGrads = grad => Backpropagate(grad.Reshape(this.shape));
        
        return new Tensor(_data, shape, newStrides, _start, reshapeGrads, noGrad);
    }

    public Tensor Squeeze(int dim = -1)
    {
        var squeezedDims = new List<int>();
        var newShape = new List<int>();
        var newStrides = new List<int>();
        
        if (dim >= 0 && shape[dim] > 1)
            throw new ArgumentException($"Squeeze({dim}) cannot squeeze dimension of size {shape[dim]}");
        
        for (var i = 0; i < rank; i++)
        {
            if (shape[i] > 1 || (dim >= 0 && i != dim))
            {
                newShape.Add(shape[i]);
                newStrides.Add(_strides[i]);
            }
            else
            {
                squeezedDims.Add(i);
            }
        }
        Backpropagator unsqueezeGrads = null;
        if (Backpropagate != null)
        {
            unsqueezeGrads = grad => {
                foreach (var udim in squeezedDims)
                    grad = grad.Unsqueeze(udim);
                
                Backpropagate(grad);
            };
        }
        return new Tensor(_data, newShape.ToArray(), newStrides.ToArray(), _start, unsqueezeGrads, noGrad);
    }

    public Tensor Unsqueeze(int dim, int dimLength = 1)
    {
        if (dim > rank)
            throw new ArgumentException($"Can't unsqueeze dimension {dim} in shape ({String.Join(",", shape)})");

        var newShape = new int[rank + 1];
        var newStrides = new int[rank + 1];
        for (var i = 0; i < rank; i++)
        {
            newShape[i < dim ? i : i + 1] = shape[i];
            newStrides[i < dim ? i : i + 1] = _strides[i];
        }
        newShape[dim] = dimLength;
        newStrides[dim] = 0;
        
        Backpropagator squeezeGrads = null;
        if (Backpropagate != null)
            squeezeGrads = grad => Backpropagate(grad.Squeeze(dim));
        
        return new Tensor(_data, newShape, newStrides, _start, squeezeGrads, noGrad);
    }
    
    public Tensor BroadcastToShape(params int[] shape)
    {
        if (shape.SequenceEqual(this.shape))
            return this;
        
        var broadcastDims = new List<int>();
        var newStrides = new int[shape.Length];
        for (var i = 1; i <= shape.Length; i++)
        {
            var this_i = this.shape.Length - i;
            var new_i = shape.Length - i;
            if (this_i < 0)
            {
                broadcastDims.Add(new_i);
                newStrides[new_i] = 0;
            }
            else if (this.shape[this_i] != shape[new_i])
            {
                broadcastDims.Add(new_i);
                newStrides[new_i] = 0;
            }
            else
            {
                newStrides[new_i] = _strides[this_i];
            }
        }
        Backpropagator unBroadcast = null;
        if (Backpropagate != null)
            unBroadcast = grad => {
                foreach (var dim in broadcastDims)
                    grad = grad.Mean(dim);
                
                Backpropagate(grad);
            };
            
        return new Tensor(_data, shape, newStrides, _start, unBroadcast, noGrad);
    }

    public Tensor Slice(int dim, int start, int length = 1)
    {
        if (start + length > shape[dim])
            throw new ArgumentException($"Slice({dim}, {start}, {length}) incompatible with shape {String.Join(",", shape)}");

        var newStart = _start + start * _strides[dim];
        var newShape = new int[rank];
        shape.CopyTo(newShape, 0);
        newShape[dim] = length;
        
        Backpropagator padGrads = null;
        if (Backpropagate != null)
            padGrads = grad => Backpropagate(grad.Pad(dim, start - 1, shape[dim] - start - length));
        
        return new Tensor(_data, newShape, _strides, newStart, padGrads, noGrad);
    }
    
    public Tensor Pad(int dim, int countBefore, int countAfter, double value = 0)
    {
        throw new NotImplementedException();
    }

    #endregion

    #region Tests

    public bool CloseTo(Tensor other, double tolerance = 1e-8)
    {
        var (indices, readHead) = StartPosition();
        var (indicesOther, readHeadOther) = other.StartPosition();
        while(readHead < _data.Length && readHeadOther < other._data.Length)
        {
            if (Math.Abs(_data[readHead] - other._data[readHeadOther]) > tolerance)
                return false;

            IncrementPosition(ref indices, ref readHead, shape);
            other.IncrementPosition(ref indicesOther, ref readHeadOther, other.shape);
        }
        return true;
    }

    public bool InefficientEquals(Tensor other)
    {
        return ContentsAsString() == other.ContentsAsString();
    }

    #endregion

    #region Internal Elementwise ops
    
    public delegate double OneInputCalculator(double input);
    public delegate double TwoInputCalculator(double a, double b);
    public delegate double ThreeInputCalculator(double a, double b, double c);
    
    public static Tensor Elementwise(Tensor a, OneInputCalculator calcFn)
    {
        var output = new Tensor(a.shape);
        output.noGrad = a.noGrad;
        
        var (indices, readHead) = a.StartPosition();
        var writeHead = 0;
        while(writeHead < output.size && readHead < a._data.Length)
        {
            output._data[writeHead++] = calcFn(a._data[readHead]);
            a.IncrementPosition(ref indices, ref readHead, output.shape);
        }
        return output;
    }
    
    public static Tensor Elementwise(Tensor a, Tensor b, TwoInputCalculator calcFn)
    {
        var output = new Tensor(a.shape);
        output.noGrad = a.noGrad || b.noGrad;
        
        var (indices, readHead) = a.StartPosition();
        var (indicesB, readHeadB) = b.StartPosition();
        var writeHead = 0;
        while(writeHead < output.size && readHead < a._data.Length && readHeadB < b._data.Length)
        {
            output._data[writeHead++] = calcFn(a._data[readHead], b._data[readHeadB]);
            a.IncrementPosition(ref indices, ref readHead);
            b.IncrementPosition(ref indicesB, ref readHeadB);
        }
        return output;
    }
    
    public static Tensor Elementwise(Tensor a, Tensor b, Tensor c, ThreeInputCalculator calcFn)
    {
        var output = new Tensor(a.shape);
        output.noGrad = a.noGrad || b.noGrad || c.noGrad;
        
        var (indices, readHead) = a.StartPosition();
        var (indicesB, readHeadB) = b.StartPosition();
        var (indicesC, readHeadC) = c.StartPosition();
        var writeHead = 0;
        while(writeHead < output.size && readHead < a._data.Length && readHeadB < b._data.Length && readHeadC < c._data.Length)
        {
            output._data[writeHead++] = calcFn(a._data[readHead], b._data[readHeadB], c._data[readHeadC]);
            a.IncrementPosition(ref indices, ref readHead, output.shape);
            b.IncrementPosition(ref indicesB, ref readHeadB, output.shape);
            c.IncrementPosition(ref indicesC, ref readHeadC, output.shape);
        }
        return output;
    }
    
    #endregion
    
    #region Math ops
    
    public Tensor Times(double scalar)
    {
        var output = Elementwise(this, x => x*scalar);
        if (Backpropagate != null)
            output.Backpropagate = grad => {
                Backpropagate(Elementwise(grad, g => g * scalar));
            };
        
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
    
    public Tensor Plus(double scalar)
    {
        var output = Elementwise(this, x => x + scalar);
        if (Backpropagate != null)
            output.Backpropagate = grad => {
                Backpropagate(Elementwise(grad, g => g));
            };
        
        return output;
    }
    
    public Tensor Plus(Tensor other)
    {
        var output = Elementwise(this, other, (x, y) => x + y);
        output.Backpropagate = grad => {
            if (Backpropagate != null)
                Backpropagate(grad);
                
            if (other.Backpropagate != null)
                other.Backpropagate(grad);
        };
        return output;
    }
    
    public Tensor Minus(Tensor other)
    {
        var output = Elementwise(this, other, (x, y) => x - y);
        output.Backpropagate = grad => {
            if (Backpropagate != null)
                Backpropagate(grad);
                
            if (other.Backpropagate != null)
                other.Backpropagate(Elementwise(grad, g => -g));
        };
        return output;
    }
    
    public Tensor ReLU()
    {
        var output = Elementwise(this, x => Math.Max(0, x));
        if (Backpropagate != null)
            output.Backpropagate = grad => {
                Backpropagate(Elementwise(grad, output, (g, o) => (o > 0 ? g : 0)));
            };
        
        return output;
    }
    
    public Tensor Tanh()
    {
        var output = Elementwise(this, x => Math.Tanh(x));
        if (Backpropagate != null)
            output.Backpropagate = grad => {
                Backpropagate(Elementwise(grad, output, (g, o) => (g * (1 - o*o))));
            };
        
        return output;
    }
    
    public Tensor Sigmoid()
    {
        var output = Elementwise(this, x => 1 / (1 + Math.Exp(-x)));
        if (Backpropagate != null)
            output.Backpropagate = grad => {
                Backpropagate(Elementwise(grad, output, (g, o) => (g * o * (1 - o))));
            };
        
        return output;
    }
    
    public Tensor Square()
    {
        var output = Elementwise(this, x => x*x);
        if (Backpropagate != null)
            output.Backpropagate = grad => {
                Backpropagate(Elementwise(grad, this, (g, x) => g * 2*x));
            };
        
        return output;
    }
    
    public Tensor Power(double power)
    {
        var output = Elementwise(this, x => Math.Pow(x, power));
        if (Backpropagate != null)
            output.Backpropagate = grad => {
                Backpropagate(Elementwise(grad, this, (g, x) => g * power * Math.Pow(x, power - 1)));
            };
        
        return output;
    }
    
    public Tensor MatrixMultiply(Tensor other)
    {
        // align rows of this with columns of other and unsqueeze in order to 
        // be able to iterate over the contents of each.
        var a = Unsqueeze(rank - 1);
        var otherT = other.T();
        var b = otherT.Unsqueeze(other.rank - 2);
        
        // get shapes
        var (inputBroadcastShape, outputShape) = MultiplyShapes(a.shape, b.shape);
        
        var output = new Tensor(outputShape);
        output.noGrad = noGrad || other.noGrad;
        
        // matrix multiplication is a series of vector dot products
        var vectorLength = shape[shape.Length - 1];
        var vectorHead = 0;
        double vectorDotProduct = 0;
        
        var (indicesA, readHeadA) = a.StartPosition(inputBroadcastShape);
        var (indicesB, readHeadB) = b.StartPosition(inputBroadcastShape);
        var writeHead = 0;
        while(writeHead < output.size && readHeadA < a._data.Length && readHeadB < b._data.Length)
        {
            if (vectorHead++ == vectorLength)
            {
                output._data[writeHead++] = vectorDotProduct;
                vectorDotProduct = a._data[readHeadA] * b._data[readHeadB];
                vectorHead = 1;
            }
            else
            {
                vectorDotProduct += a._data[readHeadA] * b._data[readHeadB];
            }

            a.IncrementPosition(ref indicesA, ref readHeadA, inputBroadcastShape);
            b.IncrementPosition(ref indicesB, ref readHeadB, inputBroadcastShape);
        }
        output._data[writeHead++] = vectorDotProduct;
        
        output.Backpropagate = grad => {
            if (Backpropagate != null)
                Backpropagate(grad.MatrixMultiply(otherT));
            
            if (other.Backpropagate != null)
                other.Backpropagate(T().MatrixMultiply(grad));
        };
        return output;
    }
    
    public Tensor Mean(int dim)
    {
        var order = new int[rank];
        var newShape = new int[rank - 1];
        for (var i = 0; i < rank - 1; i++)
        {
            order[i] = (i < dim ? i : i + 1);
            newShape[i] = shape[i < dim ? i : i + 1];
        }
        order[rank - 1] = dim;
        var permuted = Permute(order);
        
        var output = new Tensor(newShape);
        
        var vectorLength = shape[dim];
        var vectorHead = 0;
        double vectorSum = 0;
        
        var (indices, readHead) = permuted.StartPosition();
        var writeHead = 0;
        while (writeHead < output.size && readHead < permuted._data.Length)
        {
            if (vectorHead++ == vectorLength)
            {
                output._data[writeHead++] = vectorSum / shape[dim];
                vectorSum = permuted._data[readHead];
                vectorHead = 1;
            }
            else
            {
                vectorSum += permuted._data[readHead];
            }
            permuted.IncrementPosition(ref indices, ref readHead);
        }
        output._data[writeHead] = vectorSum / shape[dim];
        
        if (Backpropagate != null)
            output.Backpropagate = grad => {
                Backpropagate(grad.Unsqueeze(dim, shape[dim]));
            };
        return output;
    }
    
    public void Addm_(Tensor other, double multiplier = 1)
    {
        var (indices, readHead) = StartPosition();
        var (indicesB, readHeadB) = other.StartPosition();
        var writeHead = 0;
        while(writeHead < size && readHeadB < other._data.Length)
        {
            _data[writeHead++] += other._data[readHeadB] * multiplier;
            IncrementPosition(ref indices, ref readHead, shape);
            other.IncrementPosition(ref indicesB, ref readHeadB, shape);
        }
    }
    
    public Tensor Linear(Tensor W, Tensor b)
    {
        var product = MatrixMultiply(W);
        var bb = b.BroadcastToShape(product.shape);
        return product.Plus(bb);
    }
    
    #endregion
}

class Solution
{
    static void Main(string[] args)
    {
        var t = new Tensor(2, 3, 4, 5);
        Assert(t.size == 2 * 3 * 4 * 5, "Tensor creation");

        t.FillUniform_(0, 1);
        var t2 = new Tensor(2, 3, 4, 5);
        t2.FillWithRange_();
        Assert(!t.InefficientEquals(t2), "InefficientEquals01");
        Assert(t.CloseTo(t), "CloseTo01");
        Assert(!t.CloseTo(t2), "CloseTo02");
        
        Assert(t.Reshape(-1).InefficientEquals(t.T().T().Reshape(-1)), "Reshape01");
        Assert(t.Reshape(-1).InefficientEquals(t.Reshape(6, -1, 4).Reshape(-1)), "Reshape02");

        Assert(t.InefficientEquals(t.T().T()), "Permute01");
        Assert(t.InefficientEquals(t.Permute(0, 2, 1, 3).Permute(0, 2, 1, 3)), "Permute02");
        TestCopyAndFlatten(t.Permute(0, 2, 1, 3), "Permute03");
        
        t = new Tensor(6, 4, 2, 10);
        t.FillUniform_();
        var b = t.Reshape(8, 3, 2, 5, 2);
        Assert(t.Reshape(-1).CloseTo(b.Reshape(-1)), "Reshape01");

        t = new Tensor(3, 3);
        t.FillWithRange_();
        Assert(t.Slice(0, 1, 1).ContentsAsString() == "  3, 4, 5", "Slice01");
        Assert(t.Slice(0, 1, 2).ContentsAsString() == "  3, 4, 5\n  6, 7, 8", "Slice02");
        Assert(t.Slice(1, 0, 2).ContentsAsString() == "  0, 1\n  3, 4\n  6, 7", "Slice03");
        Assert(t.Slice(1, 0, 2).Slice(0, 1, 2).ContentsAsString() == "  3, 4\n  6, 7", "Slice04");

        TestCopyAndTranspose(t.Slice(0, 1, 1), "Slice05");
        TestCopyAndTranspose(t.Slice(1, 1, 1), "Slice06");
        TestCopyAndTranspose(t.Slice(1, 0, 2), "Slice07");
        TestCopyAndTranspose(t.Slice(1, 0, 2).Slice(0, 1, 2), "Slice08");
        
        t = new Tensor(2, 1, 4, 1, 4);
        t.FillWithRange_();
        int[] squeezedAt1TestShape = {2, 4, 1, 4};
        Assert(t.Squeeze(1).shape.SequenceEqual(squeezedAt1TestShape), "Squeeze01");
        Assert(t.Squeeze(1).Reshape(-1).InefficientEquals(t.Reshape(-1)), "Squeeze02");
        
        Assert(t.Squeeze(1).Unsqueeze(1).InefficientEquals(t), "Squeeze03");
        
        Assert(t.Squeeze(1).Permute(1, 0, 2, 3).Reshape(-1).InefficientEquals(t.Permute(2, 1, 0, 3, 4).Reshape(-1)), "SqueezePermute01");
        Assert(t.Squeeze(1).Permute(1, 0, 2, 3).InefficientEquals(t.Permute(2, 1, 0, 3, 4).Squeeze(1)), "SqueezePermute01");
        Assert(t.Squeeze(1).Permute(1, 0, 2, 3).Unsqueeze(1).InefficientEquals(t.Permute(2, 1, 0, 3, 4)), "SqueezePermute01");

        TestCopyAndTranspose(t.Slice(0, 1, 1), "Slice09");
        TestCopyAndTranspose(t.Slice(2, 2, 1), "Slice10");
        TestCopyAndTranspose(t.Slice(4, 0, 2), "Slice11");
        
        Console.WriteLine("Manual tests finished");
        
        // Test many variants of slice
        t = new Tensor(60);
        t.FillWithRange_();
        
        var count = 0;
        var smallDivisorsOf60 = new int[] {1, 2, 3, 4, 5, 6};
        for (int i1 = 2; i1 <= 6; i1++)
        {
            TestCopyAndFlatten(t.Reshape(i1, -1), $"ReshapeC ({i1}, -1)");
            TestCopyAndFlatten(t.Reshape(-1, i1), $"ReshapeC (-1, {i1})");
            TestCopyAndTranspose(t.Reshape(i1, -1), $"ReshapeT ({i1}, -1)");
            TestCopyAndTranspose(t.Reshape(-1, i1), $"ReshapeT (-1, {i1})");
            TestSliceAndTranspose(t.Reshape(i1, -1), $"ReshapeS ({i1}, -1)");
            TestSliceAndTranspose(t.Reshape(-1, i1), $"ReshapeS (-1, {i1})");
            count += 4;
            for (int i2 = 2; i2 <= 6; i2++)
            {
                int missingDivisor = 60 / i1 / i2;
                if (i1 * i2 * missingDivisor != 60)
                    continue;

                TestCopyAndFlatten(t.Reshape(i1, i2, -1), $"ReshapeC ({i1}, {i2}, -1)");
                TestCopyAndFlatten(t.Reshape(i1, -1, i2), $"ReshapeC ({i1}, -1, {i2})");
                TestCopyAndFlatten(t.Reshape(-1, i1, i2), $"ReshapeC (-1, {i1}, {i2})");
                TestCopyAndTranspose(t.Reshape(i1, i2, -1), $"ReshapeT ({i1}, {i2}, -1)");
                TestCopyAndTranspose(t.Reshape(i1, -1, i2), $"ReshapeT ({i1}, -1, {i2})");
                TestCopyAndTranspose(t.Reshape(-1, i1, i2), $"ReshapeT (-1, {i1}, {i2})");
                TestSliceAndTranspose(t.Reshape(i1, i2, -1), $"ReshapeS ({i1}, {i2}, -1)");
                TestSliceAndTranspose(t.Reshape(i1, -1, i2), $"ReshapeS ({i1}, -1, {i2})");
                TestSliceAndTranspose(t.Reshape(-1, i1, i2), $"ReshapeS (-1, {i1}, {i2})");
                count += 6;
            }
        }
        
        Console.WriteLine($"{count} automated tests finished");//*/
        //*/
        TestMathOps();
        Xor();
    }

    static void TestCopyAndFlatten(Tensor t, string testId)
    {
        var a = t.Copy(false).Reshape(-1);
        var b = t.Reshape(-1);
        Assert(a.InefficientEquals(b), testId);
        Assert(a.CloseTo(b), "CloseTo_" + testId);
    }

    static void TestCopyAndTranspose(Tensor t, string testId)
    {
        var a = t.Copy(false).T();
        Assert(a.InefficientEquals(t.T()), testId);
        Assert(a.CloseTo(t.T()), "CloseTo_" + testId);
    }

    static void TestSliceAndTranspose(Tensor t, string testId)
    {
        var a = t.Slice(t.rank - 1, 1, t.shape[t.rank - 1] - 1).T();
        var b = t.T().Slice(t.rank - 2, 1, t.shape[t.rank - 1] - 1);
        Assert(a.InefficientEquals(b), testId);
        Assert(a.CloseTo(b), "CloseTo_" + testId);
    }
    
    static void Assert(bool condition, string message)
    {
        if (!condition)
            Console.WriteLine("FAILED: " + message);
    }
    
    static void TestMathOps()
    {
        var count = 0;
        
        Assert(Tensor.BroadcastShape(new[] {2, 3}, new[] {2, 3}).SequenceEqual(new[] {2, 3}), "Broadcasting 01");
        Assert(Tensor.BroadcastShape(new[] {1, 3}, new[] {2, 3}).SequenceEqual(new[] {2, 3}), "Broadcasting 02");
        Assert(Tensor.BroadcastShape(new[] {1, 3}, new[] {2, 1}).SequenceEqual(new[] {2, 3}), "Broadcasting 03");
        Assert(Tensor.BroadcastShape(new[] {2, 2, 3}, new[] {2, 3}).SequenceEqual(new[] {2, 2, 3}), "Broadcasting 04");
        count += 4;
        
        var t1 = new Tensor(2, 3);
        t1.FillWithRange_();
        Assert(t1.Square().ContentsAsString() == "  0, 1, 4\n  9, 16, 25", "Square 01");
        Assert(t1.Power(2).ContentsAsString() == "  0, 1, 4\n  9, 16, 25", "Power 01");
        Assert(t1.Power(3).ContentsAsString() == "  0, 1, 8\n  27, 64, 125", "Power 02");
        count += 3;
        
        var t2 = new Tensor(2, 3);
        t2.FillWithRange_();
        
        Assert((t1.Plus(t2)).ContentsAsString() == "  0, 2, 4\n  6, 8, 10", "PlusTensor 01");
        Assert((t1.Times(t2)).ContentsAsString() == "  0, 1, 4\n  9, 16, 25", "TimesTensor 01");
        count += 2;
        
        t2 = new Tensor(3, 2);
        t2.FillWithRange_();
        Assert(t1.MatrixMultiply(t2).ContentsAsString() == "  10, 13\n  28, 40", "MatrixMultipy 01");
        count += 1;
        
        t1 = new Tensor(2, 4);
        t1.FillWithRange_();
        t2 = new Tensor(4, 3);
        t2.FillWithRange_();
        
        Assert(t1.MatrixMultiply(t2).ContentsAsString() == "  42, 48, 54\n  114, 136, 158", "MatrixMultipy 01");
        count += 1;
        
        t1 = new Tensor(2, 3, 4);
        t1.FillWithRange_();
        t2 = new Tensor(4, 5);
        t2.FillWithRange_();
        
        Assert(t1.MatrixMultiply(t2).ContentsAsString() == "0\n  70, 76, 82, 88, 94\n  190, 212, 234, 256, 278\n  310, 348, 386, 424, 462\n1\n  430, 484, 538, 592, 646\n  550, 620, 690, 760, 830\n  670, 756, 842, 928, 1014", "MatrixMultipy 01");
        count += 1;
        
        t1 = new Tensor(2, 3);
        t1.FillWithRange_();
        t2 = new Tensor(4, 3, 5);
        t2.FillWithRange_();
        
        Assert(t1.MatrixMultiply(t2).ContentsAsString()  == "0\n  25, 28, 31, 34, 37\n  70, 82, 94, 106, 118\n1\n  70, 73, 76, 79, 82\n  250, 262, 274, 286, 298\n2\n  115, 118, 121, 124, 127\n  430, 442, 454, 466, 478\n3\n  160, 163, 166, 169, 172\n  610, 622, 634, 646, 658", "MatrixMultipy 01");
        count += 1;
        
        t1 = new Tensor(2, 3);
        t1.FillWithRange_();
        t1.requiresGrad = true;
        var sigmoid = t1.Sigmoid();
        sigmoid.Backward(2);
        Assert(sigmoid.ContentsAsString() == "  0.5, 0.731058578630005, 0.880797077977882\n  0.952574126822433, 0.982013790037908, 0.993307149075715", "Sigmoid 01");
        Assert(t1.grad.ContentsAsString() == "  0.5, 0.393223866482964, 0.209987170807013\n  0.090353319461824, 0.0353254124265822, 0.0132961133415801", "Sigmoid 02");
        count += 2;
        
        t1 = new Tensor(2, 3);
        t1.FillWithRange_();
        t1.requiresGrad = true;
        var relu = t1.Plus(-2).ReLU();
        relu.Backward(2);
        Assert(relu.ContentsAsString() == "  0, 0, 0\n  1, 2, 3", "ReLU 01");
        Assert(t1.grad.ContentsAsString() == "  0, 0, 0\n  2, 2, 2", "ReLU 02");
        count += 2;

        t1 = new Tensor(3, 3);
        t1.FillWithRange_();
        t1.requiresGrad = true;
        t2 = new Tensor(3, 5);
        t2.FillWithRange_();
        t2.requiresGrad = true;
        var product = t1.MatrixMultiply(t2);
        product.Backward(2);
        Assert(t2.grad.ContentsAsString() == "  18, 18, 18, 18, 18\n  24, 24, 24, 24, 24\n  30, 30, 30, 30, 30", "Matrix derivative 01");
        Assert(t1.grad.ContentsAsString() == "  20, 70, 120\n  20, 70, 120\n  20, 70, 120", "Matrix derivative 02");
        count += 2;

        t1 = new Tensor(3, 3);
        t1.FillWithRange_();
        t1.requiresGrad = true;
        t2 = t1.Square();
        t2.Backward();
        Assert(t1.grad.CloseTo(t1.Times(2)), "Square derivative should be == Times(2)");
        count += 1;
        
        t1 = new Tensor(3, 3);
        t1.FillWithRange_();
        t1.requiresGrad = true;
        t2 = t1.Mean(0);
        Assert(t2.ContentsAsString() == "  3, 4, 5", "Mean 01");
        Assert(t1.Mean(1).ContentsAsString() == "  1, 4, 7", "Mean 02");
        t2.Backward();
        Assert(t1.grad.ContentsAsString() == "  1, 1, 1\n  1, 1, 1\n  1, 1, 1", "Mean 03");
        count += 3;

        t1 = new Tensor(2, 3);
        t1.FillWithRange_();
        t1.requiresGrad = true;
        t2 = t1.Mean(0);
        Assert(t2.ContentsAsString() == "  1.5, 2.5, 3.5", "Mean 04");
        Assert(t1.Mean(1).ContentsAsString() == "  1, 4", "Mean 05");
        t2.Backward();
        Assert(t1.grad.ContentsAsString() == "  1, 1, 1\n  1, 1, 1", "Mean 06");
        count += 3;

        t1 = new Tensor(2, 3, 4);
        t1.FillWithRange_();
        t1.requiresGrad = true;
        t2 = t1.Mean(0);
        Assert(t2.ContentsAsString() == "  6, 7, 8, 9\n  10, 11, 12, 13\n  14, 15, 16, 17", "Mean 07");
        Assert(t1.Mean(1).ContentsAsString() == "  4, 5, 6, 7\n  16, 17, 18, 19", "Mean 08");
        t2.Backward();
        Assert(t1.grad.ContentsAsString() == "0\n  1, 1, 1, 1\n  1, 1, 1, 1\n  1, 1, 1, 1\n1\n  1, 1, 1, 1\n  1, 1, 1, 1\n  1, 1, 1, 1", "Mean 09");
        count += 3;
        
        Console.WriteLine($"{count} Math ops tests finished");//*/
    }
    
    static void Xor()
    {
        Stopwatch stopwatch = Stopwatch.StartNew();
        
        var inputData = new double[8] {0,0, 0,1, 1,0, 1,1};
        var targetData = new double[4] {0, 1, 1, 0};
        
        var inputs = new Tensor(inputData).Reshape(4, 2);
        var targets = new Tensor(targetData).Reshape(4, 1);
        
        var W1 = new Tensor(2, 5);
        var W2 = new Tensor(5, 1);
        W1.InitialiseWeights_(Tensor.WeightDistribution.Uniform, Tensor.Activation.ReLU, 2, 5);
        W2.InitialiseWeights_(Tensor.WeightDistribution.Uniform, Tensor.Activation.ReLU, 5, 1);
        //var W1 = new Tensor(new[] {0.45821174, 0.03672511, 0.07344309, 0.09343814, 0.74803227, 0.44467169, 0.82936714, 0.09053218, 0.47320226, 0.70096746}).Reshape(2, 5);
        //var W2 = new Tensor(new[] {0.41225452, 0.67594099, 0.42878221, 0.28043498, 0.2172279}).Reshape(5, 1);
        W1.requiresGrad = true;
        W2.requiresGrad = true;
        W1.PrintContents();
        W2.PrintContents();
        
        var b1 = new Tensor(5);
        var b2 = new Tensor(1);
        b1.requiresGrad = true;
        b2.requiresGrad = true;
        
        const double learningRate = 1;
        
        Tensor layer2s = null;
        
        for (var epoch = 0; epoch < 1000; epoch++)
        {
            Console.Write($"\rEpoch {epoch}");
            
            var layer1 = inputs.Linear(W1, b1);
            var layer1s = layer1.Sigmoid();
            var layer2 = layer1s.Linear(W2, b2);
            layer2s = layer2.Sigmoid();
            
            var error = layer2s.Minus(targets);
            
            // insert error directly
            //layer2s.Backpropagate(error);
            
            // use mean squared error
            var loss = error.Square().Mean(0);
            loss.Backward();
            
            W1.Addm_(W1.grad, -learningRate);
            W2.Addm_(W2.grad, -learningRate);
            W1.ClearGrads();
            W2.ClearGrads();
            
            b1.Addm_(b1.grad, -learningRate);
            b2.Addm_(b2.grad, -learningRate);
            b1.ClearGrads();
            b2.ClearGrads();
        }
        Console.WriteLine("");
        Console.WriteLine("Output");
        layer2s.PrintContents();
        stopwatch.Stop();
        Console.WriteLine($"Time {stopwatch.Elapsed}");
        // 1.5 times as fast as numpy
        // nearly 3 times as fast as pytorch
    }
}