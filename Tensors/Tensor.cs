using System;
using System.Text;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Tensors
{
	public enum Padding { Same, Const, }; // TODO Reflect
	public enum Distribution {Uniform, Normal};
	public enum Activation {Linear, ReLU, Sigmoid, Tanh};

	public class Tensor
	{
		private static System.Random _rng;
		static void Seed(Int32 seed) { _rng = new Random(seed); }

		static Tensor()
		{
			_rng = new Random();
		}

		private static double[] MakeDataArray(int size)
		{
			return new double[size];
		}

		private class Dimension {
			public int size, stride, index, padLeft, padRight;
			public Padding padType;
			public double padValue;
		}

		private double[] _data;
		private bool _rented = true;
		private int _start;
		private int _offset;
		private Dimension[] _dims;
		private Stack<double> _paddingValues;

		private int[] _shape;
		public int[] shape {
			get {
				if (_shape == null)
					_shape = _dims.Select(s => s.size).ToArray();
				return _shape;
			}
		}
		public string shapeStr { get { return "(" + String.Join(",", _dims.Select(s => s.size)) + ")"; } }
		public readonly int size;
		public int rank { get { return _dims.Count(); } }
		public int lastIndexUpdated {
			get { return lastIndexUpdated; } 
			private set { lastIndexUpdated = value; }
		}
		public int[] indices { get { return _dims.Select(s => s.index).ToArray(); } }
		public double item {
			get {
				if (_paddingValues.Count() > 0)
					return _paddingValues.Peek();
				return _data[_offset];
			}
			private set { _data[_offset] = value; }
		}

		public Tensor grad;
		public bool noGrad;
		private bool _requiresGrad;

		private Action<Tensor> _Backpropagate;
		public Action<Tensor> Backpropagate {
			get { return _Backpropagate; }
			set { _Backpropagate = noGrad ? (Action<Tensor>)null : value; }
		}

		public Tensor(double[] data)
		{
			size = data.Length;
			_dims = new Dimension[] { new Dimension{ size = size } };
			_data = data;
			_rented = false;
		}

		public Tensor(params int[] sizes)
		{
			size = sizes.Aggregate((a, x) => a * x);
			_dims = new Dimension[sizes.Length];
			var currStride = 1;
			for (var i = 0; i < sizes.Length; i++)
			{
				_dims[i] = new Dimension{
					size = sizes[i],
					stride = sizes[i] > 1 ? currStride : 0,
				};
				currStride *= sizes[i];
			}
			_data = MakeDataArray(size);
			// TODO TEST assert size == currStride
		}

		private Tensor(double[] data, Dimension[] dims, int start, Action<Tensor> Backpropagate, bool noGrad)
		{
			_data = data;
			_dims = dims;
			size = shape.Aggregate((a, x) => a * x);
			_start = start;
			this.Backpropagate = Backpropagate;
			this.noGrad = noGrad;
		}

		public void ResetOffset()
		{
			_offset = _start;
			for (var i = 0; i < _dims.Count(); i++)
				_dims[i].index = 0;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public bool AdvanceOffset()
		{
			for (var i = _dims.Count() - 1; i >= 0; i--)
			{
				var dim = _dims[i];
				dim.index += 1;
				if (dim.index <= dim.padLeft)
				{
					if (dim.padType == Padding.Const)
						_paddingValues.Push(dim.padValue);
					lastIndexUpdated = i;
					return true;
				}
				if (dim.index < dim.size - dim.padRight)
				{
					_offset += dim.stride;
					if (dim.padType == Padding.Const)
						_paddingValues.Pop();
					lastIndexUpdated = i;
					return true;
				}
				if (dim.index < dim.size)
				{
					if (dim.padType == Padding.Const)
						_paddingValues.Push(dim.padValue);
					lastIndexUpdated = i;
					return true;
				}
				if (dim.padType == Padding.Const)
					_paddingValues.Pop();
				_offset -= (dim.size - 1) * dim.stride;
				dim.index = 0;
			}
			lastIndexUpdated = -1;
			return false;
		}

		public bool CloseTo(Tensor other, double tolerance = 1e-8)
		{
			if (!shape.SequenceEqual(other.shape))
				throw new ArgumentException($"Attempt to compare incompatible tensors {shapeStr} != {other.shapeStr}");

			ResetOffset();
			other.ResetOffset();
			do { if (Math.Abs(item - other.item) > tolerance) return false; }
			while (AdvanceOffset() && other.AdvanceOffset());
			return true;
		}

		public void Fill_(double value)
		{
			WarnAboutInplaceModification();
			ResetOffset();
			do { item = value; } while (AdvanceOffset());
		}

		public void FillWithRange_(double start = 0, double step = 1)
		{
			WarnAboutInplaceModification();
			ResetOffset();
			do { item = start; start += step; }
			while (AdvanceOffset());
		}

		public void FillEye_()
		{
			// TODO
		}

		public void FillUniform_(double minval = 0, double maxval = 1)
		{
			WarnAboutInplaceModification();
			ResetOffset();
			do { item = minval + _rng.NextDouble() * (maxval - minval); }
			while (AdvanceOffset());
		}

		public void FillNormal_(double mean = 0, double std = 1)
		{
			WarnAboutInplaceModification();

			// increase std to compensate for truncation at 2*std
			// .8796 = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
			std /= .8796;

			do // The Box-Muller transform gives two values each time
			{
				var distance = Math.Sqrt(-2.0 * Math.Log(_rng.NextDouble()));
				var angle = 2.0 * Math.PI * _rng.NextDouble();

				// Discard values that are more than 2*std away from the mean.
				var randomBit = distance * Math.Sin(angle);
				if (Math.Abs(randomBit) <= 2)
				{
					item = mean + std * randomBit;
					if (!AdvanceOffset())
						break;
				}
				randomBit = distance * Math.Cos(angle);
				if (Math.Abs(randomBit) <= 2)
					item = mean + std * randomBit;
			} while (AdvanceOffset());
		}

		public void InitialiseWeights_(Distribution dist, Activation activation, int fan, double activationParam = 0)
		{
			// Choosing fan_in preserves the magnitude of the variance of the weights in the forward pass. 
			// Choosing fan_out preserves the magnitudes in the backwards pass.
			double gain = 1;
			if (activation == Activation.ReLU)
				gain = Math.Sqrt(2 / (1 + activationParam));
			else if (activation == Activation.Tanh)
				gain = 1.1;

			if (dist == Distribution.Uniform)
			{
				var limit = Math.Sqrt(3.0 / fan);
				Console.WriteLine($"InitialiseWeights using limit {limit}");
				FillUniform_(-limit, limit);
			}
			else if (dist == Distribution.Normal)
			{
				var std = Math.Sqrt(1.0 / fan);
				Console.WriteLine($"InitialiseWeights using std {std}");
				FillNormal_(0, std);
			}
		}

		public override string ToString()
		{
			return $"Tensor of shape {shapeStr}" +
				(size == 1 ? $" containing value {_data[_start]}" : "");
		}

		private void ShowWarningAndStackTrace(string warning)
		{
			Console.WriteLine("WARNING: " + warning);
			Console.WriteLine(Environment.StackTrace);
		}

		private void WarnAboutInplaceModification()
		{
			if (Backpropagate != null && Backpropagate != StoreGrads)
				ShowWarningAndStackTrace("Inplace modification of tensor contents can mess up the results of automatic differentiation.");
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

			if (this.grad == null)
			{
				this.grad = grad;
			}
			else
			{
				this.grad.AddM_(grad);
			}
		}

		public void ClearGrads()
		{
			grad = null;
		}

		public Tensor Detach()
		{
			return new Tensor(_data, _dims, _start, null, noGrad);
		}

		public static void Broadcast(Tensor inA, Tensor inB, ref Tensor outA, ref Tensor outB)
		{
			var broadcastRank = Math.Max(inA.rank, inB.rank);
			var newShapeA = new Dimension[broadcastRank];
			var newShapeB = new Dimension[broadcastRank];
			var broadcastDimsA = new List<int>();
			var broadcastDimsB = new List<int>();

			for (var i = 0; i < broadcastRank; i++)
			{
				var idxA = i - broadcastRank + inA.rank;
				var idxB = i - broadcastRank + inB.rank;
				if (i < broadcastRank - inA.rank)
				{
					newShapeA[i] = new Dimension{ size = inB._dims[idxB].size };
					newShapeB[i] = inB._dims[idxB];
					broadcastDimsA.Add(i);
				}
				else if (i < broadcastRank - inB.rank)
				{
					newShapeA[i] = inA._dims[idxA];
					newShapeB[i] = new Dimension{ size = inA._dims[idxA].size };
					broadcastDimsB.Add(i);
				}
				else if (inA._dims[idxA].size == inB._dims[idxB].size)
				{
					newShapeA[i] = inA._dims[idxA];
					newShapeB[i] = inB._dims[idxB];
				}
				else if (inA._dims[idxA].size == 1)
				{
					newShapeA[i] = new Dimension{ size = inB._dims[idxB].size };
					newShapeB[i] = inB._dims[idxB];
					broadcastDimsA.Add(i);
				}
				else if (inB._dims[idxB].size == 1)
				{
					newShapeA[i] = inA._dims[idxA];
					newShapeB[i] = new Dimension{ size = inA._dims[idxA].size };
					broadcastDimsB.Add(i);
				}
				else
				{
					throw new ArgumentException($"Trying to broadcast incompatible shapes {inA.shapeStr} and {inB.shapeStr}");
				}
			}
			Action<Tensor> unBroadcastA = null, unBroadcastB = null;
			if (inA.Backpropagate != null)
			{
				unBroadcastA = grad => {
					foreach (var dim in broadcastDimsA)
						grad = grad.Mean(dim);

					inA.Backpropagate(grad);
				};
			}
			if (inB.Backpropagate != null)
			{
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
			ResetOffset();
			do { output.item = item; }
			while (AdvanceOffset() && output.AdvanceOffset());
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
				newShape[i] = new Dimension{
					size = _dims[order[i]].size,
					stride = _dims[order[i]].stride,
				};
			
			Action<Tensor> permuteGrads = null;
			if (Backpropagate != null)
				permuteGrads = grad => Backpropagate(grad.Permute(UnPermuteOrder(order)));

			return new Tensor(_data, newShape, _start, permuteGrads, noGrad);
		}

		public int[] UnPermuteOrder(params int[] order)
		{
			var output = new int[order.Length];
			for (var i = 0; i < order.Length; i++)
				output[order[i]] = i;
			return output;
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
					newShape[i < dim ? i : i - 1] = new Dimension{
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
			newShape[dim] = new Dimension{ size = newSize };
			for (var i = 0; i < rank; i++)
				newShape[i < dim ? i : i + 1] = new Dimension{
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
			newShape[dim] = new Dimension{
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
			newShape[dim] = new Dimension{
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
			a.ResetOffset();
			do { output.item = calcFn(a.item); }
			while (output.AdvanceOffset() && a.AdvanceOffset());
			return output;
		}

		public static Tensor Elementwise(Tensor a, Tensor b, Func<double, double, double> calcFn)
		{
			var output = new Tensor(a.shape);
			output.noGrad = a.noGrad || b.noGrad;
			Tensor c = null, d = null;
			Broadcast(a, b, ref c, ref d);
			c.ResetOffset();
			d.ResetOffset();
			do { output.item = calcFn(c.item, d.item); }
			while (output.AdvanceOffset() && c.AdvanceOffset() && d.AdvanceOffset());
			return output;
		}

		public static Tensor Elementwise(Tensor a, Tensor b, Tensor c, Func<double, double, double, double> calcFn)
		{
			var output = new Tensor(a.shape);
			output.noGrad = a.noGrad || b.noGrad || c.noGrad;
			a.ResetOffset();
			b.ResetOffset();
			do { output.item = calcFn(a.item, b.item, c.item); }
			while (output.AdvanceOffset() && a.AdvanceOffset() && b.AdvanceOffset() && c.AdvanceOffset());
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
			var output = Elementwise(this, x => x*scalar);
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
					Backpropagate(Elementwise(grad, output, (g, o) => (g * (1 - o*o))));
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
			for (var i = 0; i < rank - 1; i++)
			{
				order[i] = (i < dim ? i : i + 1);
				newShape[i] = _dims[i < dim ? i : i + 1].size;
			}
			order[rank - 1] = dim;
			var permuted = Permute(order);
			var output = new Tensor(newShape);
			var lastIndexUpdated = -1;
			do
			{
				if (lastIndexUpdated == rank - 1)
				{
					output.item += permuted.item;
				}
				else
				{
					output.item /= _dims[dim].size;
					output.AdvanceOffset();
					output.item = permuted.item;
				}
			} while (permuted.AdvanceOffset());

			output.item /= _dims[dim].size;

			if (Backpropagate != null)
				output.Backpropagate = grad =>
					Backpropagate(grad.Unsqueeze(dim, _dims[dim].size));
			return output;
		}

		public Tensor MatrixMultiply(Tensor other)
		{
			var otherT = other.T();
			Tensor a = null, b = null;
			Broadcast(
				this.Unsqueeze(this.rank - 1),
				otherT.Unsqueeze(other.rank - 2),
				ref a, ref b);
			var newShape = a.shape.Take(a.rank - 1).ToArray();
			newShape[a.rank - 2] = b.shape[b.rank - 2];
			var output = new Tensor(newShape);
			var lastIndexUpdated = -1;
			do
			{
				if (lastIndexUpdated == a.rank - 1)
					output.AdvanceOffset();
				output.item += a.item * b.item;
				a.AdvanceOffset();
				lastIndexUpdated = a.lastIndexUpdated;
			} while (lastIndexUpdated >= 0 && b.AdvanceOffset());

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
			ResetOffset();
			other.ResetOffset();
			do { item += other.item * multiplier; }
			while (AdvanceOffset() && other.AdvanceOffset());
		}
	}
}
