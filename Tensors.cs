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
 * - Should gradients be Tensors, or double[] ? 
 *   Tensors, since they are easier to manipulate
 *   - But how to avoid making Backward at each op?
 *     - bool t.noGrad perhaps
 * 
 * - Math for neural nets
 * 
 * - Parallelise some ops using Environment.ProcessorCount
 * 
 * - Tests
 *   - given an order check
 *     t.Slice(d, ..).Permute(order) == t.Permute(order).Slice(order[d], ..)
 *     t.Permute(order).Slice(d, ..).Permute(UnPermuteOrder(order)) == t.Slice(order[d], ..)
 * 
 *
 * - Question: how to get _data out efficiently without making it public?
 *   - tensor[coords] gettor for one element at a time...
 *   - DangerouslyAccessData_() returns the data array by reference
 *
 * - algo strassen
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
    public bool            requiresGrad = false;
    public bool            noGrad = false;

    private readonly int[] _strides = new int[] {1};
    private readonly int   _start = 0;
    private double[]       _data;
    private double[]       _grad;

    public int size { get { return shape.Aggregate((a, x) => a * x); } }
    public int rank { get { return shape.Length; } }

    public delegate void BackwardsMethod(double[] grad);
    public BackwardsMethod Backward;

    #endregion

    #region Constructors

    public Tensor(double[] data)
    {
        this.shape = new int[] {data.Length};
        _data = data;
        Detach_();
    }

    public Tensor(params int[] shape)
    {
        this.shape = shape;
        _data = new double[size];
        _strides = Tensor.MakeStrides(shape, 1);
        Detach_();
    }

    public Tensor(double scalar)
    {
        _data = new double[] { scalar };
        Detach_();
    }

    private Tensor(double[] data, int[] shape, int[] strides, int start,
        BackwardsMethod backward)
    {
        this.shape = shape;
        _data = data;
        _strides = strides;
        _start = start;
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
        Backward = backward;
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
        if (Backward != DefaultBackward)
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
        for (var i = 0; i < size; i++)
        {
            // Box-Muller transform
            var uniform1 = random.NextDouble();
            var uniform2 = random.NextDouble();
            var distance = Math.Sqrt(-2.0 * Math.Log(uniform1));
            var angle = 2.0 * Math.PI * uniform2;

            _data[i] = mean + std * distance * Math.Sin(angle);

            if (++i < size)
                _data[i + _start] = mean + std * distance * Math.Cos(angle);
        }
    }

    #endregion

    #region Display and debug

    public override string ToString()
    {
        return $"Tensor of shape ({String.Join(",", shape)}) " +
            (size == 1 ? $"containing value {_data[_start]}" : $"total size {size}");
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

        for (var dim = indices.Length - 1; dim >= 0; dim--)
        {
            indices[dim] += 1;
            if (indices[dim] < desiredShape[dim])
            {
                if (shape[dim] > 1) // Only modify dataIndex if we are not broadcasting along dim.
                {
                    dataIndex += _strides[dim];
                }
                break;
            }
            else
            {
                indices[dim] = 0;
                if (shape[dim] > 1)
                {
                    dataIndex -= _strides[dim] * (shape[dim] - 1);
                }
                if (dim == 0) // End of indexable data.
                {
                    dataIndex = _data.Length;
                }
            }
        }
    }
    
    public (int[] indices, int head) StartPosition()
    {
        return (new int[rank], _start);
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

    /// <summary>
    /// Returns a new Tensor with the default Backward delegate.
    ///
    /// If this new Tensor is used in calculations which are then differentiated,
    /// then backpropagation stops here.
    /// </summary>
    public Tensor Detach()
    {
        return new Tensor(_data, shape, _strides, _start, DefaultBackward);
    }
    
    /// <summary>
    /// Sets the Backward delegate to a default one that either accumulates
    /// gradients in _grad if requiresGrad, or does nothing.
    ///
    /// If this Tensor is used in calculations which are then differentiated,
    /// then backpropagation will not flow back beyond this Tensor.
    /// </summary>
    public void Detach_()
    {
        Backward = DefaultBackward;
    }

    public void DefaultBackward(double[] grad)
    {
        if (!requiresGrad)
            return;

        if (_grad == null)
            _grad = new double[_data.Length];

        if (contiguous && naturalOrder)
        {
            Array.Copy(grad, 0, _grad, _start, size);
        }
        else
        {
            var (indices, writeHead) = StartPosition();
            foreach (var element in grad)
            {
                _grad[writeHead] = element;
                IncrementPosition(ref indices, ref writeHead, shape);
            }
        }
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
        if (contiguous && naturalOrder)
        {
            if (warnIfContiguous)
                ShowWarningAndStackTrace("Copying contiguous tensor data, is this necessary?");

            Array.Copy(_data, _start, newData, 0, size);
            return new Tensor(newData).Reshape(shape);
        }

        var (indices, readHead) = StartPosition();

        for (var writeHead = 0; writeHead < newData.Length; writeHead++)
        {
            newData[writeHead] = _data[readHead];
            IncrementPosition(ref indices, ref readHead, shape);
        }
        return new Tensor(newData).Reshape(shape);
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
        BackwardsMethod permuteGrads = DefaultBackward;
        if (requiresGrad || Backward != DefaultBackward)
        {
            permuteGrads = grad => this.Backward(grad); // TODO unpermute grads
        }
        return new Tensor(_data, newShape, newStrides, _start, permuteGrads);
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
        //Console.WriteLine($"\nReshape({String.Join(",", shape)}) from ({String.Join(",", this.shape)})");
        //PrintImplementationDetails();
        
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
        //Console.WriteLine($"New size {newSize} = shape ({String.Join(",", shape)})");
        
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
                    //Console.WriteLine("same size dims");
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
                    //Console.WriteLine("new dim smaller");
                    newShapeIndex += 1;
                    newDimSize *= newShape[newShapeIndex];
                    newStrides[newShapeIndex] = newStrides[newShapeIndex - 1] / newShape[newShapeIndex];
                }
                else if (oldDimSize < newDimSize)
                {
                    //Console.WriteLine("old dim smaller");
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
                //Console.WriteLine($"newI {newShapeIndex} {newDimSize} oldI {oldShapeIndex} {oldDimSize} -> {String.Join(",", newStrides)}");
            }
            //Console.WriteLine($" -> {String.Join(",", newStrides)}");
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

        BackwardsMethod reshapeGrads = DefaultBackward;
        if (requiresGrad || Backward != DefaultBackward)
        {
            // reshapeGrads = grad => this.Backward(grad.Reshape(orig_shape));
            reshapeGrads = grad => this.Backward(grad); // TODO
        }
        return new Tensor(_data, shape, newStrides, _start, reshapeGrads);
    }

    public Tensor Squeeze(int dim = -1)
    {
        var newShape = new List<int>();
        var newStrides = new List<int>();
        if (dim == -1)
        {
            for (var i = 0; i < rank; i++)
            {
                if (shape[i] > 1)
                {
                    newShape.Add(shape[i]);
                    newStrides.Add(_strides[i]);
                }
            }
        }
        else
        {
            if (shape[dim] > 1)
                throw new ArgumentException($"Squeeze({dim}) cannot squeeze dimension of size {shape[dim]}");

            for (var i = 0; i < rank; i++)
            {
                if (i != dim)
                {
                    newShape.Add(shape[i]);
                    newStrides.Add(_strides[i]);
                }
            }
        }
        BackwardsMethod unsqueezeGrads = DefaultBackward;
        if (requiresGrad || Backward != DefaultBackward)
        {
            /*unsqueezeGrads = grad => {
                foreach (var dim in squeezedDims)
                    grad = grad.Unsqueeze(dim);
                
                return this.Backward(grad);
            }*/
            unsqueezeGrads = grad => this.Backward(grad); // TODO
        }

        return new Tensor(_data, newShape.ToArray(), newStrides.ToArray(), _start, unsqueezeGrads);
    }

    public Tensor Unsqueeze(int dim)
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
        newShape[dim] = 1;
        newStrides[dim] = newStrides[dim - 1];
        
        BackwardsMethod squeezeGrads = DefaultBackward;
        if (requiresGrad || Backward != DefaultBackward)
        {
            // squeezeGrads = grad => this.Backward(grad.Squeeze(dim));
            squeezeGrads = grad => this.Backward(grad); // TODO
        }
        return new Tensor(_data, newShape, newStrides, _start, squeezeGrads);
    }

    public Tensor Slice(int dim, int start, int length = 1)
    {
        if (start + length > shape[dim])
            throw new ArgumentException($"Slice({dim}, {start}, {length}) incompatible with shape {String.Join(",", shape)}");

        var newStart = _start + start * _strides[dim];
        var newShape = new int[rank];
        shape.CopyTo(newShape, 0);
        newShape[dim] = length;
        
        BackwardsMethod padGrads = DefaultBackward;
        if (requiresGrad || Backward != DefaultBackward)
        {
            // padGrads = grad => this.Backward(grad.Pad(dim, start - 1, shape[dim] - start - length));
            padGrads = grad => this.Backward(grad); // TODO
        }
        return new Tensor(_data, newShape, _strides, newStart, padGrads);
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

    #region Math ops
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
        
        t = new Tensor(6, 4, 10, 2).T();
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
        //*/
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
    }

    static void TestCopyAndFlatten(Tensor t, string testId)
    {
        //Console.WriteLine("TestCopyAndFlatten "+testId);
        //t.PrintImplementationDetails();
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
    
    static void TestPermute(Tensor t)
    {
        var order = new int[t.rank];
        for (var i = 0; i < order.Length; i++)
            order[i] = i;
        
        foreach (var permutation in Permutate(order, order.Length))
        {
            
        }
    }
    
    static void Assert(bool condition, string message)
    {
        if (!condition)
            Console.WriteLine("FAILED: " + message);
    }
    
    static void Xor()
    {
        var input = new double[8] {0,0, 0,1, 1,0, 1,1};
        var output = new double[4] {0, 1, 1, 0};
        
    }
}