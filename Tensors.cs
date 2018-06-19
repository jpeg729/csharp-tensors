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
 * - why _base_stride? and is it truly necessary?
 *
 * - Equality
 *
 * - Tests
 *   - which methods alter _start, shape and _strides
 *   - test them comprehensively
 *
 * - Question: how to get _data out efficiently without making it public?
 *   - tensor[coords] accessor for one element at a time...
 *   - DangerouslyAccessData_() which returns the data array by reference
 *
 *
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
/// var t1 = Tensor(shape1, data1, requires_grad=true);
/// var t2 = Tensor(shape2, data2, requires_grad=false);
///
/// var t3 = t1.plus(t2);
/// var loss = t3.loss_function(target_values);
/// loss.Backwards(); // this should backpropagate all the way back to the inputs.
///
/// var t1_gradient = t1.grad;
/// </code>
/// </example>
class Tensor
{
    #region Properties

    public readonly int[]  shape = new int[] {1};
    public readonly bool   contiguous = true;
    public bool            requires_grad = false;

    private readonly int   _base_stride = 1;
    private readonly int[] _strides = new int[] {1};
    private readonly int   _start = 0;
    private double[]       _data;
    private double[]       _grad;

    public int size { get { return shape.Aggregate((a, x) => a * x); } }
    public int rank { get { return shape.Length; } }

    public delegate void BackwardsMethod(double[] grad);
    public BackwardsMethod Backwards;

    #endregion

    #region Constructors

    public Tensor(double[] data)
    {
        this.shape = new int[] {data.Length};
        _data = data;
        Detach();
    }

    public Tensor(params int[] shape)
    {
        this.shape = shape;
        _data = new double[size];
        _strides = Tensor.MakeStrides(shape, _base_stride);
        Detach();
    }

    public Tensor(double scalar)
    {
        _data = new double[] { scalar };
        Detach();
    }

    private Tensor(double[] data, int[] shape, int base_stride, int[] strides, int start,
        BackwardsMethod backwards, bool contiguous = true)
    {
        this.shape = shape;
        _data = data;
        _base_stride = base_stride;
        _strides = strides ?? Tensor.MakeStrides(shape, _base_stride);
        _start = start;
        var natural_order = true;
        for (var i = 0; i < rank - 1; i++)
        {
            if (_strides[i] != 0 && _strides[i] <= _strides[i + 1])
            {
                natural_order = false;
                break;
            }
        }
        this.contiguous = natural_order && _strides[0] * shape[0] == size;
        Detach();
    }

    private static int[] MakeStrides(int[] shape, int base_stride)
    {
        // This function may be better off in a Utils class.
        var strides = new int[shape.Length];
        var stride = base_stride;
        for (var i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    #endregion

    #region Inplace fill operations

    private void WarnAboutInplaceModification()
    {
        if (Backwards != DefaultBackwards)
        {
            Console.WriteLine("WARNING inplace modification of tensor contents can mess up the results of automatic differentiation.");
            Console.WriteLine(Environment.StackTrace);
        }
    }

    public void Fill_(double value)
    {
        Array.Fill(_data, value, _start, size);
        WarnAboutInplaceModification();
    }

    public void FillWithCount_()
    {
        for (var i = 0; i < size; i++)
            _data[i + _start] = i;

        WarnAboutInplaceModification();
    }

    public void Uniform_(double minval = 0, double maxval = 1)
    {
        var random = new Random();

        for (var i = 0; i < size; i++)
            _data[i + _start] = minval + random.NextDouble() * (maxval - minval);

        WarnAboutInplaceModification();
    }

    public void Normal_(double mean = 0, double std = 1)
    {
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

        WarnAboutInplaceModification();
    }

    #endregion

    #region Display and debug

    public override string ToString()
    {
        return $"Tensor of shape ({String.Join(",", shape)}) " +
            (size == 1 ? $"containing value {_data[_start]}" : $"total size {size}");
    }

    public void ImplementationDetails()
    {
        Console.Error.WriteLine(this);
        Console.Error.WriteLine($"Strides {_base_stride}, ({String.Join(",", _strides)})");
        Console.Error.WriteLine($"Start {_start}, data size {_data.Length}, contiguous {contiguous}");
    }

    public void PrintContents()
    {
        Console.Error.WriteLine(this);
        Console.Error.WriteLine(ContentsAsString());
    }

    public string ContentsAsString()
    {
        var output = new List<string>();
        var indices = new int[rank];
        var data_index = _start;
        var matrix_row = new double[shape[rank - 1]];
        var row_head = 0;
        while (data_index < _data.Length)
        {
            if (rank > 2 && indices[rank - 2] == 0 && indices[rank - 1] == 0)
                output.Add($"{String.Join(",", indices.Take(rank - 2))}");

            matrix_row[row_head++] = _data[data_index];

            if (row_head == shape[rank - 1])
            {
                row_head = 0;
                output.Add($"  {String.Join(", ", matrix_row)}");
            }
            IncrementPosition(ref indices, ref data_index, shape);
        }
        return String.Join("\n", output);
    }

    #endregion

    #region Indexing and broadcasting

    // single element indexing via coordinates
    public double this[params int[] indices] { get { return _data[DataIndex(indices)]; } }

    public double[] DangerouslyAccessData_()
    {
        return _data;
    }

    public bool BroadcastableTo(int[] desired_shape)
    {
        int[] shape_to_imitate;
        if (desired_shape.Length > rank)
        {
            shape_to_imitate = new int[rank];
            Array.Copy(desired_shape, desired_shape.Length - rank, shape_to_imitate, 0, rank);
        }
        else
        {
            shape_to_imitate = desired_shape;
        }

        if (shape_to_imitate.Length != rank)
        {
            throw new ArgumentException(
                $"Bad desired shape {String.Join(",", desired_shape)} for broadcasting to from {String.Join(",", shape)}.");
        }

        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] > 1 && shape[i] != shape_to_imitate[i])
                return false;
        }
        return true;
    }

    /// <summary>
    /// Use this function for looping over a tensor.
    ///
    /// indices should be initialised to all zeros.
    /// data_index should be initialised to _start.
    ///
    /// First check that this tensor is BroadcastableTo(desired_shape).
    ///
    /// Use of ref serves as a reminder that this function mutates those inputs.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void IncrementPosition(ref int[] indices, ref int data_index, int[] desired_shape = null)
    {
        if (desired_shape == null)
            desired_shape = shape;

        for (var dim = indices.Length - 1; dim >= 0; dim--)
        {
            indices[dim] += 1;
            if (indices[dim] < desired_shape[dim])
            {
                if (shape[dim] > 1) // Only modify data_index if we are not broadcasting along dim.
                {
                    data_index += _strides[dim];
                }
                break;
            }
            else
            {
                indices[dim] = 0;
                if (shape[dim] > 1)
                {
                    data_index -= _strides[dim] * (shape[dim] - 1);
                }
                if (dim == 0) // End of indexable data.
                {
                    data_index = _data.Length;
                }
            }
        }
    }

    /// <summary>
    /// Given an array of indices, find the address of the underlying data element.
    /// </summary>
    private int DataIndex(int[] indices)
    {
        var data_index = _start;
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

            data_index += index * _strides[i];
        }
        return data_index;
    }

    #endregion

    #region Autograd

    /// <summary>
    /// Sets the Backwards delegate to a default one that either accumulates
    /// gradients in _grad if requires_grad, or does nothing.
    ///
    /// This stops backpropagation from going any further back through the
    /// computation graph.
    /// </summary>
    public void Detach()
    {
        Backwards = DefaultBackwards;
    }

    public void DefaultBackwards(double[] grad)
    {
        if (!requires_grad)
            return;

        if (_grad == null)
            _grad = new double[_data.Length];

        if (contiguous)
        {
            Array.Copy(grad, 0, _grad, _start, size);
        }
        else
        {
            int[] indices = new int[rank];
            int write_head = _start;
            foreach (var element in grad)
            {
                _grad[write_head] = element;
                IncrementPosition(ref indices, ref write_head, shape);
            }
        }
    }

    #endregion

    #region Basic tensor ops, Copy, View, Slice, etc.

    private void ShowWarningAndStackTrace(string warning)
    {
        if (Backwards != DefaultBackwards)
        {
            Console.WriteLine("WARNING: " + warning);
            Console.WriteLine(Environment.StackTrace);
        }
    }

    public Tensor Copy(bool warnIfContiguous = true)
    {
        var new_data = new double[size];
        if (contiguous)
        {
            if (warnIfContiguous)
                ShowWarningAndStackTrace("Copying contiguous tensor data, is this necessary?");

            Array.Copy(_data, _start, new_data, 0, size);
            return new Tensor(new_data).View(shape);
        }

        int[] indices = new int[rank];
        int read_head = _start;

        for (var write_head = 0; write_head < new_data.Length; write_head++)
        {
            new_data[write_head] = _data[read_head];
            IncrementPosition(ref indices, ref read_head, shape);
        }
        return new Tensor(new_data).View(shape);
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
        var new_shape = new int[rank];
        var new_strides = new int[rank];
        for (var i = 0; i < rank; i++)
        {
            new_shape[i] = shape[order[i]];
            new_strides[i] = _strides[order[i]];
        }
        return new Tensor(_data, new_shape, _base_stride, new_strides, _start, grad => this.Backwards(grad));
    }

    public Tensor View(params int[] shape)
    {
        var wildcard_index = Array.IndexOf(shape, -1);
        if (wildcard_index != Array.LastIndexOf(shape, -1))
            throw new ArgumentException("Too many unknowns in shape");

        var new_size = shape.Aggregate((a, x) => a * x);

        if (wildcard_index >= 0)
        {
            int value = size / -new_size;
            shape[wildcard_index] = value;
            new_size *= -value;
        }

        if (!contiguous)
        {
            // each new dimension must either exactly match an existing dimension,
            // or the product of consecutive new dimensions must match a contiguous
            // section of the existing tensor dimensions.
            var new_shape_i = 0;
            var shape_i = 0;
            var new_dim_size = shape[0];
            var dim_size = this.shape[0];
            while (new_shape_i <= shape.Length && shape_i <= this.shape.Length)
            {
                if (new_dim_size == dim_size)
                {
                    new_dim_size = shape[++new_shape_i];
                    dim_size = this.shape[++shape_i];
                }
                else if (new_dim_size < dim_size && new_shape_i < shape.Length - 1)
                {
                    new_dim_size *= shape[++new_shape_i];
                }
                else if (dim_size < new_dim_size && shape_i < this.shape.Length - 1)
                {
                    if (_strides[shape_i] != _strides[shape_i + 1] * shape[shape_i + 1])
                        throw new ArgumentException($"View({String.Join(",", shape)}) tensor is not contiguous enough. Run Contiguous() first.");
                    dim_size *= this.shape[++shape_i];
                }
            }
        }

        if (size != new_size)
            throw new ArgumentException($"View({String.Join(",", shape)}) is incompatible with tensor shape ({String.Join(",", this.shape)})");

        //if (!contiguous)
        //    throw new ArgumentException($"View({String.Join(",", shape)}) tensor is not contiguous enough. Run Contiguous() first.");

        var new_strides = Tensor.MakeStrides(shape, _base_stride);
        return new Tensor(_data, shape, _base_stride, new_strides, _start, grad => this.Backwards(grad));
    }

    public Tensor Slice(int dim, int start, int length = 1)
    {
        if (start + length > shape[dim])
            throw new ArgumentException($"Slice({dim}, {start}, {length}) incompatible with shape {String.Join(",", shape)}");

        var new_start = _start + start * _strides[dim];
        var new_shape = new int[rank];
        shape.CopyTo(new_shape, 0);
        new_shape[dim] = length;
        return new Tensor(_data, new_shape, _base_stride, _strides, new_start, grad => this.Backwards(grad));
    }

    public Tensor Squeeze(int dim)
    {
        if (shape[dim] > 1)
            throw new ArgumentException($"Squeeze({dim}) cannot squeeze dimension of size {shape[dim]}");

        var new_shape = new int[rank - 1];
        var new_strides = new int[rank - 1];
        for (var i = 0; i < rank - 1; i++)
        {
            new_shape[i] = shape[i < dim ? i : i + 1];
            new_strides[i] = _strides[i < dim ? i : i + 1];
        }
        var new_base_stride = _base_stride;
        if (dim == rank - 1)
            new_base_stride *= _strides[rank - 1];

        return new Tensor(_data, new_shape, new_base_stride, new_strides, _start, grad => this.Backwards(grad));
    }

    public Tensor Unsqueeze(int dim)
    {
        if (dim > rank)
            throw new ArgumentException($"Can't unsqueeze dimension {dim} in shape ({String.Join(",", shape)})");

        var new_shape = new int[rank + 1];
        var new_strides = new int[rank + 1];
        for (var i = 0; i < rank; i++)
        {
            new_shape[i < dim ? i : i + 1] = shape[i];
            new_strides[i < dim ? i : i + 1] = _strides[i];
        }
        new_shape[dim] = 1;
        new_strides[dim] = 1;
        return new Tensor(_data, new_shape, _base_stride, new_strides, _start, grad => this.Backwards(grad));
    }

    #endregion

    #region Tests

    public bool CloseTo(Tensor other, double tolerance = 1e-8)
    {
        int[] indices = new int[rank];
        int read_head = _start;

        int[] indices_other = new int[other.rank];
        int read_head_other = other._start;

        while(read_head < _data.Length && read_head_other < other._data.Length)
        {
            if (Math.Abs(_data[read_head] - other._data[read_head_other]) > tolerance)
                return false;

            IncrementPosition(ref indices, ref read_head, shape);
            other.IncrementPosition(ref indices_other, ref read_head_other, other.shape);
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
        Debug.Assert(t.size == 2 * 3 * 4 * 5, "Tensor creation");

        t.Uniform_(0, 1);
        Debug.Assert(t.View(-1).InefficientEquals(t.T().T().View(-1)), "View01");
        Debug.Assert(t.View(-1).InefficientEquals(t.View(6, -1, 4).View(-1)), "View02");

        Debug.Assert(t.InefficientEquals(t.T().T()), "Permute01");
        Debug.Assert(t.InefficientEquals(t.Permute(0, 2, 1, 3).Permute(0, 2, 1, 3)), "Permute02");
        TestCopyAndFlatten(t.Permute(0, 2, 1, 3), "Permute03");

        t = new Tensor(3, 3);
        t.FillWithCount_();
        Debug.Assert(t.Slice(0, 1, 1).ContentsAsString() == "Tensor of shape (1,3), total size 3\n  3, 4, 5", "Slice01");
        Debug.Assert(t.Slice(0, 1, 2).ContentsAsString() == "Tensor of shape (2,3), total size 6\n  3, 4, 5\n  6, 7, 8", "Slice02");
        Debug.Assert(t.Slice(1, 0, 2).ContentsAsString() == "Tensor of shape (3,2), total size 6\n  0, 1\n  3, 4\n  6, 7", "Slice03");
        Debug.Assert(t.Slice(1, 0, 2).Slice(0, 1, 2).ContentsAsString() == "Tensor of shape (2,2), total size 4\n  3, 4\n  6, 7", "Slice04");

        TestCopyAndTranspose(t.Slice(0, 1, 1), "Slice05");
        TestCopyAndTranspose(t.Slice(1, 1, 1), "Slice06");
        TestCopyAndTranspose(t.Slice(1, 0, 2), "Slice07");
        TestCopyAndTranspose(t.Slice(1, 0, 2).Slice(0, 1, 2), "Slice08");

        t = new Tensor(2, 1, 4, 1, 4);
        t.FillWithCount_();
        int[] squeezedAt1TestShape = {2, 4, 1, 4};
        Debug.Assert(t.Squeeze(1).shape == squeezedAt1TestShape, "Squeeze01");
        Debug.Assert(t.Squeeze(1).View(-1).InefficientEquals(t.View(-1)), "Squeeze02");
        Debug.Assert(t.Squeeze(1).Unsqueeze(1).InefficientEquals(t), "Squeeze03");
        Debug.Assert(t.Squeeze(1).Permute(1, 0, 3, 4).View(-1).InefficientEquals(t.Permute(2, 1, 0, 3, 4).View(-1)), "SqueezePermute01");
        Debug.Assert(t.Squeeze(1).Permute(1, 0, 3, 4).InefficientEquals(t.Permute(2, 1, 0, 3, 4).Squeeze(1)), "SqueezePermute01");
        Debug.Assert(t.Squeeze(1).Permute(1, 0, 3, 4).Unsqueeze(1).InefficientEquals(t.Permute(2, 1, 0, 3, 4)), "SqueezePermute01");

        TestCopyAndTranspose(t.Slice(0, 1, 1), "Slice09");
        TestCopyAndTranspose(t.Slice(2, 2, 1), "Slice10");
        TestCopyAndTranspose(t.Slice(4, 0, 2), "Slice11");

        Console.WriteLine("Tests finished");
    }

    static void TestCopyAndFlatten(Tensor t, string testId)
    {
        var a = t.Copy().View(-1);
        Debug.Assert(a.InefficientEquals(t.View(-1)), testId);
    }

    static void TestCopyAndTranspose(Tensor t, string testId)
    {
        var a = t.Copy().T();
        Debug.Assert(a.InefficientEquals(t.T()), testId);
    }
}