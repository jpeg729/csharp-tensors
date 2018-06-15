using System;
using System.Linq;
using System.IO;
using System.Runtime.CompilerServices;

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
    
    public readonly int[]  shape;
    public readonly bool   contiguous;
    public bool            requires_grad;
    
    private readonly int   _base_stride;
    private readonly int[] _strides;
    private readonly int   _start;
    private double[]       _data;
    private double[]       _grad;
    
    public int size { get { return shape.Aggregate((a, x) => a * x); } }
    public int rank { get { return shape.Length; } }
    
    public delegate void BackwardsMethod(double[] grad);
    public BackwardsMethod Backwards;
    
    #endregion
    
    #region Constructors
    
    public Tensor(int[] shape, double[] data = null, bool requires_grad = false)
    {
        this.shape = shape;
        this.requires_grad = requires_grad;
        if (data != null && data.Length != size)
        {
            throw new ArgumentException(
                $"Shape {String.Join(",", shape)} = size {size} is incompatible with the size of the data {data.Length}");
        }
        _data = data ?? new double[size];
        _start = 0;
        _base_stride = 1;
        _strides = Tensor.MakeStrides(shape, _base_stride);
        contiguous = true;
        Detach();
    }
    
    public Tensor(int[] shape, double fill_value, bool requires_grad = false)
    {
        this.shape = shape;
        this.requires_grad = requires_grad;
        _data = new double[size];
        Array.Fill(_data, fill_value);
        _start = 0;
        _base_stride = 1;
        _strides = Tensor.MakeStrides(shape, _base_stride);
        contiguous = true;
        Detach();
    }
    
    public Tensor(double scalar, bool requires_grad = false)
    {
        shape = new int[] { 1 };
        this.requires_grad = requires_grad;
        _data = new double[] { scalar };
        _start = 0;
        _base_stride = 1;
        _strides = new int[] { 1 };
        contiguous = true;
        Detach();
    }
    
    private Tensor(double[] data, int[] shape, int base_stride, int[] strides, int start, 
        BackwardsMethod backwards, bool contiguous = true
        )
    {
        this.shape = shape;
        requires_grad = false;
        _data = data;
        _base_stride = 1;
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
    
    #region Custom initialisers
    
    public static Tensor Full(int[] shape, double value, bool requires_grad = false)
    {
        var size = shape.Aggregate((a, x) => a * x);
        var data = new double[size];
        Array.Fill(data, value);
        
        return new Tensor(shape, data, requires_grad);
    }
    
    public static Tensor FullWithCount(int[] shape, bool requires_grad = false)
    {
        var size = shape.Aggregate((a, x) => a * x);
        var data = new double[size];
        
        for (var i = 0; i < size; i++)
            data[i] = i;
        
        return new Tensor(shape, data, requires_grad);
    }
    
    public static Tensor Uniform(int[] shape, double minval, double maxval, bool requires_grad = false)
    {
        var size = shape.Aggregate((a, x) => a * x);
        var data = new double[size];
        var random = new Random();
        
        for (var i = 0; i < size; i++)
            data[i] = minval + random.NextDouble() * (maxval - minval);
        
        return new Tensor(shape, data, requires_grad);
    }
    
    public static Tensor Normal(int[] shape, double mean = 0, double std = 1, bool requires_grad = false)
    {
        var size = shape.Aggregate((a, x) => a * x);
        var data = new double[size];
        var random = new Random();
        
        for (var i = 0; i < size; i++)
        {
            // Box-Muller transform
            var uniform1 = random.NextDouble();
            var uniform2 = random.NextDouble();
            var distance = Math.Sqrt(-2.0 * Math.Log(uniform1));
            var angle = 2.0 * Math.PI * uniform2;
            
            data[i] = mean + std * distance * Math.Sin(angle);
            
            if (++i < size)
                data[i] = mean + std * distance * Math.Cos(angle);
        }
        
        return new Tensor(shape, data, requires_grad);
    }
    
    #endregion
    
    #region Display and debug
    
    public override string ToString()
    {
        if (size == 1)
        {
            return $"Tensor of shape ({String.Join(",", shape)}) containing value {_data[_start]}";
        }
        return $"Tensor of shape ({String.Join(",", shape)}), total size {size}";
    }
    
    public void Describe()
    {
        Console.Error.WriteLine(this.ToString());
        Console.Error.WriteLine($"Strides {_base_stride}, ({String.Join(",", _strides)})");
        Console.Error.WriteLine($"Start {_start}, data size {_data.Length}, contiguous {contiguous}");
    }
    
    public void PrintContents()
    {
        Console.Error.WriteLine(this.ToString());
        
        var indices = new int[rank];
        var data_index = _start;
        var matrix_row = new double[shape[rank - 1]];
        var row_head = 0;
        while (data_index < _data.Length)
        {
            if (rank > 2 && indices[rank - 2] == 0 && indices[rank - 1] == 0)
                Console.Error.WriteLine($"{String.Join(",", indices.Take(rank - 2))}");
            
            matrix_row[row_head++] = _data[data_index];
            
            if (row_head == shape[rank - 1])
            {
                row_head = 0;
                Console.Error.WriteLine($"  {String.Join(", ", matrix_row)}");
            }
            IncrementPosition(ref indices, ref data_index, shape);
        }
    }
    
    #endregion
    
    #region Indexing and broadcasting
    
    public double this[params int[] indices] { get { return _data[DataIndex(indices)]; } }
    
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
    private void IncrementPosition(ref int[] indices, ref int data_index, int[] desired_shape)
    {
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
        Backwards = grad => {
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
            };
    }
    
    #endregion
    
    #region Basic tensor ops, Copy, View, Slice, etc.
    
    public Tensor Copy()
    {
        var new_data = new double[size];
        if (contiguous)
        {
            Array.Copy(_data, _start, new_data, 0, size);
            return new Tensor(shape, new_data);
        }
        
        int[] indices = new int[rank];
        int read_head = _start;
        
        for (var write_head = 0; write_head < new_data.Length; write_head++)
        {
            new_data[write_head] = _data[read_head];
            IncrementPosition(ref indices, ref read_head, shape);
        }
        return new Tensor(shape, new_data);
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
    
    public Tensor Permute(int[] order)
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
    
    public Tensor View(int[] shape)
    {
        var wildcard_index = Array.IndexOf(shape, -1);
        if (wildcard_index != Array.LastIndexOf(shape, -1))
        {
            throw new ArgumentException("Too many unknowns in shape");
        }
        
        var new_size = shape.Aggregate((a, x) => a * x);
        
        if (wildcard_index >= 0)
        {
            int value = size / -new_size;
            shape[wildcard_index] = value;
            new_size *= -value;
        }
        
        if (size != new_size)
        {
            throw new ArgumentException("Given shape is incompatible with the size of the Tensor");
        }
        
        if (!contiguous)
        {
            return Copy().View(shape); // In some cases, copying isn't absolutely necessary.
        }
        var new_strides = Tensor.MakeStrides(shape, _base_stride);
        return new Tensor(_data, shape, _base_stride, new_strides, _start, grad => this.Backwards(grad));
    }
    
    public Tensor Slice(int dim, int start, int length = 1)
    {
        if (start >= shape[dim])
        {
            throw new ArgumentException("Given start incompatible with the shape of the Tensor");
        }
        var new_start = _start + start * _strides[dim];
        var new_shape = new int[rank];
        shape.CopyTo(new_shape, 0);
        new_shape[dim] = length;
        return new Tensor(_data, new_shape, _base_stride, _strides, new_start, grad => this.Backwards(grad));
    }
    
    public Tensor Squeeze(int dim)
    {
        if (shape[dim] != 1)
        {
            return this;
        }
        var new_shape = new int[rank - 1];
        var new_strides = new int[rank - 1];
        for (var i = 0; i < rank - 1; i++)
        {
            new_shape[i] = shape[i < dim ? i : i + 1];
            new_strides[i] = _strides[i < dim ? i : i + 1];
        }
        var new_base_stride = _base_stride;
        if (dim == rank - 1)
        {
            new_base_stride *= _strides[rank - 1];
        }
        return new Tensor(_data, new_shape, new_base_stride, new_strides, _start, grad => this.Backwards(grad));
    }
    
    public Tensor Unsqueeze(int dim)
    {
        if (dim > rank)
        {
            throw new ArgumentException($"Can't unsqueeze dim {dim} in shape ({String.Join(",", shape)})");
        }
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
    
    #region Math ops
    #endregion
}

class Solution
{
    static void Main(string[] args)
    {
        int[] shape = { 2, 3, 4 };
        double[] data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };

        var t1 = new Tensor(shape, data);
        t1.PrintContents();
        
        t1.T().PrintContents();
        
        int[] order = {1, 0, 2};
        t1.Permute(order).PrintContents();
        
        int[] new_shape = {1, 1, -1, 4};
        t1.View(new_shape).PrintContents();
    }
}