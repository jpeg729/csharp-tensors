A Tensor class intended to work somewhat like a cross between numpy and pytorch,
with the goal of building and training neural networks.

### Copy on write

Most of these methods return a new Tensor that, where possible, shares the 
underlying data array with the original Tensor.

### Tensor creation

```c#
var tensor_of_zeros = new Tensor(dimensions, as, args, or, as, list);

double[] data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
var tensor_of_data = new Tensor(data);
```

If you need to create a tensor from data with a specified shape, then just `Reshape()` it.

### Tensor filling

These methods modify the underlying data array, which is against the copy-on-write 
approach used by the rest of the library. The method names end with an underscore
as a reminder of that fact. These methods will issue a warning in some circumstances.

These methods return `void` and cannot be chained.

* `Fill_(value)`
* `FillWithRange_(start, step)` fills the tensor with the numbers start, start + step, start + 2*step
* `FillUniform_(min, max)` uniformly distributed random numbers
* `FillNormal_(mean, std)` normally distributed random numbers

### String representation and printing

* `ToString()` returns a string containing tensor shape and size.
* `PrintContents()` prints the _entire_ contents of the tensor to the console.
* `PrintImplementationDetails()` prints the size of the tensor and its strides, etc.
* `ContentsAsString()` returns the _entire_ contents of the tensor as a string
  formatted in the same way `PrintContents()` would print it.

### Basic tensor operation methods

When an operation need only modify the strides used, the underlying flat data array
is shared with the new Tensor. This is similar to the concept of views in numpy.

* `Copy()` - always copies the data, regardless of whether the tensor is contiguous or not.
* `T()` - switches the last two dimensions.
* `Permute(new, order, as, args, or, array)` - reorders the dimensions.
* `Reshape(new, dims, as, args, or, array)` - changes the shape.
  N.B. `Reshape(..)` will throw an error if the tensor is not contiguous enough.
* `Squeeze(dim)` - removes the given dimension, if the length of that dimension is 1.
* `Unsqueeze(dim)` - inserts a new dimension of length 1.
* `Slice(dim, start, count)` - takes `count` elements from `start` along dimension `dim`.

<details>
  <summary>Expand for examples</summary>

```c#
var t1 = new Tensor(2, 3, 4);
t1.FillWithRange_();
t1.T().PrintContents();

// Tensor of shape (2,4,3), total size 24
// 0
//   1, 5, 9
//   2, 6, 10
//   3, 7, 11
//   4, 8, 12
// 1
//   13, 17, 21
//   14, 18, 22
//   15, 19, 23
//   16, 20, 24

int[] order = {1, 0, 2};
t1.Permute(order).PrintContents();

// Tensor of shape (3,2,4), total size 24
// 0
//   1, 2, 3, 4
//   13, 14, 15, 16
// 1
//   5, 6, 7, 8
//   17, 18, 19, 20
// 2
//   9, 10, 11, 12
//   21, 22, 23, 24

int[] new_shape = {1, 1, -1, 4};
t1.Reshape(new_shape).PrintContents();

// Tensor of shape (1,1,6,4), total size 24
// 0,0
//   1, 2, 3, 4
//   5, 6, 7, 8
//   9, 10, 11, 12
//   13, 14, 15, 16
//   17, 18, 19, 20
//   21, 22, 23, 24
```
</details>

## Accessing data

* `tensor[coords, of, element]` allows getting, but not setting the value of the
  element at the specified coordinates.
* `DangerouslyAccessData_()` returns a reference to the underlying data array,
  use with caution.

## Automatic differentiation

All math operations should return a Tensor containing the result of the calculation
with a Backwards delegate that calculates the gradient and calls Backwards on all 
the tensors that were given as inputs to the operation.

## TODO 

* [ ] Add unit tests
* [ ] Add math operations
