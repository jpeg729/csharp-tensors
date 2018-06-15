A Tensor class intended to work somewhat like a cross between numpy and pytorch,
with the goal of building and training neural networks.

### Tensor creation

To create a 2x3x4 tensor filled with the numbers 1 to 24.

```c#
int[] shape = { 2, 3, 4 };
double[] data = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 
    };
var t1 = new Tensor(shape, data);
t1.PrintContents();

// Tensor of shape (2,3,4), total size 24
// 0
//   1, 2, 3, 4
//   5, 6, 7, 8
//   9, 10, 11, 12
// 1
//   13, 14, 15, 16
//   17, 18, 19, 20
//   21, 22, 23, 24
```

#### Arguments

`new Tensor(shape, data, requires_grad)`
* `shape` - required array of `int` lengths for each dimension.
* `data` - optional flat array of `double`s, defaults to an array of zeros.
* `requires_grad` - optional `bool`, defaults to `false`.

The data is stored in a flat array, and strides are used to mimic multi-dimensional 
indexing.

### Custom initialisers

* `Full(shape, value, requires_grad)` returns a tensor filled with value.
* `FullWithCount(shape, requires_grad)` returns a tensor filled with a sequence of 
  positive numbers.
* `Uniform(shape, minval, maxval, requires_grad)` returns a tensor filled with random 
  numbers uniformly sampled between `minval` and `maxval`.
* `Normal(shape, mean, std, requires_grad)` returns a tensor filled from a normal 
  distribution with the specified mean and standard deviation.

### Printing

* `PrintContents()` prints the _entire_ contents of the tensor to the console.
* `Describe()` prints the size of the tensor and its strides, etc.

### Basic tensor operation methods

When an operation need only modify the strides used, the underlying flat data array
is shared with the new Tensor. This is similar to views in numpy.

* Copy() - always 
* T() - switches the last two dimensions.
* Permute(order) - allows reordering dimensions.
* View(shape) - allows changing the shape of the Tensor.
* Slice(dim, start, count) - takes `count` elements from `start` along dimension `dim`.
* Squeeze(dim) - removes the given dimension, if the length of that dimension is 1.
* Unsqueeze(dim) - inserts a new dimension of length 1.

<details>
  <summary>Expand for examples</summary>

```c#
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
t1.View(new_shape).PrintContents();

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

## Automatic differentiation

All math operations should return a Tensor containing the result of the calculation
with a Backwards delegate that calculates the gradient and calls Backwards on all 
the tensors that were given as inputs to the operation.

## TODO 

* [ ] Consider replacing View(shape) with two methods...
  * MergeDimWithNext(dim, count)
  * SplitDim(dim, new_shape_for_dim)
  
  This would be clearer, would cover all common use cases and would allow keeping the
  same underlying storage array in a few more cases.
  
* [ ] Add unit tests
* [ ] Add math operations
