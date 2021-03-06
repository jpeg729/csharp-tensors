A Tensor class intended to work somewhat like a cross between numpy and pytorch.

## Why?

1. As a learning experience
2. For competitions on Codingame.com. Hence we need 
   - good single-thread performance
   - small code-size (submissions are limited to 100k file size) 
   - fairly complete functionality

### Design decisions

Data is stored in a flat array of doubles. The multi-dimensional indexing logic is 
written so that many tensor reshaping operations can be reduced to re-ordering or
modification of the strides used to access the data. Most Tensor manipulation 
operations return a new Tensor object that shares the underlying data array.

Directly modifying the data contained in a Tensor is avoided where possible in order 
to enable automatic differentiation. For efficiency, tensor manipulation operations 
generally return a new Tensor that shares the same underlying data array as the 
original Tensor.

By necessity, some methods do modify the data stored in the Tensor. These methods have
names that end in an underscore, and all of them return `void` so they can't be chained.
I may decide that being able to chain inplace modification ops leads to nicer code, in 
which case they will all return `this`.

Most of the Tensor initializers have been implemented as extension methods because 
generally we only need them when training new neural networks. Mostly I anticipate 
running pre-trained neural networks on codingame's servers and this makes it easy to 
remove the unused initialiser code.

## Status

Full rewrite in progress.

The main change has been to overhaul the indexing system. Padding has been built in, though
the whether the resulting slowdown is acceptable remains to be tested.

* Contiguity tests - needed for Reshape stuff

* Tensor manipulation functions - almost done
  * Still to do
    - `MergeDimWithNext(Count)`
    - `ReshapeDim(dim, newShape)`
  * Unless needed the following will be left out...
    - `Reshape(arbitraryShape)`
    - `Flatten()` - Slower alternative `tensor.Copy().MergeDimWithNext(tensor.rank - 1)`
  * Maybe add
    - `Stack(dim, tensor1, tensor2, ...)`
    - `Split(dim, size)`
* Basic math - done
* Conv1d - TODO. Note that we don't really need 2d convs as a 2d conv can be separated into two 1d convs 1&nbsp;x&nbsp;N then N&nbsp;x&nbsp;1. The resulting model uses fewer params, runs faster, and gives great results.
* Tests - partially done, the following need to be done
  - `CloseTo`
  - `Broadcast`
  - `Copy`
  - `Squeeze`
  - `Unsqueeze`
  - Sequential slicing - should be ok
  - Sequential padding - padding the same dim twice currently overwrites the padding
  - Slicing after padding - will fail if the slice occurs over the same dim as the padding
  - Other ops after padding - many will fail to copy the padding
  - `Elementwise`
  - Math ops

### Tensor creation

```c#
var tensorOfZeros = new Tensor(arrayOfDimSizes);

double[] data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
var tensorOfData = new Tensor(data);
```

If you need to create a tensor from data with a specified shape, then just use `ReshapeDim(0, arrayOfDimSizes)`.

### Tensor filling

These methods modify the underlying data array and will issue a warning in some circumstances.

* `Fill_(value)`
* `FillWithRange_(start, step)` fills the tensor with the numbers start, start + step, start + 2*step
* `FillUniform_(min, max)` uniformly distributed random numbers
* `FillNormal_(mean, std)` normally distributed random numbers

### String representation and printing

* `ToString()` returns a string containing tensor shape and size.

Other methods are provided in the Utils class.

* `PrintContents(tensor)` outputs the contents of the tensor to the console.
* `ContentsToString(tensor)` returns a string representation of the entire tensor.

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
* `Pad(dim, left, right, type, value)` - adds padding to the given dimension.

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

WIP

## Automatic differentiation

All math operations should return a Tensor containing the result of the calculation
with a Backwards delegate that calculates the gradient and calls Backwards on all 
the tensors that were given as inputs to the operation.

## Performance improvement ideas

### Implement a pool for data arrays

The problem is that a tensor's data array can be huge and when it is too big (> 85kbytes?)
the data array is automatically put into GC's generation 2 pool, which means collecting it
will be expensive. Besides an array is always zeroed on allocation, which isn't always
necessary.

Maintaining a pool of reusable data arrays would be a great solution.

1. by making Tensor Disposable in order to track use of said arrays
  > output = model.forward(input);
  > loss = Lossfn(output, target);
  > loss.backward();
  > loss.Dispose(); // should back up the chain disposing all calculated Tensors it finds.
    class Tensor : IDisposable {
      public void Dispose() {
        _data.referenceCount--;
        _data = null;
        GC.SuppressFinalize(this);
      }
      ~Tensor() {
        Dispose();
      }
      public double[] GetDataArray(int size) {
        returns a free data array from the pool, or makes a new one.
      }
    }
  Question:
    when the finaliser runs, has the data array already been disposed of?


2. using weak refs to the Tensors that use each array
  for each data array maintain a list of WeakReferences to the tensors that use it...
  pool = Dict<int, List<WeakReference<Tensor>>>
  pool[size].RemoveAll(w => !w.IsAlive);

3. using GCHandles directly
  WeakReferences are objects with destructors, hence expensive...
  Use list of GCHandle.Alloc(tensor, GCHandleType.Weak)
  Change target by handle.Target = otherTensor
  Note that a GCHandle must be .Free()d after use

### Independant AdvanceOffset for padded Tensors

I could create PlainAdvanceOffset and PaddedAdvanceOffset methods and an AdvanceOffset 
delegate to hold the relevant one, but calling the delegate will be more expensive than
calling a single AdvanceOffset method directly. Besides, the current AdvanceOffset method
is aggressively inlined which would be impossible with a delegate.

What about subclassing Tensor?
Could I do Tensor mytensor = new PaddedTensor(...);

### If Tensor were a struct instead of a class...

Firstly, is it possible and what changes would be necessary?
Secondly, what would be the consequences of copying the entire struct every time you pass it around?