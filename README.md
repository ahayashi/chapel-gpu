# GPUIterator

## Summary
A primary goal of this module is to provide an appropriate interface between Chapel and accelerator programs such that expert accelerator programmers can explore different variants in a portable way (i.e., CPU-only, GPU-only, X% for CPU + Y% for GPU on a single or multiple CPU+GPU node(s)). To address these challenges, we introduce a Chapel module, ```GPUIterator```, which provides the capability of creating and distributing tasks across a single or multiple CPU+GPU node(s).

## Motivation
Chapel allows expert GPU programmers to develop manually prepared GPU programs that can be callable from a Chapel program. This can be done by invoking CUDA/OpenCL programs using [the C interoperability feature](https://chapel-lang.org/docs/master/technotes/extern.html).

To understand this, consider the following baseline forall implementation that performs vector copy:
```chapel
// Chapel file
var A: [1..n] real(32);
var B: [1..n] real(32);
forall i in 1..n {
  A(i) = B(i);
}
```

The equivalent Chapel+GPU code is shown below:
```
// Chapel file
extern proc GPUfunc(A: [] real(32), B: [] real(32),
                      lo: int, hi: int, N: int);

var A: [1..n] real(32);
var B: [1..n] real(32);
GPUfunc(A, B, 1, n);

// Separate C file
void GPUfunc(float *A, float *B, int start, int end) {
  // GPU Implementation (CUDA/OpenCL)
  // Note: A[0] and B[0] here corresponds to
  // A(1) and B(1) in the Chapel part respectively
}
```

The key difference is that the original ```forall``` loop is replaced with the function call to the native function that should include typical host and device operations including device memory allocations, data transfers, and kernel invocations.

Unfortunately, the source code is not very portable particularly when the user wants to explore different variants to get higher performance. Since GPUs are not always faster than CPUs (and vice versa), the user has to be juggling ```forall``` with ```GPUfunc()``` depending on the data size and the complexity of the computation (e.g., by commenting in/out each version). One intuitive workaround is to put an if statement to decide whether to use which version (CPUs or GPUs). However, this raises another problem: it is still not very portable when doing 1) multi-locale CPU+GPU execution, and 2) further advanced workload distributions such as hybrid execution of the CPU and GPU versions, the latter of which could give additional performance improvement for a certain class of applications and platforms.

One may argue that it is still technically possible to do so at the user-level. For multi-locale GPU execution, we could do ```coforall loc in Locales { on loc { GPUfunc(...); } }``` with appropriate arguments to ```GPUfunc``` - i.e., a local portion of a distributed array, and a subspace of original iteration space. For hybrid CPU+GPU execution, one could create $c$ tasks and $g$ tasks that take care of a subspace of the original iteration space per locale, where $c$ and $g$ are the numbers of CPUs and GPUs. However, that is what we want to let the ```GPUIterator``` do to reduce the complexity of the user-level code.


## How to Use the GPUIterator
Here is an example code of the GPUIterator:

```
use GPUIterator;

extern proc GPUfunc(A: [] real(32), B: [] real(32),
                      lo:int, hi: int, N: int);

var A: [1..n] real(32);
var B: [1..n] real(32);

// Users need to prepare a callback function which is
// invoked after the GPUIterator has computed the GPU portion
var GPUWrapper = lambda (lo:int, hi: int, n: int) {
  GPUfunc(A, B, lo, hi, n);
};
var CPUPercent = 50; // CPUPercent is optional
forall i in GPU(1..n, GPUWrapper, CPUPercent) {
  // CPU code
  A(i) = B(i);
}

// Separate C file
void GPUfunc(float *A, float *B, int start, int end, int n) {
  // GPU Implementation (CUDA/OpenCL)
  // Note: A[0] and B[0] here corresponds to
  // A(1) and B(1) in the Chapel part respectively
}
```
You need to 1) import the GPUIterator module, 2) create a wrapper function (```GPUWrapper```) which is a callback function invoked after the module has created a task for the GPU portion of the iteration space (```lo```, ```hi```, ```n```) and eventually invokes the GPU function (```GPUfunc```), 3) then wrap the iteration space using ```GPU()``` with the wrapper function ```GPUWrapper```. Note that the last argument (```CPUPercent```), the percentage of the iteration space will be executed on the CPU, is optional. The default number for it is zero, meaning the whole itreration space goes to the GPU side.

Also, currently you need to use [our Chapel compiler](https://github.com/ahayashi/chapel/tree/gpu-iterator) that includes the GPU locale model tailored for this module. Define```CHPL_LOCALE_MODEL=gpu``` when compiling a Chapel program with ```GPUIterator```.

## Further Readings
"GPUIterator: Bridging the Gap between Chapel and GPU Platforms", Akihiro Hayashi, Sri Raj Paul, Vivek Sarkar, The ACM SIGPLAN 6th Annual Chapel Implementers and Users Workshop (CHIUW), June 2019. (co-located with PLDI2019/ACM FCRC2019)