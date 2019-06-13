# GPUIterator

## Summary
A primary goal of this module is to provide an appropriate interface between Chapel and accelerator programs such that expert accelerator programmers can explore different variants in a portable way (i.e., CPU-only, GPU-only, X% for CPU + Y% for GPU on a single or multiple CPU+GPU node(s)). To address these challenges, here we introduce a Chapel module, ```GPUIterator```, which can be invoked in a ```forall``` loop like this:
```chapel
// original forall
forall i in 1..n { }

// The GPUIterator
forall i in GPU(1..n, GPUWrapper, CPUPercent) { }
```

## Motivation
Chapel allows expert GPU programmers to develop manually prepared GPU programs that can be callable from a Chapel program. This can be done by invoking CUDA/OpenCL programs using [the C interoperability feature](https://chapel-lang.org/docs/master/technotes/extern.html).

To understand this, consider the following baseline ```forall``` implementation that performs vector copy:
```chapel
// Chapel file
var A: [1..n] real(32);
var B: [1..n] real(32);
forall i in 1..n {
  A(i) = B(i);
}
```

The equivalent Chapel+GPU code is shown below:
```chapel
// Chapel file
extern proc GPUfunc(A: [] real(32), B: [] real(32),
                      lo: int, hi: int);

var A: [1..n] real(32);
var B: [1..n] real(32);
GPUfunc(A, B, 1, n);
```

```c
// Separate C file
void GPUfunc(float *A, float *B, int start, int end) {
  // GPU Implementation (CUDA/OpenCL API)
  // 1. device memory allocations
  // 2. host-to-device data transfers
  // 3. GPU kernel compilations (if needed)
  // 4. GPU kernel invocations
  // 5. device-to-host data transfers
  // 6. clean up
  // Note: A[0] and B[0] here corresponds to
  // A(1) and B(1) in the Chapel part respectively
}
```

The key difference is that the original ```forall``` loop is replaced with the function call to the native function that should include typical host and device operations including device memory allocations, data transfers, and kernel invocations.

Unfortunately, the source code is not very portable particularly when the user wants to explore different variants to get higher performance. Since GPUs are not always faster than CPUs (and vice versa), the user has to be juggling ```forall``` with ```GPUfunc()``` depending on the data size and the complexity of the computation (e.g., by commenting in/out each version). One intuitive workaround is to put an if statement to decide whether to use which version (CPUs or GPUs). However, this raises another problem: it is still not very portable when doing 1) multi-locale CPU+GPU execution, and 2) further advanced workload distributions such as hybrid execution of the CPU and GPU versions, the latter of which could give additional performance improvement for a certain class of applications and platforms.

One may argue that it is still technically possible to do so at the user-level. For multi-locale GPU execution, we could do like this with appropriate arguments to ```GPUfunc``` - i.e., a local portion of a distributed array, and a subspace of original iteration space ```coforall loc in Locales { on loc { GPUfunc(...); } }```. For hybrid CPU+GPU execution, one could create c tasks and g tasks that take care of a subspace of the original iteration space per locale, where c and $g$ are the numbers of CPUs and GPUs. However, that is what we want to let the ```GPUIterator``` to reduce the complexity of the user-level code.

## How to Use the GPUIterator
Here is an example code of the GPUIterator:

```chapel
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
```

```c
// Separate C file
void GPUfunc(float *A, float *B, int start, int end, int n) {
  // GPU Implementation (CUDA/OpenCL API)
  // 1. device memory allocations
  // 2. host-to-device data transfers
  // 3. GPU kernel compilations (if needed)
  // 4. GPU kernel invocations
  // 5. device-to-host data transfers
  // 6. clean up
  // Note: A[0] and B[0] here corresponds to
  // A(1) and B(1) in the Chapel part respectively
}
```

You need to 1) import the GPUIterator module, 2) create a wrapper function (```GPUWrapper```) which is a callback function invoked after the module has created a task for the GPU portion of the iteration space (```lo```, ```hi```, ```n```) and eventually invokes the GPU function (```GPUfunc```), 3) then wrap the iteration space using ```GPU()``` with the wrapper function ```GPUWrapper```. Note that the last argument (```CPUPercent```), the percentage of the iteration space will be executed on the CPU, is optional. The default number for it is zero, meaning the whole itreration space goes to the GPU side.

It is worth noting that the GPUIterator gives freedom to you of designing ```GPUfunc()```. In addition to the automatically computed numbers (```lo```, ```hi```, and ```n```), you are required to give appropriate arguments so that the GPU part can work properly. We will discuss how to write the GPU part below.

Also, currently you need to use [our Chapel compiler](https://github.com/ahayashi/chapel/tree/gpu-iterator) that includes the GPU locale model tailored for this module. Define```CHPL_LOCALE_MODEL=gpu``` when compiling a Chapel program with ```GPUIterator```.

## How to Compile Your Chapel Programs with the GPUIterator
Here we explain how to compile a Chapel program with the GPUIterator. In the following, ```$CHPL_GPU_HOME``` represents the top directory of this repository and we take the vector copy code in ```$CHPL_GPU_HOME/apps/vector_copy``` as an example.

### Create an object file with a GPU compiler
First, create an object file for the GPU program by compiling it with the ```-c``` option:
```bash
$ cd $CHPL_GPU_HOME/apps/vector_copy
// CUDA
$ nvcc -O3 -arch sm_60 -std=c++11 -c vc.cu -o vc.gpu.o
// OpenCL
$ gcc -O3 -c -std=c++11 -c vc.opencl.c -o vc.gpu.o
```

### Create an executable with the Chapel compiler
Then, compile the Chapel program (```vc.hybrid.chpl```) with the object file. You will also required to give the path to the GPUIterator module with the ```-M``` option unless it is in the module search path [$CHPL_MODULE_PATH](https://chapel-lang.org/docs/master/technotes/module_search.html):
```bash
// CUDA
$ CHPL_LOCALE_MODEL=gpu chpl --fast -M $CHPL_GPU_HOME/chapel-gpu/modules vc.hybrid.chpl vc.gpu.o -lcudart -lcuda
// OpenCL
$ CHPL_LOCALE_MODEL=gpu chpl --fast -M $CHPL_GPU_HOME/chapel-gpu/modules vc.hybrid.chpl vc.gpu.o -lOpenCL
```

Depending on your setting, it may be required to give the ```-L``` option to let the Chapel compiler know the location of CUDA/OpenCL libraries (e.g., ```-L/usr/local/cuda/lib64```).

### Run
Now you are ready to run the application. Since the CPU/GPU percentage is defined as [a configurable constant](https://chapel-lang.org/docs/master/users-guide/base/configs.html), you can explore different variants easily (CPU-only, GPU-only, X% for CPU + Y% for GPU).

```bash
// CPU:0%, GPU:100%
$ ./vc.hybrid
// CPU:50%, GPU:50%
$ ./vc.hybrid --CPUratio=50
// CPU:100%, GPU:0%
$ ./vc.hybrid --CPUratio=100
```

## Guide to Write GPU programs with the GPUIterator
In summary, GPU programs for the GPUIterator should include typical host and device operations including device memory allocations, data transfers, and kernel invocations, which is pretty much the same as typical CUDA/OpenCL programs with the exception that 1) input/output data are passed from/to the Chapel part, and 2) the GPU portion of the original iteration space are provided. Here is a complete example of the GPU part for the vector copy program:

```c
// Separate .cu file

// CUDA kernel for Vector Copy
__global__ void vc(float *dA, float *dB, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
	  dA[id] = dB[id];
    }
}

void GPUfunc(float *A, float *B, int start, int end, int n) {
  // GPU Implementation (CUDA/OpenCL)
  // Note: A[0] and B[0] here corresponds to
  // A(1) and B(1) in the Chapel part respectively
  assert(end - start + 1 == n)
  if (n > 0) {
    // device memory allocation
    cudaMalloc(&dA, sizeof(float) * n));
    cudaMalloc(&dB, sizeof(float) * n));

    // Optimization 1: only transferring the array B because A will be updated on the device
    cudaMemcpy(dB, B + start, sizeof(float) * n, cudaMemcpyHostToDevice);

    // kernel invocation
    vc<<<ceil(((float)n)/1024), 1024>>>(dA, dB, n);

    // wait for the completion of the kernel invocation
    cudaDeviceSynchronize();

    // Optimization 2: only transferring back the array A because B is not updated on the device
    cudaMemcpy(A + start, dA, sizeof(float) * n, cudaMemcpyDeviceToHost));

    // device memory deallocation
    cudaFree(dA);
    cudaFree(dB);
  }

}
```

Again, the code above is pretty much the same as typical CUDA programs. Thanks to the C interoperability feature, the Chapel arrays A and B can be treated as just C pointers, and they can be directly passed to CUDA API functions. Thus, all you have to do is to make sure that 1) the GPU kernel works only for the given subspace of the original iteration space, and 2) pass all the required data to the ```GPUfunc``` from the Chapel side.

Please note that the above example does data transfer optimizations where there is no host-to-device transfer of  ```A``` or device-to-host transfer of ```B```. Also, it only allocates and transfers a subarray of ```A``` and ```B``` and invokes the kernel with the subspace since they are safe to do so in this example. However, please be careful about how to optimize your GPU program because the legality of doing so depends on a kernel. For more details, please see the applications in the ```app``` directory.

## Further Readings
"GPUIterator: Bridging the Gap between Chapel and GPU Platforms", Akihiro Hayashi, Sri Raj Paul, Vivek Sarkar, The ACM SIGPLAN 6th Annual Chapel Implementers and Users Workshop (CHIUW), June 2019. (co-located with PLDI2019/ACM FCRC2019)