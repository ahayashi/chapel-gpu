#include <stdio.h>

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 1024
#endif

__global__ void myKernel(int64_t *dA, size_t N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        dA[id] = dA[id] + 1;
    }
}

extern "C" {
    void kernel(int64_t *ptr, size_t N) {
        myKernel<<<ceil(((float)N)/THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(ptr, N);
    }
}