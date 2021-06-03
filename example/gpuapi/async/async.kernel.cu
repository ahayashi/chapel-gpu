#include <stdio.h>

__global__ void myKernel(int64_t *dA) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    dA[id] = dA[id] + 1;
}

extern "C" {
    void kernel(int64_t *ptr) {
        myKernel<<<1,128>>>(ptr);
        cudaDeviceSynchronize();
    }
}