#include <stdio.h>

__global__ void myKernel(int64_t *dA, size_t dpitchInBytes) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int64_t *dA_row = (int64_t*)((char*)dA + i*dpitchInBytes);
    dA_row[j] = dA_row[j] + 1;
}

extern "C" {
    void kernel(int64_t *ptr, size_t nRows, size_t nCols, size_t dpitch) {
        myKernel<<<nRows, nCols>>>(ptr, dpitch);
    }
}