#ifndef USE_LAMBDA
__global__ void vc(float *dA, float *dB, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        dA[id] = dB[id];
    }
}
#else
#include "lambda.h"
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 1024
#endif

extern "C" {
#ifndef USE_LAMBDA
    void LaunchVC(float* dA, float *dB, int N) {
        vc<<<ceil(((float)N)/THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(dA, dB, N);
    }
#else
    void LaunchVC(float *dA, float *dB, int N) {
        call_gpu_functor(N, THREADS_PER_BLOCK, NULL, [=] __device__ (int i) { dA[i] = dB[i]; });
    }
#endif
}
