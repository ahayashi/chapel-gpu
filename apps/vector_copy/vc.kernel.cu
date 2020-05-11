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

extern "C" {
#ifndef USE_LAMBDA    
    void LaunchVC(float* dA, float *dB, int N) {
        vc<<<ceil(((float)N)/1024), 1024>>>(dA, dB, N);
    }
#else
    void LaunchVC(float *dA, float *dB, int N) {
        call_gpu_functor(N, 1024, NULL, [=] __device__ (int i) { dA[i] = dB[i]; });
    }
#endif    
}
