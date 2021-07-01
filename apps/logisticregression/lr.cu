#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 1024
#endif

#define CUDA_ERROR_CHECK
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

__global__ void kernel1(float *dW, float *dWcurr, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        dWcurr[id] = dW[id];
    }
}

__global__ void kernel2(float *dW, float *dWcurr, float *dX, float *dY, float alpha, int nSamples, int nFeatures, int start, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        float err = 0.0;
        for (int s = 0; s < nSamples; s++) {
            float arg = 0.0;
            for (int f = 0; f < nFeatures; f++) {
                arg += dWcurr[f] * dX[s * (nFeatures) + f];
            }
            float hypo = 1 / (1 + exp(-arg));
            err += (hypo - dY[s]) * dX[s * (nFeatures) + start + id];
        }
        dW[id] = dWcurr[start + id] - alpha * err;
    }
}

extern "C" {
    void lrCUDA1(float *W, float *Wcurr, int start, int end, int GPUN) {
        float *dW, *dWcurr;
        if (GPUN > 0) {
            assert(end - start + 1 == GPUN);
#ifdef VERBOSE
            printf("In lrCUDA1\n");
            printf("\t GPUN: %d\n", GPUN);
            printf("\t range: %d..%d\n", start, end);
#endif
            CudaSafeCall(cudaMalloc(&dW, sizeof(float) * GPUN));
            CudaSafeCall(cudaMalloc(&dWcurr, sizeof(float) * GPUN));

            CudaSafeCall(cudaMemcpy(dW, W + start, sizeof(float) * GPUN, cudaMemcpyHostToDevice));
            kernel1<<<ceil(((float)GPUN)/THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(dW, dWcurr, GPUN);

            CudaSafeCall(cudaDeviceSynchronize());
            CudaSafeCall(cudaMemcpy(Wcurr + start, dWcurr, sizeof(float) * GPUN, cudaMemcpyDeviceToHost));

            CudaSafeCall(cudaFree(dW));
            CudaSafeCall(cudaFree(dWcurr));
        }
    }

    void lrCUDA2(float* X, float *Y, float *W, float *Wcurr, float alpha, int nSamples, int nFeatures, int start, int end, int GPUN) {
        float *dX, *dY, *dW, *dWcurr;
        if (GPUN > 0) {
            assert(end - start + 1 == GPUN);
#ifdef VERBOSE
            printf("In lrCUDA2\n");
            printf("\t GPUN: %d\n", GPUN);
            printf("\t range: %d..%d\n", start, end);
#endif
            CudaSafeCall(cudaMalloc(&dX, sizeof(float) * nSamples * nFeatures));
            CudaSafeCall(cudaMalloc(&dY, sizeof(float) * nSamples));
            CudaSafeCall(cudaMalloc(&dWcurr, sizeof(float) * nFeatures));
            CudaSafeCall(cudaMalloc(&dW, sizeof(float) * GPUN));

            CudaSafeCall(cudaMemcpy(dX, X, sizeof(float) * nSamples * nFeatures, cudaMemcpyHostToDevice));
            CudaSafeCall(cudaMemcpy(dY, Y, sizeof(float) * nSamples, cudaMemcpyHostToDevice));
            CudaSafeCall(cudaMemcpy(dWcurr, Wcurr, sizeof(float) * nFeatures, cudaMemcpyHostToDevice));

            kernel2<<<ceil(((float)GPUN)/THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(dW, dWcurr, dX, dY, alpha, nSamples, nFeatures, start-1, GPUN);
            CudaSafeCall(cudaDeviceSynchronize());
            CudaSafeCall(cudaMemcpy(W, dW, sizeof(float) * GPUN, cudaMemcpyDeviceToHost));

            CudaSafeCall(cudaFree(dX));
            CudaSafeCall(cudaFree(dY));
            CudaSafeCall(cudaFree(dW));
            CudaSafeCall(cudaFree(dWcurr));
        }
    }
}
