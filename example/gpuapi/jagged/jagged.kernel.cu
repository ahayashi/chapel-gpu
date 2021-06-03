#include<stdio.h>
__global__ void myKernel(int64_t **dA) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 256*(i+1); j++) {
            dA[i][j] = dA[i][j] + 1;
        }
    }
}

extern "C" {
    void kernelLOW(int64_t **hPtrs, size_t *hPtrSizes, int64_t N) {
        int64_t **dA = (int64_t**)malloc(sizeof(int64_t*)*N);
        for (int i = 0; i < N; i++) {
            cudaMalloc(&dA[i], hPtrSizes[i]*sizeof(int64_t));
            cudaMemcpy(dA[i], hPtrs[i], hPtrSizes[i]*sizeof(int64_t), cudaMemcpyHostToDevice);
        }
        int64_t **dAs;
        cudaMalloc(&dAs, sizeof(int64_t*)*N);
        cudaMemcpy(dAs, dA, sizeof(int64_t*)*N, cudaMemcpyHostToDevice);

        myKernel<<<1,1>>>(dAs);
        cudaDeviceSynchronize();
        for (int i = 0; i < N; i++) {
            cudaMemcpy(hPtrs[i], dA[i], hPtrSizes[i]*sizeof(int64_t), cudaMemcpyDeviceToHost);
        }
    }

    void kernelMIDLOW(int64_t **dAs, int64_t N) {
        myKernel<<<1,1>>>(dAs);
        cudaDeviceSynchronize();
    }
}