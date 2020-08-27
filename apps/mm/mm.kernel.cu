#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#ifdef __NVCC__
#include <cublas_v2.h>
#endif

__global__ void mm(float *dA, float *dB, float *dC, int DIM, int N, int GPUN) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id <= GPUN) {
        int i = id / DIM;
        int j = id % DIM;
        float sum = 0.0f;
        for (int k = 0; k < DIM; k++) {
            sum += dA[i*DIM+k] * dB[k*DIM+j];
        }
        dC[id] += sum;
    }
}

extern "C" {

    void LaunchMM(float *A, float *B, float *C, int N, int low, int hi, int GPUN, int tiled) {
        if (GPUN > 0) {
            assert(hi - low + 1 == GPUN);
#ifdef VERBOSE
            printf("In mmCUDA\n");
            printf("\t GPUN: %d\n", GPUN);
            printf("\t range: %d..%d\n", start, end);
#endif
            if (!tiled) {
                mm<<<ceil(((float)GPUN)/1024), 1024>>>(A, B, C, ceil(sqrt(N)), N, GPUN);
            }
            else if(tiled == 1) {
                printf("Tile not imlemented\n");
                assert(false);
            }
            else {
#ifdef __NVCC__
                printf("Using cublas\n");
                cublasHandle_t handle;

                cublasCreate(&handle);
                float alpha = 1.0F;
                float beta = 0.0F;
                int lda = sqrt(N), ldb = sqrt(N), ldc = sqrt(N);

                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sqrt(N), GPUN/sqrt(N), sqrt(N), &alpha, B, ldb, A, lda, &beta, C, ldc);
#endif
            }
        }
    }

}
