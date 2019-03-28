#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <cublas_v2.h>

#define VERBOSE
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

__global__ void mm(float *dA, float *dB, float *dC, int DIM, int N, int GPUN) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id <= GPUN) {
	int i = id / DIM;
	int j = id % DIM;
	int sum = 0;
	for (int k = 0; k < DIM; k++) {
	    sum += dA[i*DIM+k] * dB[k*DIM+j];
	}
	dC[id] += sum;
    }
}

__global__ void mm_tiled(float *dA, float *dB, float *dC, int DIM, int N, int GPUN) {
    int it, jt, kt, i, j, k;
    __shared__ int sA[32][32], sB[32][32];

    // (it, jt) => the first element of a specific tile
    it = blockIdx.y * 32;
    jt = blockIdx.x * 32;

    // (i, j) => specific element
    i = it + threadIdx.y;
    j = jt + threadIdx.x;

    if (i*DIM+j <= GPUN) {
	int sum = 0;
	// per tile loop
	for (kt = 0; kt < DIM; kt += 32) {
	    // copy to shared memory
	    sA[threadIdx.y][threadIdx.x] = dA[(it+threadIdx.y)*DIM + kt + threadIdx.x];
	    sB[threadIdx.y][threadIdx.x] = dB[(kt+threadIdx.y)*DIM + jt + threadIdx.x];
	    __syncthreads();
	    
	    // two 32x32 small shared (dB[it + 0:31][kt + 0:31], dC[kt+0:31][jt + 0:31]) at this point
	    for (k = kt; k < kt+32; k++) {
		sum += sA[i-it][k-kt] * sB[k-kt][j-jt];
	    }
	    
	    __syncthreads();
	}
	dC[i*DIM+j] = sum;
    }
}

extern "C" {
    void mmCUDA(float* A, float *B, float *C, int N, int start, int end, int GPUN, int tiled) {
	float *dA, *dB, *dC;
	if (GPUN > 0) {
	    assert(end - start + 1 == GPUN);
#ifdef VERBOSE
	    printf("In mmCUDA\n");
	    printf("\t GPUN: %d\n", GPUN);
	    printf("\t range: %d..%d\n", start, end);
#endif	
	    CudaSafeCall(cudaMalloc(&dA, sizeof(float) * N));
	    CudaSafeCall(cudaMalloc(&dB, sizeof(float) * N));
	    CudaSafeCall(cudaMalloc(&dC, sizeof(float) * N));
	    
	    CudaSafeCall(cudaMemcpy(dA, A, sizeof(float) * N, cudaMemcpyHostToDevice));
	    CudaSafeCall(cudaMemcpy(dB, B, sizeof(float) * N, cudaMemcpyHostToDevice));
	    
	    if (!tiled) {
		mm<<<ceil(((float)N)/1024), 1024>>>(dA, dB, dC, ceil(sqrt(N)), N, N);
	    } else if (tiled == 1){
		dim3 block(32,32);
		dim3 grid(ceil(sqrt(N)/32), ceil(sqrt(N)/32));
		mm_tiled<<<grid, block>>>(dA, dB, dC, ceil(sqrt(N)), N, N);
	    } else {
	        cublasHandle_t handle;
		cublasCreate(&handle);           
	        float alpha = 1.0F;
		float beta = 0.0F;
	        int lda = sqrt(N), ldb = sqrt(N), ldc = sqrt(N);
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sqrt(N), sqrt(N), sqrt(N), &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
	    }	    
	    
	    CudaSafeCall(cudaDeviceSynchronize());
	    CudaSafeCall(cudaMemcpy(C + start, dC + start, sizeof(float) * GPUN, cudaMemcpyDeviceToHost));
	    //for (int i = 0; i < GPUN; i++) {
	    //	printf("C[%d] = %lf\n", start+i, C[start+i]);
	    //}
	    
	    CudaSafeCall(cudaFree(dA));
	    CudaSafeCall(cudaFree(dB));
	    CudaSafeCall(cudaFree(dC));
	}
    }
}
