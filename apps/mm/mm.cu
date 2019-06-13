#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <cublas_v2.h>

#define VERBOSE
#define PROF
#define CUDA_ERROR_CHECK
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

long long getCurrentTime() {
  struct timeval te;
  gettimeofday(&te, NULL); // get current time
  long long microseconds = te.tv_sec*1000000LL + te.tv_usec;
  return microseconds;
}

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

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
	       file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  err = cudaDeviceSynchronize();
  if( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
	       file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
#endif
}

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

__global__ void mm_tiled(float *dA, float *dB, float *dC, int DIM, int N, int GPUN) {
    int it, jt, kt, i, j, k;
    __shared__ float sA[32][32], sB[32][32];

    // (it, jt) => the first element of a specific tile
    it = blockIdx.y * 32;
    jt = blockIdx.x * 32;

    // (i, j) => specific element
    i = it + threadIdx.y;
    j = jt + threadIdx.x;

    if (i*DIM+j <= GPUN) {
	float sum = 0.0f;
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
#ifdef PROF
	    cudaEvent_t startCudaMallocEvent, endCudaMallocEvent;
	    cudaEvent_t startCudaMemcpyH2DEvent, endCudaMemcpyH2DEvent;
	    cudaEvent_t startCudaKernelEvent, endCudaKernelEvent;
	    cudaEvent_t startCudaMemcpyD2HEvent, endCudaMemcpyD2HEvent;
	    CudaSafeCall(cudaEventCreate(&startCudaMallocEvent));
	    CudaSafeCall(cudaEventCreate(&endCudaMallocEvent));
	    CudaSafeCall(cudaEventCreate(&startCudaMemcpyH2DEvent));
	    CudaSafeCall(cudaEventCreate(&endCudaMemcpyH2DEvent));
	    CudaSafeCall(cudaEventCreate(&startCudaKernelEvent));
	    CudaSafeCall(cudaEventCreate(&endCudaKernelEvent));
	    CudaSafeCall(cudaEventCreate(&startCudaMemcpyD2HEvent));
	    CudaSafeCall(cudaEventCreate(&endCudaMemcpyD2HEvent));
#endif

#ifdef PROF
	    CudaSafeCall(cudaEventRecord(startCudaMallocEvent));
#endif	    
	    CudaSafeCall(cudaMalloc(&dA, sizeof(float) * N));
	    CudaSafeCall(cudaMalloc(&dB, sizeof(float) * N));
	    CudaSafeCall(cudaMalloc(&dC, sizeof(float) * N));
#ifdef PROF
	    CudaSafeCall(cudaEventRecord(endCudaMallocEvent));
	    CudaSafeCall(cudaEventSynchronize(endCudaMallocEvent));
#endif
	    
#ifdef PROF
	    CudaSafeCall(cudaEventRecord(startCudaMemcpyH2DEvent));
#endif
	    CudaSafeCall(cudaMemcpy(dA, A, sizeof(float) * N, cudaMemcpyHostToDevice));
	    CudaSafeCall(cudaMemcpy(dB, B, sizeof(float) * N, cudaMemcpyHostToDevice));
#ifdef PROF
	    CudaSafeCall(cudaEventRecord(endCudaMemcpyH2DEvent));
	    CudaSafeCall(cudaEventSynchronize(endCudaMemcpyH2DEvent));
#endif
	    
#ifdef PROF
	    CudaSafeCall(cudaEventRecord(startCudaKernelEvent));
#endif
	    if (!tiled) {
		mm<<<ceil(((float)N)/1024), 1024>>>(dA, dB, dC, ceil(sqrt(N)), N, N);
	    } else if (tiled == 1){
		dim3 block(32,32);
		dim3 grid(ceil(sqrt(N)/32), ceil(sqrt(N)/32));
		mm_tiled<<<grid, block>>>(dA, dB, dC, ceil(sqrt(N)), N, N);
	    } else {
	        cublasHandle_t handle;
#ifdef PROF
		long long start = getCurrentTime();
#endif	
		cublasCreate(&handle);           
	        float alpha = 1.0F;
		float beta = 0.0F;
	        int lda = sqrt(N), ldb = sqrt(N), ldc = sqrt(N);
#ifdef PROF
		long long end = getCurrentTime();
		printf("cuBLAS prep: %lf msec\n", (float)(end-start)/1000);
#endif
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sqrt(N), sqrt(N), sqrt(N), &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
#ifdef PROF
		long long end2 = getCurrentTime();
		printf("cuBLAS finish: %lf msec\n", (float)(end2-start)/1000);
#endif
	    }	    
	    CudaCheckError();
#ifdef PROF
	    CudaSafeCall(cudaEventRecord(endCudaKernelEvent));
	    CudaSafeCall(cudaEventSynchronize(endCudaKernelEvent));
#endif
	    CudaSafeCall(cudaDeviceSynchronize());

#ifdef PROF
	    CudaSafeCall(cudaEventRecord(startCudaMemcpyD2HEvent));
#endif
	    CudaSafeCall(cudaMemcpy(C + start, dC + start, sizeof(float) * GPUN, cudaMemcpyDeviceToHost));
#ifdef PROF
	    CudaSafeCall(cudaEventRecord(endCudaMemcpyD2HEvent));
	    CudaSafeCall(cudaEventSynchronize(endCudaMemcpyD2HEvent));
#endif

#ifdef PROF
	    float msecMalloc, msecH2D, msecKernel, msecD2H;
	    CudaSafeCall(cudaEventElapsedTime(&msecMalloc, startCudaMallocEvent, endCudaMallocEvent));
	    CudaSafeCall(cudaEventElapsedTime(&msecH2D, startCudaMemcpyH2DEvent, endCudaMemcpyH2DEvent));
	    CudaSafeCall(cudaEventElapsedTime(&msecKernel, startCudaKernelEvent, endCudaKernelEvent));
	    CudaSafeCall(cudaEventElapsedTime(&msecD2H, startCudaMemcpyD2HEvent, endCudaMemcpyD2HEvent));
	    printf("CUDA malloc: %lf msec\n", msecMalloc);
	    printf("CUDA h2d: %lf msec\n", msecH2D);
	    printf("CUDA kernel: %lf msec\n", msecKernel);
	    printf("CUDA d2h: %lf msec\n", msecD2H);
#endif

	    //for (int i = 0; i < GPUN; i++) {
	    //	printf("C[%d] = %lf\n", start+i, C[start+i]);
	    //}
	    
	    CudaSafeCall(cudaFree(dA));
	    CudaSafeCall(cudaFree(dB));
	    CudaSafeCall(cudaFree(dC));
	}
    }
}
