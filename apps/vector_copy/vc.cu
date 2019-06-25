#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>

//#define VERBOSE
#define PROF
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

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

__global__ void vc(float *dA, float *dB, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
	dA[id] = dB[id];
    }
}

extern "C" {
    void vcCUDA(float* A, float *B, int start, int end, int GPUN) {
	float *dA, *dB;
	if (GPUN > 0) {
	    assert(end - start + 1 == GPUN);
#ifdef VERBOSE
	    printf("In vcCUDA\n");
	    printf("\t GPUN: %d\n", GPUN);
	    printf("\t range: %d..%d\n", start, end);
#endif	
#ifdef PROF
	    cudaEvent_t startCudaKernelEvent, endCudaKernelEvent;
	    CudaSafeCall(cudaEventCreate(&startCudaKernelEvent));
	    CudaSafeCall(cudaEventCreate(&endCudaKernelEvent));
#endif

	    CudaSafeCall(cudaMalloc(&dA, sizeof(float) * GPUN));
	    CudaSafeCall(cudaMalloc(&dB, sizeof(float) * GPUN));
	    CudaSafeCall(cudaMemcpy(dB, B + start, sizeof(float) * GPUN, cudaMemcpyHostToDevice));
#ifdef PROF
	    CudaSafeCall(cudaEventRecord(startCudaKernelEvent));
#endif	    
	    vc<<<ceil(((float)GPUN)/1024), 1024>>>(dA, dB, GPUN);
#ifdef PROF
	    CudaSafeCall(cudaEventRecord(endCudaKernelEvent));
	    CudaSafeCall(cudaEventSynchronize(endCudaKernelEvent));
#endif	    
	    CudaCheckError();
	    CudaSafeCall(cudaDeviceSynchronize());
	    CudaSafeCall(cudaMemcpy(A + start, dA, sizeof(float) * GPUN, cudaMemcpyDeviceToHost));

#ifdef PROF
	    float msecKernel;
	    CudaSafeCall(cudaEventElapsedTime(&msecKernel, startCudaKernelEvent, endCudaKernelEvent));
	    printf("CUDA kernel: %lf msec\n", msecKernel);
#endif
	    
	    CudaSafeCall(cudaFree(dA));
	    CudaSafeCall(cudaFree(dB));
	}
    }
}
