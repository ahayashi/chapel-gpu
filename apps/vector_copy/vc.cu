#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
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
	    CudaSafeCall(cudaMalloc(&dA, sizeof(float) * GPUN));
	    CudaSafeCall(cudaMalloc(&dB, sizeof(float) * GPUN));
	    CudaSafeCall(cudaMemcpy(dA, A + start, sizeof(float) * GPUN, cudaMemcpyHostToDevice));
	    CudaSafeCall(cudaMemcpy(dB, B + start, sizeof(float) * GPUN, cudaMemcpyHostToDevice));
	    
	    vc<<<ceil(((float)GPUN)/1024), 1024>>>(dA, dB, GPUN);
	    
	    CudaSafeCall(cudaDeviceSynchronize());
	    CudaSafeCall(cudaMemcpy(A + start, dA, sizeof(float) * GPUN, cudaMemcpyDeviceToHost));
	    
	    CudaSafeCall(cudaFree(dA));
	    CudaSafeCall(cudaFree(dB));
	}
    }
}
