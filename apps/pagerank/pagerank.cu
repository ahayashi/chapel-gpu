#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 1024
#endif

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

#define NDIMS 2

__global__ void kernel1(float *dranks, int *dlinks, int *dlink_counts, float *dlink_weights, int nLinks, int start, int end, int GPUN)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (start <= id && id <= end) {
	dlink_weights[id-start] = dranks[dlinks[id*NDIMS+0]-1] / (float) dlink_counts[dlinks[id*NDIMS+0]-1];	
    }
}

__global__ void kernel2(float *dranks, int *dlinks, float *dlink_weights, int nDocs, int nLinks, int start, int end, int GPUN)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (start <= id && id <= end) {
	float new_rank = 0.0f;
	// look for links pointing to this document
	for (int l = 0; l < nLinks; l++) {
	    int dst = dlinks[l*NDIMS+1] - 1;
	    if (dst == id) {
		new_rank += dlink_weights[l];
	    }
	}
	dranks[id-start] = new_rank;
    }
}

extern "C" {
    
    void prCUDA1(float* ranks, int *links, int *link_counts, float *link_weights, int nDocs, int nLinks, int start, int end, int GPUN) {
	float *dranks, *dlink_weights;
	int *dlinks, *dlink_counts;
	if (GPUN > 0) {
	    assert(end - start + 1 == GPUN);
#ifdef VERBOSE
	    printf("In prCUDA1\n");
	    printf("\t GPUN: %d\n", GPUN);
	    printf("\t range: %d..%d\n", start, end);
#endif		
	    CudaSafeCall(cudaMalloc(&dranks, sizeof(float) * nDocs));
	    CudaSafeCall(cudaMalloc(&dlinks, sizeof(int) * nLinks * 2));
	    CudaSafeCall(cudaMalloc(&dlink_counts, sizeof(int) * nDocs));
	    CudaSafeCall(cudaMalloc(&dlink_weights, sizeof(float) * nLinks));
	    
	    CudaSafeCall(cudaMemcpy(dranks, ranks, sizeof(float) * nDocs, cudaMemcpyHostToDevice));
	    CudaSafeCall(cudaMemcpy(dlinks, links, sizeof(int) * nLinks * 2, cudaMemcpyHostToDevice));
	    CudaSafeCall(cudaMemcpy(dlink_counts, link_counts, sizeof(int) * nDocs, cudaMemcpyHostToDevice));
	    
	    kernel1<<<ceil(((float)nLinks)/THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(dranks, dlinks, dlink_counts, dlink_weights, nLinks, start, end, GPUN);
	    
	    CudaSafeCall(cudaDeviceSynchronize());
	    CudaSafeCall(cudaMemcpy(link_weights + start, dlink_weights, sizeof(float) * GPUN, cudaMemcpyDeviceToHost));

	    CudaSafeCall(cudaFree(dranks));
	    CudaSafeCall(cudaFree(dlinks));
	    CudaSafeCall(cudaFree(dlink_counts));
	    CudaSafeCall(cudaFree(dlink_weights));
	}
    }
    
    void prCUDA2(float* ranks, int *links, float *link_weights, int nDocs, int nLinks, int start, int end, int GPUN) {
	float *dranks, *dlink_weights;
	int *dlinks;
	if (GPUN > 0) {
	    assert(end - start + 1 == GPUN);
#ifdef VERBOSE
	    printf("In prCUDA2\n");
	    printf("\t GPUN: %d\n", GPUN);
	    printf("\t range: %d..%d\n", start, end);
#endif			
	    CudaSafeCall(cudaMalloc(&dranks, sizeof(float) * GPUN));
	    CudaSafeCall(cudaMalloc(&dlinks, sizeof(int) * nLinks * 2));
	    CudaSafeCall(cudaMalloc(&dlink_weights, sizeof(float) * nLinks));
	    
	    CudaSafeCall(cudaMemcpy(dlinks, links, sizeof(int) * nLinks * 2, cudaMemcpyHostToDevice));
	    CudaSafeCall(cudaMemcpy(dlink_weights, link_weights, sizeof(float) * nLinks, cudaMemcpyHostToDevice));
	    
	    kernel2<<<ceil(((float)nDocs)/THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(dranks, dlinks, dlink_weights, nDocs, nLinks, start, end, GPUN);
	    
	    CudaSafeCall(cudaDeviceSynchronize());	    
	    CudaSafeCall(cudaMemcpy(ranks + start, dranks, sizeof(float) * GPUN, cudaMemcpyDeviceToHost));
	    
	    CudaSafeCall(cudaFree(dranks));
	    CudaSafeCall(cudaFree(dlinks));
	    CudaSafeCall(cudaFree(dlink_weights));
	}
    }
}
