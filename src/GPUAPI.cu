#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <cuda_profiler_api.h>

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

template<typename functor_type>
static __global__ void driver_kernel(functor_type functor, unsigned niters) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < niters) {
        functor(tid);
    }
}

template <typename functor_type>
inline void call_gpu_functor(unsigned niters, unsigned tile_size,
        cudaStream_t stream, functor_type functor) {
    //functor_type *actual = (functor_type *)functor;

    const unsigned block_size = tile_size;
    const unsigned nblocks = (niters + block_size - 1) / block_size;
    driver_kernel<<<nblocks, block_size, 0, stream>>>(functor, niters);
}

extern "C" {

  void GetDeviceCount(int *count) {
    CudaSafeCall(cudaGetDeviceCount(count));
  }

  void GetDevice(int *device) {
    CudaSafeCall(cudaGetDevice(device));
  }

  void SetDevice(int device) {
    CudaSafeCall(cudaSetDevice(device));
  }

  void ProfilerStart() {
    CudaSafeCall(cudaProfilerStart());
  }

  void ProfilerStop() {
    CudaSafeCall(cudaProfilerStop());
  }

  void Malloc(void** devPtr, size_t size) {
    CudaSafeCall(cudaMalloc(devPtr, size));
    printf("in malloc ptr: %p\n", *devPtr);
  }

  void Memcpy(void* dst, void* src, size_t count, int kind) {
      switch (kind) {
      case 0:
          CudaSafeCall(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
          break;
      case 1:
          CudaSafeCall(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
          break;
      default:
          printf("Warning\n");
      }
  }

  void Launch(float *dA, float *dB, int N) {
    printf("Launching kernel\n");
    call_gpu_functor(N, 1024, NULL, [=] __device__ (int i) { dA[i] = dB[i]; });
    cudaDeviceSynchronize();
    CudaCheckError();
  }
}
