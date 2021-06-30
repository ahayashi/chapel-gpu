#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <cuda.h>
#ifdef __NVCC__
#include <cuda_profiler_api.h>
#endif

#define CUDA_ERROR_CHECK
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

//TODO: Remove this hack added for DPC++
#ifdef DPCT_COMPATIBILITY_TEMP
#undef DPCT_COMPATIBILITY_TEMP
#endif

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
#ifdef __NVCC__
    CudaSafeCall(cudaProfilerStart());
#endif
  }

  void ProfilerStop() {
#ifdef __NVCC__
    CudaSafeCall(cudaProfilerStop());
#endif
  }

  void DeviceSynchronize() {
    CudaCheckError();
    CudaSafeCall(cudaDeviceSynchronize());
  }

  void Malloc(void** devPtr, size_t size) {
    CudaSafeCall(cudaMalloc(devPtr, size));
  }

  void MallocPtr(void*** devPtr, size_t size) {
    CudaSafeCall(cudaMalloc(devPtr, size));
  }

  void MallocPtrPtr(void**** devPtr, size_t size) {
    CudaSafeCall(cudaMalloc(devPtr, size));
  }

  void MallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) {
    CudaSafeCall(cudaMallocPitch(devPtr, pitch, width, height));
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
          printf("Fatal: Wrong Memcpy kind!\n");
          exit(1);
      }
  }

  void Memcpy2D(void* dst, size_t dpitch, void* src, size_t spitch, size_t width, size_t height, int kind) {
      switch (kind) {
      case 0:
          CudaSafeCall(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyHostToDevice));
          break;
      case 1:
          CudaSafeCall(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToHost));
          break;
      default:
          printf("Fatal: Wrong Memcpy kind!\n");
          exit(1);
      }
  }

  void Free(void* devPtr) {
      CudaSafeCall(cudaFree(devPtr));
  }
}
