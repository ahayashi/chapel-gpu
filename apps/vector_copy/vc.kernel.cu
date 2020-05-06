#ifndef USE_LAMBDA
__global__ void vc(float *dA, float *dB, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        dA[id] = dB[id];
    }
}
#else
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
#endif

extern "C" {
#ifndef USE_LAMBDA    
    void LaunchVC(float* dA, float *dB, int N) {
        vc<<<ceil(((float)N)/1024), 1024>>>(dA, dB, N);
    }
#else
    void LaunchVC(float *dA, float *dB, int N) {
        call_gpu_functor(N, 1024, NULL, [=] __device__ (int i) { dA[i] = dB[i]; });
    }
#endif    
}
