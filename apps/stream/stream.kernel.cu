
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 1024
#endif

__global__ void stream(float* dA, float* dB, float* dC, float alpha, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        dA[id] = dB[id] + alpha*dC[id];
    }
}

extern "C" {
    void LaunchStream(float* dA, float *dB, float* dC, float alpha, int N) {
        stream<<<ceil(((float)N)/THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(dA, dB, dC, alpha, N);
    }
}
