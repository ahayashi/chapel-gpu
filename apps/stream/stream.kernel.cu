

__global__ void stream(float* dA, float* dB, float* dC, float alpha, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        dA[id] = dB[id] + alpha*dC[id];
    }
}

extern "C" {
    void LaunchStream(float* dA, float *dB, float* dC, float alpha, int N) {
        stream<<<ceil(((float)N)/1024), 1024>>>(dA, dB, dC, alpha, N);
    }
}
