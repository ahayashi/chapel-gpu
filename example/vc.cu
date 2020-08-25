__global__ void vc(float *dA, float *dB, int N) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < N) {
        dA[id] = dB[id];
  }
}

extern "C" {
  void vcGPU(float* A, float *B, int start, int end, int GPUN) {
    float *dA, *dB;
    cudaMalloc(&dA, sizeof(float) * GPUN);
    cudaMalloc(&dB, sizeof(float) * GPUN);
    cudaMemcpy(dB, B + start, sizeof(float) * GPUN, cudaMemcpyHostToDevice);
    vc<<<ceil(((float)GPUN)/1024), 1024>>>(dA, dB, GPUN);
    cudaDeviceSynchronize();
    cudaMemcpy(A + start, dA, sizeof(float) * GPUN, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
  }
}
