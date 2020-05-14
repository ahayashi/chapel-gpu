#ifndef USE_LAMBDA
__global__ void kernel2(float *dW, float *dWcurr, float *dX, float *dY, float alpha, int nSamples, int nFeatures, int start, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
	float err = 0.0;
	for (int s = 0; s < nSamples; s++) {
	    float arg = 0.0;
	    for (int f = 0; f < nFeatures; f++) {
		arg += dWcurr[f] * dX[s * (nFeatures) + f];
	    }
	    float hypo = 1 / (1 + exp(-arg));
	    err += (hypo - dY[s]) * dX[s * (nFeatures) + start + id];
	}
	dW[id] = dWcurr[start + id] - alpha * err;
    }
}
#else
#include "lambda.h"
#endif

extern "C" {
#ifndef USE_LAMBDA
    void LaunchLR(float* dX, float *dY, float *dW, float *dWcurr, float alpha, int nSamples, int nFeatures, int start, int end, int GPUN) {
 	    kernel2<<<ceil(((float)GPUN)/1024), 1024>>>(dW, dWcurr, dX, dY, alpha, nSamples, nFeatures, start-1, GPUN);
    }
#else
    void LaunchLR(float* dX, float *dY, float *dW, float *dWcurr, float alpha, int nSamples, int nFeatures, int start, int end, int GPUN) {
        call_gpu_functor(GPUN, 1024, NULL, [=] __device__ (int id) {
                float err = 0.0;
                for (int s = 0; s < nSamples; s++) {
                    float arg = 0.0;
                    for (int f = 0; f < nFeatures; f++) {
                        arg += dWcurr[f] * dX[s * (nFeatures) + f];
                    }
                    float hypo = 1 / (1 + exp(-arg));
                    err += (hypo - dY[s]) * dX[s * (nFeatures) + (start - 1) + id];
                }
                dW[id] = dWcurr[(start - 1) + id] - alpha * err;
            });
    }
#endif    
}
