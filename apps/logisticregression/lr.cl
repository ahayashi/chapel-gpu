__kernel void lr1(__global float *Wcurr, __global const float *W, int n) {
    int id = get_global_id(0);
    if (id < n) {
        Wcurr[id] = W[id];
    }
}

__kernel void lr2(__global float *dW, __global const float *dWcurr, __global const float *dX, __global const float *dY, float alpha, int nSamples, int nFeatures, int start, int N) {
    int id = get_global_id(0);
    if (id < N) {
        float err = 0.0;
        for (int s = 0; s < nSamples; s++) {
            float arg = 0.0;
            for (int f = 0; f < nFeatures; f++) {
                arg += dWcurr[f] * dX[s * (nFeatures) + f];
            }
            float hypo = 1 / (1 + exp(-arg));
            err += (hypo - dY[s]) * dX[s * (nFeatures) + id];
        }
        dW[id] = dWcurr[start + id] - alpha * err;
    }
}
