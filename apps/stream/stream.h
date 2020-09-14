//extern "C" {
    void LaunchStream(float *dA, float *dB, float *dC, float alpha, int N);
    void streamCUDA(float* A, float *B, float *C, float alpha, int start, int end, int GPUN);
//}
