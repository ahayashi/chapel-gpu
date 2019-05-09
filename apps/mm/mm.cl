__kernel void mm(__global const float *A, __global const float *B, __global float *C, int DIM, int N, int GPUN) {
    int id = get_global_id(0);
    if (id <= GPUN) {
        int i = id / DIM;
        int j = id % DIM;
        int sum = 0;
        for (int k = 0; k < DIM; k++) {
            sum += A[i*DIM+k] * B[k*DIM+j];
        }
        C[id] += sum;
    }
}
