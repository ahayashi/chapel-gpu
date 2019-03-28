__kernel void stream(__global float *A, __global const float *B, __global const float *C, float alpha, int n) {
    int id = get_global_id(0);
    if (id < n) {
      A[id] = B[id] + alpha * C[id];
    }
}
