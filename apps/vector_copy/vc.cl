__kernel void vc(__global float *A, __global const float *B, int n) {
    int id = get_global_id(0);
    if (id < n) {
      A[id] = B[id];
    }
}
