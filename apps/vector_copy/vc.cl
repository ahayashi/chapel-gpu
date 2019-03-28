__kernel void vc(__global float *A, __global const float *B) {
    int i = get_global_id(0);
    A[i] = B[i];
}
