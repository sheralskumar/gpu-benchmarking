__kernel void vector_add(__global const float* A, __global const float* B, __global float* C, int N)
{
    int i = get_global_id(0);
    if (i < N)
    C[i] = A[i] + B[i];
}

__kernel void vector_mul(__global float* A, __global float* B, __global float* C, const int N) {
    int idx = get_global_id(0);
    if(idx < N) {
        C[idx] = A[idx] * B[idx];
    }
}

__kernel void relu(__global float* input, __global float* output, const int N) {
    int idx = get_global_id(0);
    if(idx < N) {
        output[idx] = fmax(input[idx], 0.0f);
    }
}

__kernel void sigmoid(__global float* input, __global float* output, const int N) {
    int idx = get_global_id(0);
    if(idx < N) {
        output[idx] = 1.0f / (1.0f + exp(-input[idx]));
    }
}




