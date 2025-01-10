#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256

__global__ void reduce_sum_kernel(float* arr, int n, float* res) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float smem[THREAD_PER_BLOCK];

    float tmp = 0.f;
    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        tmp += arr[i];
    }
    smem[tid] = tmp;
    
    __syncthreads();

    for (int i = THREAD_PER_BLOCK / 2; i >= 1; i >>= 1) {
        if (tid < i) {
            smem[tid] += smem[tid + i];
        }
        __syncthreads();
    }

    if (tid == 0) {
        res[blockIdx.x] = smem[0];
    }
}

bool check(float result, float result_ref) {
    printf("result: %f, result_ref: %f\n", result, result_ref);
    return fabs(result - result_ref) < 1e-3;
}

int main() {
    constexpr int N = 1024 * 1024;
    constexpr int threads_per_block = THREAD_PER_BLOCK;
    constexpr int num_blocks = threads_per_block;

    float* arr = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        arr[i] = 1;
    }
    
    float* d_arr;
    cudaMalloc((void**)&d_arr, N * sizeof(float));
    cudaMemcpy(d_arr, arr, N * sizeof(float), cudaMemcpyHostToDevice);

    float* d_out;
    cudaMalloc((void**)&d_out, num_blocks * sizeof(float));

    float result_ref = 0.f;
    for (int i = 0; i < N; ++i) {
        result_ref += arr[i];
    }

    dim3 grid_dim(num_blocks);
    dim3 block_dim(threads_per_block);

    reduce_sum_kernel<<<grid_dim, block_dim>>>(d_arr, N, d_out);
    cudaDeviceSynchronize();

    float result;
    float* d_result;
    cudaMalloc((void**)&d_result, sizeof(float));
    reduce_sum_kernel<<<1, block_dim>>>(d_out, num_blocks, d_result);
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if (check(result, result_ref)) {
        std::cout << "succeed!" << std::endl;
    } else {
        std::cout << "failed!" << std::endl;
    }

    return 0;
}