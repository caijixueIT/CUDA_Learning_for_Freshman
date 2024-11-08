#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>


template <typename T, unsigned int WarpSize=32>
__device__ __forceinline__ T warpReduceSum(T sum) {
    if (WarpSize >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (WarpSize >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (WarpSize >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (WarpSize >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (WarpSize >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

// template <typename T, unsigned int WarpSize=32>
// __device__ __forceinline__ T warpReduceSum(T sum) {
//     if (WarpSize <= 2) sum += __shfl_up_sync(0xffffffff, sum, 1); // 0-16, 1-17, 2-18, etc.
//     if (WarpSize <= 4) sum += __shfl_up_sync(0xffffffff, sum, 2);// 0-8, 1-9, 2-10, etc.
//     if (WarpSize <= 8) sum += __shfl_up_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
//     if (WarpSize <= 16) sum += __shfl_up_sync(0xffffffff, sum, 8);// 0-2, 1-3, 4-6, 5-7, etc.
//     if (WarpSize <= 32) sum += __shfl_up_sync(0xffffffff, sum, 16);// 0-1, 2-3, 4-5, etc.
//     return sum;
// }

// template <typename T, unsigned int WarpSize=32>
// __device__ __forceinline__ T warpReduceSum(T val) {
//     for(int mask = WarpSize/2; mask > 0; mask >>= 1)
//         val += __shfl_xor_sync(0xffffffff, val, mask, WarpSize);
//     return val;
// }

template <typename T, unsigned int ThreadsPerBlock>
__device__ __forceinline__ T blockReduceSum(T val) {
    int lane = threadIdx.x & 0x1f; 
    int wid = threadIdx.x >> 5;
    constexpr int NUM_WARPS = ThreadsPerBlock >> 5;
    static __shared__ T shared[NUM_WARPS]; 

    val = warpReduceSum<T, 32>(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < NUM_WARPS) ? shared[lane] : (T)(0.0f);
    val = warpReduceSum<T, NUM_WARPS>(val);
                                
    return val;
}

template <unsigned int ThreadsPerBlock>
__global__ void arr_sum(float* arr, float* res, int n) {
    int tid = threadIdx.x;
    float sum = tid < n ? arr[tid] : 0.f;
    // sum = warpReduceSum(sum);
    sum = blockReduceSum<float, ThreadsPerBlock>(sum);
    if (tid == 0) *res = sum;
}

int main() {
    constexpr int N = 64;
    float* arr = (float*)malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) {
        arr[i] = i;
    }

    float* d_arr;
    cudaMalloc((void**)&d_arr, sizeof(float) * N);
    cudaMemcpy(d_arr, arr, sizeof(float) * N, cudaMemcpyHostToDevice);
    float *d_res;
    cudaMalloc((void**)&d_res, sizeof(float) * 1);
    arr_sum<N><<<1, N>>>(d_arr, d_res, N);
    float res;
    cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    float ref = N * (N - 1) / 2;
    bool is_equal = ref == res;
    printf("res: %f, ref: %f, is_equal: %d\n", res, ref, is_equal);
    return 0;
}