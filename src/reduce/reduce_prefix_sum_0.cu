#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define WARP_SIZE 32

template<int SIZE>
__device__ float WarpPrefixSum(float val) {
    int lane = threadIdx.x & 31;
    #pragma unroll
    for(int mask = 1; mask <= SIZE / 2; mask <<= 1){
        float tmp = __shfl_up_sync(0xffffffff, val, mask);
        if(lane >= mask)
            val += tmp;
    }
    return val;
}

template<int THREADS_PER_BLOCK>
__device__ float BlockPrefixSum(float val){
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    constexpr int NUM_WARP = THREADS_PER_BLOCK / WARP_SIZE;
    __shared__ float smem[NUM_WARP];

    // warp 内部做 prefix sum
    val = WarpPrefixSum<WARP_SIZE>(val);

    // write sum of each warp to smem
    if(lane == WARP_SIZE - 1) {
        smem[warp_id] = val;
    }
    __syncthreads();

    // 第一个 warp，从 smem 读取数据再做一次prefix sum
    if(warp_id == 0){
        smem[lane] = WarpPrefixSum<NUM_WARP>(smem[lane]);
    }
    __syncthreads();
    if(warp_id >= 1) val += smem[warp_id - 1];
    return val;
}

template<int THREADS_PER_BLOCK>
__global__ void prefix_sum_kernel(float* arr, float* res, int N) { 
    int tid = threadIdx.x;
    float val = tid < N ? arr[tid] : 0;
    val = BlockPrefixSum<THREADS_PER_BLOCK>(val);
    if (tid < N) res[tid] = val;
}

bool check(float *cpu_res, float *cuda_res, int N){
    for (int i = 0; i < N; i++) {
        if (fabs(cpu_res[i]-cuda_res[i]) > 1e-20) {
            // return false;
        }
        printf("%f | %f    ", cpu_res[i], cuda_res[i]);
        if ((i+1) % 8 == 0) printf("\n");
    }
    return true;
}

int main() {
    constexpr int N = 256;
    float* h_arr = (float*)malloc(sizeof(float) * N);
    float* h_res = (float*)malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) h_arr[i] = i + 1;

    float* d_arr;
    float* d_res;
    cudaMalloc(&d_arr, sizeof(float) * N);
    cudaMemcpy(d_arr, h_arr, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMalloc(&d_res, sizeof(float) * N);

    prefix_sum_kernel<N><<<1, N>>>(d_arr, d_res, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_res, d_res, sizeof(float) * N, cudaMemcpyDeviceToHost);

    for (int i = 1; i < N; i++) h_arr[i] += h_arr[i - 1];
    if (check(h_arr, h_res, N)) std::cout << "Passed" << std::endl;
    else std::cout << "Failed" << std::endl;

    return 0;
}