/*
* @Description: 计算大数组的前缀和
* @LastEditors: Bruce Li
* @LastEditTime: 2025-07-29 11:42
*/

#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <random>
#include <chrono>

#define WARMING_TIMES 50
#define PROFILING_TIMES 100
#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32
#define MIN_VAL -100
#define MAX_VAL 100

int* h_arr; // N
int* h_res; // N
int* h_ref; // N
int* d_arr; // N
int* d_res; // N


int random_int(int min, int max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min, max);
    return dis(gen);
}

void cpu_prefix_sum(int* arr, int* ref, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
        ref[i] = sum;
    }
}

bool check(int* cpu_prefix_sum, int* gpu_prefix_sum, int n) {
    for (int i = 0; i < n; i++) {
        if (cpu_prefix_sum[i] != gpu_prefix_sum[i]) {
            return false;
        }
    }
    return true;
}

void init_arr(int* arr, int n, bool zero_init=false) {
    for (int i = 0; i < n; i++) {
        arr[i] = i; //zero_init ? 0.f : random_int(MIN_VAL, MAX_VAL);
    }
}

void create_arr(int n) {
    h_arr = (int*)malloc(n * sizeof(int));
    init_arr(h_arr, n);
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    h_res = (int*)malloc(n * sizeof(int));
    h_ref = (int*)malloc(n * sizeof(int));
    cudaMalloc((void**)&d_res, n * sizeof(int));
    init_arr(h_res, 1, true);
}

void destroy_arr() {
    cudaFree(d_arr);
    cudaFree(d_res);
    free(h_arr);
    free(h_res);
    free(h_ref);
    d_arr = nullptr;
    d_res = nullptr;
    h_arr = nullptr;
    h_res = nullptr;
    h_ref = nullptr;
}

template <typename Kernel, typename... Args, int WARMUP=50, int PROFILE=100>
void cpu_kernel_profiling(Kernel kernel, std::string fun_name, int* gpu_prefix_sum, int* cpu_prefix_sum, int n, Args&&... args) {
    std::cout << std::endl << std::string(100, '=') << std::endl;
    // check
    if (check(cpu_prefix_sum, gpu_prefix_sum, n)) {
        printf("%s PASS\n", fun_name.c_str());
    } else {
        printf("%s FAIL\n", fun_name.c_str());
        return;
    }
    // warming up
    for (int i = 0; i < WARMUP; i++) {
        kernel(std::forward<Args>(args)...);
    }
    // profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventSynchronize(start);
    for (int i = 0; i < PROFILE; i++) {
        kernel(std::forward<Args>(args)...);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("%s Average elasped time = %f ms\n", fun_name.c_str(), elapsed_time / PROFILING_TIMES);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

__global__ void kernel_0(int* d_arr, int* d_res, int* d_sum, int n) { 
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int block_offset = bid * blockDim.x * 2;
    __shared__ int temp[THREAD_PER_BLOCK * 2];
    temp[tid << 1] = (block_offset + (tid << 1)) < n ? d_arr[block_offset + (tid << 1)] : 0;
    temp[(tid << 1) + 1] = (block_offset + ((tid << 1) + 1)) < n ? d_arr[block_offset + ((tid << 1) + 1)] : 0;
    __syncthreads();

    int offset = 1;
    for (int d = blockDim.x; d >= 1; d >>= 1) {
        if (tid < d) {
            int a = offset * ( 2 * tid + 1) - 1;
            int b = offset * ( 2 * tid + 2) - 1;
            temp[b] += temp[a];
        }
        offset <<= 1;
        __syncthreads();
    }

    if (tid == 0) {
        d_sum[bid] = temp[blockDim.x * 2 - 1];
        temp[blockDim.x * 2 - 1] = 0;
    }
    __syncthreads();

    for (int d = 1; d <= blockDim.x; d <<= 1) {
        offset >>= 1;
        if (tid < d) {
            int a = offset * (2 * tid + 1) - 1;
            int b = offset * (2 * tid + 2) - 1;
            int v = temp[a];
            temp[a] = temp[b];
            temp[b] += v;
        }
        __syncthreads();
    }

    if ((tid << 1) + block_offset < n) {
        d_res[(tid << 1) + block_offset] = temp[tid << 1];
    }
    if ((tid << 1) + 1 + block_offset < n) {
        d_res[(tid << 1) + 1 + block_offset] = temp[(tid << 1) + 1];
    }
}

__global__ void kernel_0_add(int* d_prefix_sum, int* d_block_sum, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_offset = bid * blockDim.x * 2;

    int a = tid + block_offset;
    int b = tid + block_offset + blockDim.x;
    if (a < n) {
        d_prefix_sum[a] += d_block_sum[bid];
    }
    if (b < n) {
        d_prefix_sum[b] += d_block_sum[bid];
    }
}

void recursive_launch_kernel_0(int* d_arr, int* d_res, int n) { 
    int num_blocks = (n + THREAD_PER_BLOCK * 2 - 1) / (THREAD_PER_BLOCK * 2);
    int *d_sum, *d_prefix_sum;
    cudaMalloc((void**)&d_sum, num_blocks * sizeof(int));
    cudaMalloc((void**)&d_prefix_sum, num_blocks * sizeof(int));
    kernel_0<<<num_blocks, THREAD_PER_BLOCK>>>(d_arr, d_res, d_sum, n);
    if (num_blocks > 1) {
        recursive_launch_kernel_0(d_sum, d_prefix_sum, num_blocks);
        kernel_0_add<<<num_blocks, THREAD_PER_BLOCK>>>(d_res, d_prefix_sum, n);
    }
    cudaFree(d_sum);
    cudaFree(d_prefix_sum);
    d_sum = nullptr;
    d_prefix_sum = nullptr;
}

void launch_kernel_0(int* d_arr, int* d_res, int* h_arr, int* h_res, int* h_ref, int n) {
    recursive_launch_kernel_0(d_arr, d_res, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_res, d_res + 1, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
    h_res[n - 1] = h_res[n - 2] + h_arr[n - 1];
    cpu_kernel_profiling(recursive_launch_kernel_0, std::string(__func__), h_res, h_ref, n, d_arr, d_res, n);
}


template<int SIZE>
__device__ int WarpPrefixSum(int val) {
    int lane = threadIdx.x & 31;
    #pragma unroll
    for(int mask = 1; mask <= SIZE / 2; mask <<= 1){
        int tmp = __shfl_up_sync(0xffffffff, val, mask);
        if(lane >= mask)
            val += tmp;
    }
    return val;
}

template<int THREADS_PER_BLOCK>
__device__ int BlockPrefixSum(int val){
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    constexpr int NUM_WARP = THREADS_PER_BLOCK / WARP_SIZE;
    __shared__ int smem[NUM_WARP];

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
__global__ void kernel_1(int* arr, int* res, int* sum, int n) { 
    int tid = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    int val = tid < n ? arr[tid] : 0;
    val = BlockPrefixSum<THREADS_PER_BLOCK>(val);
    if (tid < n) res[tid] = val;
    if (threadIdx.x == THREADS_PER_BLOCK - 1) {
        sum[blockIdx.x] = val;
    }
}

__global__ void kernel_1_add(int* arr, int* sum, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) arr[tid] += blockIdx.x > 0 ? sum[blockIdx.x - 1] : 0;
}

void recursive_launch_kernel_1(int* d_arr, int* d_res, int n) {
    int num_blocks = (n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    int *d_sum, *d_prefix_sum;
    cudaMalloc(&d_sum, num_blocks * sizeof(int));
    cudaMalloc(&d_prefix_sum, num_blocks * sizeof(int));
    kernel_1<THREAD_PER_BLOCK><<<num_blocks, THREAD_PER_BLOCK>>>(d_arr, d_res, d_sum, n);
    if (num_blocks > 1) {
        recursive_launch_kernel_1(d_sum, d_prefix_sum, num_blocks);
        kernel_1_add<<<num_blocks, THREAD_PER_BLOCK>>>(d_res, d_prefix_sum, n);
    }
    cudaFree(d_sum);
    cudaFree(d_prefix_sum);
    d_sum = nullptr;
    d_prefix_sum = nullptr;
}

void launch_kernel_1(int* d_arr, int* d_res, int* h_arr, int* h_res, int* h_ref, int n) {
    recursive_launch_kernel_1(d_arr, d_res, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_res, d_res, n * sizeof(int), cudaMemcpyDeviceToHost);
    cpu_kernel_profiling(recursive_launch_kernel_1, std::string(__func__), h_res, h_ref, n, d_arr, d_res, n);
}

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 1000000;
    printf("Array length = %d\n", N);
    create_arr(N);
    cpu_prefix_sum(h_arr, h_ref, N);
    launch_kernel_0(d_arr, d_res, h_arr, h_res, h_ref, N);
    launch_kernel_1(d_arr, d_res, h_arr, h_res, h_ref, N);
    destroy_arr();
    return 0;
}