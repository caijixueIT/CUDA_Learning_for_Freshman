/*
 * @Description: 计算大数组的累加和
 * @LastEditors: Bruce Li
 * @LastEditTime: 2025-07-28 15:00
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

struct int4_pack {
    int x, y, z, w;
    __device__ __host__ __forceinline__ int sum() {
        return x + y + z + w;
    }
};

#define FETCH_INT4(pointer) (reinterpret_cast<int4_pack*>(&(pointer))[0])

#define WARMING_TIMES 50
#define PROFILING_TIMES 100
#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32
#define MIN_VAL -100
#define MAX_VAL 100

int* h_arr; // N
int* h_res; // 1
int* d_arr; // N
int* d_res; // 1

int random_int(int min, int max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min, max);
    return dis(gen);
}

int cpu_reduce_sum(int* arr, int n) {
    int sum = 0.f;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

bool check(int cpu_sum, int gpu_sum) {
    return cpu_sum == gpu_sum;
}

void init_arr(int* arr, int n, bool zero_init=false) {
    for (int i = 0; i < n; i++) {
        arr[i] = zero_init ? 0.f : random_int(MIN_VAL, MAX_VAL);
    }
}

void create_arr(int n) {
    h_arr = (int*)malloc(n * sizeof(int));
    init_arr(h_arr, n);
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    h_res = (int*)malloc(sizeof(int));
    cudaMalloc((void**)&d_res, sizeof(int));
}

void destroy_arr() {
    cudaFree(d_arr);
    cudaFree(d_res);
    free(h_arr);
    free(h_res);
    d_arr = nullptr;
    d_res = nullptr;
    h_arr = nullptr;
    h_res = nullptr;
}

template <typename Kernel, typename... Args, int WARMUP=50, int PROFILE=100>
void kernel_profiling(Kernel kernel, dim3 grid, dim3 block, std::string fun_name, int gpu_sum, int cpu_sum, Args&&... args) {
    std::cout << std::endl << std::string(100, '=') << std::endl;
    // check
    if (check(cpu_sum, gpu_sum)) {
        printf("%s PASS, cpu sum = %d, gpu sum = %d\n", fun_name.c_str(), cpu_sum, gpu_sum);
    } else {
        printf("%s FAIL, cpu sum = %d, gpu sum = %d\n", fun_name.c_str(), cpu_sum, gpu_sum);
        return;
    }
    // warming up
    for (int i = 0; i < WARMUP; i++) {
        kernel<<<grid, block>>>(std::forward<Args>(args)...);
    }
    // profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventSynchronize(start);
    for (int i = 0; i < PROFILE; i++) {
        kernel<<<grid, block>>>(std::forward<Args>(args)...);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("%s Average elasped time = %f ms\n", fun_name.c_str(), elapsed_time / PROFILING_TIMES);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <typename Kernel, typename... Args, int WARMUP=50, int PROFILE=100>
void cpu_kernel_profiling(Kernel kernel, std::string fun_name, int gpu_sum, int cpu_sum, Args&&... args) {
    std::cout << std::endl << std::string(100, '=') << std::endl;
    // check
    if (check(cpu_sum, gpu_sum)) {
        printf("%s PASS, cpu sum = %d, gpu sum = %d\n", fun_name.c_str(), cpu_sum, gpu_sum);
    } else {
        printf("%s FAIL, cpu sum = %d, gpu sum = %d\n", fun_name.c_str(), cpu_sum, gpu_sum);
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

__global__ void kernel_0(int* arr, int n, int* res) { 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ int smem[THREAD_PER_BLOCK];
    smem[tid] = idx < n ? arr[idx] : 0;
    __syncthreads();

    for (int i = THREAD_PER_BLOCK / 2; i >= 1; i >>= 1) {
        if (tid < i) {
            smem[tid] += smem[tid + i];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(res, smem[0]);
    }
}

void launch_kernel_0(int* d_arr, int* d_res, int* h_res, int n, int cpu_sum) {
    init_arr(h_res, 1, true);
    cudaMemcpy(d_res, h_res, sizeof(int), cudaMemcpyHostToDevice);
    dim3 grid_dim((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 block_dim(THREAD_PER_BLOCK);
    kernel_0<<<grid_dim, block_dim>>>(d_arr, n, d_res);
    cudaMemcpy(h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    int gpu_sum = h_res[0];
    kernel_profiling(kernel_0, grid_dim, block_dim, std::string(__func__), gpu_sum, cpu_sum, d_arr, n, d_res);
}

__global__ void kernel_1(int* arr, int n, int* res) { 
    int idx = blockIdx.x * (blockDim.x << 1) + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ int smem[THREAD_PER_BLOCK << 1];
    smem[tid] = idx < n ? arr[idx] : 0;
    smem[tid + THREAD_PER_BLOCK] = idx + THREAD_PER_BLOCK < n ? arr[idx + THREAD_PER_BLOCK] : 0;
    __syncthreads();

    for (int i = THREAD_PER_BLOCK; i >= 1; i >>= 1) {
        if (tid < i) {
            smem[tid] += smem[tid + i];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(res, smem[0]);
    }
}

void launch_kernel_1(int* d_arr, int* d_res, int* h_res, int n, int cpu_sum) {
    init_arr(h_res, 1, true);
    cudaMemcpy(d_res, h_res, sizeof(int), cudaMemcpyHostToDevice);
    dim3 grid_dim((n + 2 * THREAD_PER_BLOCK - 1) / (2 * THREAD_PER_BLOCK));
    dim3 block_dim(THREAD_PER_BLOCK);
    kernel_1<<<grid_dim, block_dim>>>(d_arr, n, d_res);
    cudaMemcpy(h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    int gpu_sum = h_res[0];
    kernel_profiling(kernel_1, grid_dim, block_dim, std::string(__func__), gpu_sum, cpu_sum, d_arr, n, d_res);
}

__device__ __forceinline__ void kernel_2_helper(int* arr, long int offset, int idx, int n, int& sum) {
    offset += idx * (sizeof(int4_pack) / sizeof(int));
    if (offset + sizeof(int4_pack) / sizeof(int) <= n) {
        int4_pack v = FETCH_INT4(arr[offset]);
        sum += v.sum();
    } else {
        for (int j = 0; j < n - offset; j++) {
            sum += arr[offset + j];
        }
    }
}

__global__ void kernel_2(int* arr, int n, int* res, int load_float4_time) { 
    int tid = threadIdx.x;
    int base_offset_former = blockIdx.x * (blockDim.x * 2 * (sizeof(int4_pack) / sizeof(int)) * load_float4_time) + tid * (sizeof(int4_pack) / sizeof(int)) * load_float4_time;
    int base_offset_latter = base_offset_former + THREAD_PER_BLOCK * load_float4_time * (sizeof(int4_pack) / sizeof(int)); 

    __shared__ int smem[THREAD_PER_BLOCK << 1];
    int local_sum_former = 0;
    int local_sum_latter = 0;
    for (int i = 0; i < load_float4_time; i++) {
        kernel_2_helper(arr, base_offset_former, i, n, local_sum_former);
        kernel_2_helper(arr, base_offset_latter, i, n, local_sum_latter);
    }
    smem[tid] = local_sum_former;
    smem[tid + THREAD_PER_BLOCK] = local_sum_latter;
    __syncthreads();

    for (int i = THREAD_PER_BLOCK; i >= 1; i >>= 1) {
        if (tid < i) {
            smem[tid] += smem[tid + i];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(res, smem[0]);
    }
}

void launch_kernel_2(int* d_arr, int* d_res, int* h_res, int n, int cpu_sum) {
    init_arr(h_res, 1, true);
    cudaMemcpy(d_res, h_res, sizeof(int), cudaMemcpyHostToDevice);
    int load_float4_time = 2;
    int values_per_block = 2 * (sizeof(int4_pack) / sizeof(int)) * load_float4_time * THREAD_PER_BLOCK;
    dim3 grid_dim((n + values_per_block - 1) / values_per_block);
    dim3 block_dim(THREAD_PER_BLOCK);
    kernel_2<<<grid_dim, block_dim>>>(d_arr, n, d_res, load_float4_time);
    cudaMemcpy(h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    int gpu_sum = h_res[0];
    kernel_profiling(kernel_2, grid_dim, block_dim, std::string(__func__), gpu_sum, cpu_sum, d_arr, n, d_res, load_float4_time);
}

template <int KWARP_SIZE = WARP_SIZE>
__device__ __forceinline__ int warp_reduce(int val) {
    #pragma unroll
    for (int i = KWARP_SIZE / 2; i >= 1; i >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, i);
    }
    return val;
}

template <int KBLOCK_SIZE = THREAD_PER_BLOCK>
__device__ __forceinline__ int block_reduce(int val) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    constexpr int NUM_WARPS = KBLOCK_SIZE / WARP_SIZE;
    static __shared__ int smem[NUM_WARPS];
    
    val = warp_reduce<WARP_SIZE>(val);
    if (lane == 0) {
        smem[wid] = val;
    }
    __syncthreads();
    val = (lane < NUM_WARPS) ? smem[lane] : 0;
    val = warp_reduce<NUM_WARPS>(val);
    return val;
}

__global__ void kernel_3(int* arr, int n, int* res, int load_float4_time) { 
    int tid = threadIdx.x;
    int base_offset_former = blockIdx.x * (blockDim.x * 2 * (sizeof(int4_pack) / sizeof(int)) * load_float4_time) + tid * (sizeof(int4_pack) / sizeof(int)) * load_float4_time;
    int base_offset_latter = base_offset_former + THREAD_PER_BLOCK * load_float4_time * (sizeof(int4_pack) / sizeof(int)); 

    int local_sum = 0;
    for (int i = 0; i < load_float4_time; i++) {
        kernel_2_helper(arr, base_offset_former, i, n, local_sum);
        kernel_2_helper(arr, base_offset_latter, i, n, local_sum);
    }
    int block_sum = block_reduce<THREAD_PER_BLOCK>(local_sum);
    if (tid == 0) {
        atomicAdd(res, block_sum);
    }
}

void launch_kernel_3(int* d_arr, int* d_res, int* h_res, int n, int cpu_sum) {
    init_arr(h_res, 1, true);
    cudaMemcpy(d_res, h_res, sizeof(int), cudaMemcpyHostToDevice);
    int load_float4_time = 4;
    int values_per_block = 2 * (sizeof(int4_pack) / sizeof(int)) * load_float4_time * THREAD_PER_BLOCK;
    dim3 grid_dim((n + values_per_block - 1) / values_per_block);
    dim3 block_dim(THREAD_PER_BLOCK);
    kernel_3<<<grid_dim, block_dim>>>(d_arr, n, d_res, load_float4_time);
    cudaMemcpy(h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    int gpu_sum = h_res[0];
    kernel_profiling(kernel_3, grid_dim, block_dim, std::string(__func__), gpu_sum, cpu_sum, d_arr, n, d_res, load_float4_time);
}

__global__ void kernel_4(int* arr, int n, int* res, int load_float4_time) { 
    int tid = threadIdx.x;
    int base_offset_former = blockIdx.x * (blockDim.x * 2 * (sizeof(int4_pack) / sizeof(int)) * load_float4_time) + tid * (sizeof(int4_pack) / sizeof(int)) * load_float4_time;
    int base_offset_latter = base_offset_former + THREAD_PER_BLOCK * load_float4_time * (sizeof(int4_pack) / sizeof(int)); 

    int local_sum = 0;
    for (int i = 0; i < load_float4_time; i++) {
        kernel_2_helper(arr, base_offset_former, i, n, local_sum);
        kernel_2_helper(arr, base_offset_latter, i, n, local_sum);
    }
    int block_sum = block_reduce<THREAD_PER_BLOCK>(local_sum);
    if (tid == 0) {
        res[blockIdx.x] = block_sum;
    }
}

void kernel_4_helper(int* d_arr, int* d_res, int* h_res, int n, int cpu_sum, int* d_res_arr, int d_res_arr_length, int load_float4_time) {
    kernel_4<<<d_res_arr_length, THREAD_PER_BLOCK>>>(d_arr, n, d_res_arr, load_float4_time);
    cudaDeviceSynchronize();
    kernel_4<<<1, THREAD_PER_BLOCK>>>(d_res_arr, d_res_arr_length, d_res, load_float4_time);
    cudaDeviceSynchronize();
}

/**
 * @brief 调用两次kernel，第一次统计block内数据，第二次统计block间数据
 */
void launch_kernel_4(int* d_arr, int* d_res, int* h_res, int n, int cpu_sum) {
    int load_float4_time = 1;
    int values_per_block_before_repeat = 2 * (sizeof(int4_pack) / sizeof(int)) * THREAD_PER_BLOCK;
    while (values_per_block_before_repeat * load_float4_time * values_per_block_before_repeat * load_float4_time < n) {
        load_float4_time++;
    }
    int values_per_block = values_per_block_before_repeat * load_float4_time;
    dim3 grid_dim((n + values_per_block - 1) / values_per_block);
    int* d_res_arr;
    cudaMalloc(&d_res_arr, grid_dim.x * sizeof(int));
    kernel_4_helper(d_arr, d_res, h_res, n, cpu_sum, d_res_arr, grid_dim.x, load_float4_time);
    cudaMemcpy(h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    int gpu_sum = h_res[0];
    cpu_kernel_profiling(kernel_4_helper, std::string(__func__), gpu_sum, cpu_sum, 
                            d_arr, d_res, h_res, n, cpu_sum, d_res_arr, grid_dim.x, load_float4_time);
    cudaFree(d_res_arr);
}

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 1000000;
    printf("Array length = %d\n", N);
    create_arr(N);
    int cpu_sum = cpu_reduce_sum(h_arr, N);
    launch_kernel_0(d_arr, d_res, h_res, N, cpu_sum);
    launch_kernel_1(d_arr, d_res, h_res, N, cpu_sum);
    launch_kernel_2(d_arr, d_res, h_res, N, cpu_sum);
    launch_kernel_3(d_arr, d_res, h_res, N, cpu_sum);
    launch_kernel_4(d_arr, d_res, h_res, N, cpu_sum);
    destroy_arr();
    return 0;
}