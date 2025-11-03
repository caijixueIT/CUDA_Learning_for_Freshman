#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#define THREAD_PER_BLOCK 256
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
__device__ float BlockPrefixSum(float val){
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % 32;
    constexpr int NUM_WARP = THREAD_PER_BLOCK / WARP_SIZE;
    __shared__ float smem[NUM_WARP];
    // warp 内部做 prefix sum
    val = WarpPrefixSum<WARP_SIZE>(val);
    // write sum of each warp to smem
    if(lane == WARP_SIZE - 1) smem[warp_id] = val;
    __syncthreads();
    // 第一个 warp，从 smem 读取数据再做一次prefix sum
    if(warp_id == 0){
        smem[lane] = WarpPrefixSum<NUM_WARP>(smem[lane]);
    }
    __syncthreads();
    if(warp_id >= 1) val += smem[warp_id - 1];
    return val;
}
template<int REDUCE_SIZE>
__device__ float WarpReduceSum(float val) {
    #pragma unroll
    for(int mask = REDUCE_SIZE / 2; mask >= 1; mask >>=1){
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}
__device__ float BlockReduceSum(float val) {
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    val = WarpReduceSum<WARP_SIZE>(val);
    constexpr int NUM_WARP = THREAD_PER_BLOCK / WARP_SIZE;
    __shared__ float smem[NUM_WARP];
    if(lane_id == 0) smem[warp_id] = val;
    __syncthreads();
    val = lane_id < NUM_WARP ? smem[lane_id] : 0;
    __syncthreads();
    val = WarpReduceSum<WARP_SIZE>(val);
    return val;
}
template<int N> 
__global__ void reduce_part_sum(const float* in, float* part){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = tid + bid * THREAD_PER_BLOCK;
    float val = idx < N ? in[idx] : 0;
    float sum = BlockReduceSum(val);
    if(tid == 0) part[bid] = sum;
}
template<int N>
__global__ void acc_part_sum(float* part) {
  int32_t acc = 0;
  constexpr int part_num = N / THREAD_PER_BLOCK;
  for (size_t i = 0; i < part_num; ++i) {
    acc += part[i];
    part[i] = acc;
  }
}
template<int N>
__global__ void prefix_sum_kernel(const float* in, float* output, float* part){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = tid + bid * THREAD_PER_BLOCK;
    float val = idx < N ? in[idx] : 0;
    val = BlockPrefixSum(val);
    // __syncthreads();
    if(bid >= 1) val += part[bid-1];
    if(idx < N) output[idx] = val;
}
bool check(float *out, float *res, int N){
    for(int i=0; i<N; i++){
        // printf("out[%d]=%f, ref[%d]=%f\n", i, out[i], i, res[i]);
        if(out[i]!= res[i])
            return false;
    }
    return true;
}
int main() {
    constexpr int N = 1024 * 1024;
    constexpr int threads_per_block = THREAD_PER_BLOCK;
    float* arr = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        arr[i] = 1;
    }
    float* d_arr, *d_part;
    cudaMalloc((void**)&d_arr, N * sizeof(float));
    cudaMalloc((void**)&d_part, N / THREAD_PER_BLOCK * sizeof(float));
    cudaMemcpy(d_arr, arr, N * sizeof(float), cudaMemcpyHostToDevice);
    float* out = (float*)malloc(N * sizeof(float));
    float* out_ref = (float*)malloc(N * sizeof(float));
    float* d_out;
    cudaMalloc((void**)&d_out, N * sizeof(float));
    float sum = 0;
    for(int i = 0; i < N; i++){
        sum += arr[i];
        out_ref[i] = sum;
    }
    // for (int i = 0; i < N/threads_per_block; ++i) {
    //     float sum = 0.f;
    //     for (int j = threads_per_block - 1; j >= 0; --j) {
    //         sum += arr[i * threads_per_block + j];
    //         out_ref[i * threads_per_block + j] = sum;
    //     }
    // }
    dim3 grid_dim(N / threads_per_block);
    dim3 block_dim(threads_per_block);
    reduce_part_sum<N><<<grid_dim, block_dim>>>(d_arr, d_part);
    acc_part_sum<N><<<grid_dim, block_dim>>>(d_part);
    prefix_sum_kernel<N><<<grid_dim, block_dim>>>(d_arr, d_out, d_part);
    cudaMemcpy(out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (check(out, out_ref, N)) {
        std::cout << "succeed!" << std::endl;
    } else {
        std::cout << "failed!" << std::endl;
        return 1;
    }
    int TEST_TIMES = 100;
    for (int i = 0; i < TEST_TIMES; ++i) {
        reduce_part_sum<N><<<grid_dim, block_dim>>>(d_arr, d_part);
        acc_part_sum<N><<<grid_dim, block_dim>>>(d_part);
        prefix_sum_kernel<N><<<grid_dim, block_dim>>>(d_arr, d_out, d_part);
    }
    float time_elapsed=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);    //记录当前时间
    for (int i = 0; i < TEST_TIMES; ++i) {
        reduce_part_sum<N><<<grid_dim, block_dim>>>(d_arr, d_part);
        acc_part_sum<N><<<grid_dim, block_dim>>>(d_part);
        prefix_sum_kernel<N><<<grid_dim, block_dim>>>(d_arr, d_out, d_part);
    }
    cudaEventRecord(stop,0);    //记录当前时间
    cudaEventSynchronize(start);    //Waits for an event to complete.
    cudaEventSynchronize(stop);     //Waits for an event to complete.Record之前的任务
    cudaEventElapsedTime(&time_elapsed, start, stop);    //计算时间差
    std::cout << "prefix_sum_kernel elasped time = " << time_elapsed/TEST_TIMES << "ms" << std::endl;
    cudaFree(d_arr);
    cudaFree(d_out);
    free(arr);
    free(out);
    free(out_ref);
    return 0;
}