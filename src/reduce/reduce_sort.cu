#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <algorithm>

#define THREAD_PER_BLOCK 256

// 要求数组中任意两个元素不相等
__global__ void sort_kernel(float *arr, float *res, int N)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int val = arr[idx];
    int count = 0;
    __shared__ int smem[THREAD_PER_BLOCK];

    for (int i = tid; i < N; i += THREAD_PER_BLOCK) {
        smem[tid] = arr[i];
        __syncthreads();
        for (int j = 0; j < THREAD_PER_BLOCK; ++j) {
            if (val > smem[j]) {
                count++;
            }
        }
        __syncthreads();
    }

    res[count] = val;
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
    constexpr int N = 1024000;
    constexpr int threads_per_block = THREAD_PER_BLOCK;
    std::vector<float> arr_vec(N);
    std::vector<float> out_ref_vec(N);
    for (int i = 0; i < N; ++i) {
        arr_vec[i] = static_cast<float>(N - i);
        out_ref_vec[i] = arr_vec[i];
    }
    
    float* arr = (float*)arr_vec.data();
    float* out_ref = (float*)out_ref_vec.data();

    float* d_arr;
    cudaMalloc((void**)&d_arr, N * sizeof(float));
    cudaMemcpy(d_arr, arr, N * sizeof(float), cudaMemcpyHostToDevice);

    float* out = (float*)malloc(N * sizeof(float));
    float* d_out;
    cudaMalloc((void**)&d_out, N * sizeof(float));

    // sort by cpu
    std::sort(out_ref_vec.begin(), out_ref_vec.end());

    // sort by cuda
    dim3 grid_dim(N / threads_per_block);
    dim3 block_dim(threads_per_block);
    sort_kernel<<<grid_dim, block_dim>>>(d_arr, d_out, N);
    cudaMemcpy(out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (check(out, out_ref, N)) {
        std::cout << "succeed!" << std::endl;
    } else {
        std::cout << "failed!" << std::endl;
    }

    int TEST_TIMES = 100;
    for (int i = 0; i < TEST_TIMES; ++i) {
        sort_kernel<<<grid_dim, block_dim>>>(d_arr, d_out, N);
    }
    float time_elapsed=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);    //记录当前时间
    for (int i = 0; i < TEST_TIMES; ++i) {
        sort_kernel<<<grid_dim, block_dim>>>(d_arr, d_out, N);
    }
    cudaEventRecord(stop,0);    //记录当前时间
    cudaEventSynchronize(start);    //Waits for an event to complete.
    cudaEventSynchronize(stop);     //Waits for an event to complete.Record之前的任务
    cudaEventElapsedTime(&time_elapsed, start, stop);    //计算时间差
    std::cout << "sort_kernel elasped time = " << time_elapsed/TEST_TIMES << "ms" << std::endl;

    float cpu_time = 0.f;
    for (int i = 0; i < TEST_TIMES; ++i) {
        std::reverse(out_ref_vec.begin(), out_ref_vec.end());
        cudaEventRecord(start,0);    //记录当前时间
        sort_kernel<<<grid_dim, block_dim>>>(d_arr, d_out, N);
        cudaEventRecord(stop,0);    //记录当前时间
        cudaEventSynchronize(start);    //Waits for an event to complete.
        cudaEventSynchronize(stop);     //Waits for an event to complete.Record之前的任务
        cudaEventElapsedTime(&time_elapsed, start, stop);    //计算时间差
        cpu_time += time_elapsed;
    }

    std::cout << "sort_kernel elasped cpu time = " << cpu_time/TEST_TIMES << "ms" << std::endl;

    cudaFree(d_arr);
    cudaFree(d_out);
    free(out);
    return 0;
}