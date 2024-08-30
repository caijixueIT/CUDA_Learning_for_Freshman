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

    smem[tid] = arr[idx];
    __syncthreads();

    for (int i = 1; i < THREAD_PER_BLOCK; i = i * 2) {
        int x = tid * i * 2;
        if (x < THREAD_PER_BLOCK) {
            smem[x] += smem[x + i];
        }
        __syncthreads();
    }

    if (tid == 0) {
        res[blockIdx.x] = smem[0];
    }
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
        arr[i] = 1; // i * (N - 1) / (float) (N * 10);
    }
    
    float* d_arr;
    cudaMalloc((void**)&d_arr, N * sizeof(float));
    cudaMemcpy(d_arr, arr, N * sizeof(float), cudaMemcpyHostToDevice);

    float* out = (float*)malloc(N / threads_per_block * sizeof(float));
    float* out_ref = (float*)malloc(N / threads_per_block * sizeof(float));

    float* d_out;
    cudaMalloc((void**)&d_out, N / threads_per_block * sizeof(float));

    for (int i = 0; i < N / threads_per_block; ++i) {
        float sum = 0.f;
        for (int j = 0; j < threads_per_block; ++j) {
            sum += arr[i * threads_per_block + j];
        }
        out_ref[i] = sum;
    }

    dim3 grid_dim(N / threads_per_block);
    dim3 block_dim(threads_per_block);

    reduce_sum_kernel<<<grid_dim, block_dim>>>(d_arr, N, d_out);
    cudaMemcpy(out, d_out, N/threads_per_block * sizeof(float), cudaMemcpyDeviceToHost);

    if (check(out, out_ref, N/threads_per_block)) {
        std::cout << "succeed!" << std::endl;
    } else {
        std::cout << "failed!" << std::endl;
    }

    int TEST_TIMES = 100;
    for (int i = 0; i < TEST_TIMES; ++i) {
        reduce_sum_kernel<<<grid_dim, block_dim>>>(d_arr, N, d_out);
    }
    float time_elapsed=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);    //记录当前时间
    for (int i = 0; i < TEST_TIMES; ++i) {
        reduce_sum_kernel<<<grid_dim, block_dim>>>(d_arr, N, d_out);
    }
    cudaEventRecord(stop,0);    //记录当前时间
    cudaEventSynchronize(start);    //Waits for an event to complete.
    cudaEventSynchronize(stop);     //Waits for an event to complete.Record之前的任务
    cudaEventElapsedTime(&time_elapsed, start, stop);    //计算时间差
    std::cout << "reduce_sum_kernel elasped time = " << time_elapsed/TEST_TIMES << "ms" << std::endl;

    cudaFree(d_arr);
    cudaFree(d_out);
    free(arr);
    free(out);
    free(out_ref);
    return 0;
}