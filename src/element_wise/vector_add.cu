#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256

// transfer vector
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

bool check(float *out, float *res, int N){
    for(int i=0; i<N; i++){
        if(out[i]!= res[i])
            return false;
    }
    return true;
}


__global__ void vector_add_kernel(float* a, float* b, float* c, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

__global__ void vector_add_kernel_float4(float* a, float* b, float* c, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
    if (idx < N) {
        float4 aa = FETCH_FLOAT4(a[idx]);
        float4 bb = FETCH_FLOAT4(b[idx]);
        float4 cc;
        cc.x = aa.x + bb.x;
        cc.y = aa.y + bb.y;
        cc.z = aa.z + bb.z;
        cc.w = aa.w + bb.w;
        FETCH_FLOAT4(c[idx]) = cc;
    }
}

template <int unroll_num>
__global__ void vector_add_kernel_float4_unroll(float* a, float* b, float* c, int N) {
    int offset = (blockIdx.x * blockDim.x * unroll_num + threadIdx.x) << 2;
    
    #pragma unroll
    for (int i = 0; i < unroll_num; ++i) {
        int idx = offset + (i * blockDim.x) << 2;
        if (idx < N) {
            float4 aa = FETCH_FLOAT4(a[idx]);
            float4 bb = FETCH_FLOAT4(b[idx]);
            float4 cc;
            cc.x = aa.x + bb.x;
            cc.y = aa.y + bb.y;
            cc.z = aa.z + bb.z;
            cc.w = aa.w + bb.w;
            FETCH_FLOAT4(c[idx]) = cc;
        }
    }
}

int main() {
    constexpr int N = 32 * 1024 * 1024;
    constexpr int unroll_num = 32;

    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_b = (float*)malloc(N * sizeof(float));
    float* h_c = (float*)malloc(N * sizeof(float));
    float* h_c_ref = (float*)malloc(N * sizeof(float));

    float* d_a;
    float* d_b;
    float* d_c;
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_c, N * sizeof(float));
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // reference by CPU
    for (int i = 0; i < N; ++i) {
        h_c_ref[i] = h_a[i] + h_b[i];
    }
    
    // vector_add_kernel
    dim3 grid_dim(N / THREAD_PER_BLOCK);
    dim3 block_dim(THREAD_PER_BLOCK);

    vector_add_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N);
    cudaMemcpy(d_c, h_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (!check(h_c, h_c_ref, N)) {
        std::cout << "vector_add_kernel error!" <<std::endl;
    } else {
        std::cout << "vector_add_kernel pass!" <<std::endl;
    }

    // vector_add_kernel_float4
    dim3 grid_dim4(N / THREAD_PER_BLOCK / 4);
    dim3 block_dim4(THREAD_PER_BLOCK);
    cudaMemset(d_c, 0, N * sizeof(float));
    vector_add_kernel_float4<<<grid_dim4, block_dim4>>>(d_a, d_b, d_c, N);
    cudaMemcpy(d_c, h_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (!check(h_c, h_c_ref, N)) {
        std::cout << "vector_add_kernel_float4 error!" <<std::endl;
    } else {
        std::cout << "vector_add_kernel_float4 pass!" <<std::endl;
    }

    // vector_add_kernel_float4_unroll
    dim3 grid_dim_u(N / THREAD_PER_BLOCK / 4 / unroll_num);
    dim3 block_dim_u(THREAD_PER_BLOCK);
    cudaMemset(d_c, 0, N * sizeof(float));
    vector_add_kernel_float4_unroll<unroll_num><<<grid_dim_u, block_dim_u>>>(d_a, d_b, d_c, N);
    cudaMemcpy(d_c, h_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (!check(h_c, h_c_ref, N)) {
        std::cout << "vector_add_kernel_float4_unroll error!" <<std::endl;
    } else {
        std::cout << "vector_add_kernel_float4_unroll pass!" <<std::endl;
    }

    // profile
    int TEST_TIMES = 100;
    // warming up
    for (int i = 0; i < TEST_TIMES; ++i) {
        vector_add_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N);
    }

    float time_elapsed=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);    //记录当前时间
    for (int i = 0; i < TEST_TIMES; ++i) {
        vector_add_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop,0);    //记录当前时间
    cudaEventSynchronize(start);    //Waits for an event to complete.
    cudaEventSynchronize(stop);     //Waits for an event to complete.Record之前的任务
    cudaEventElapsedTime(&time_elapsed, start, stop);    //计算时间差
    std::cout << "vector_add_kernel elasped time = " << time_elapsed/TEST_TIMES << std::endl;

    cudaEventRecord(start,0);    //记录当前时间
    for (int i = 0; i < TEST_TIMES; ++i) {
        vector_add_kernel_float4<<<grid_dim4, block_dim4>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop,0);    //记录当前时间
    cudaEventSynchronize(start);    //Waits for an event to complete.
    cudaEventSynchronize(stop);     //Waits for an event to complete.Record之前的任务
    cudaEventElapsedTime(&time_elapsed, start, stop);    //计算时间差
    std::cout << "vector_add_kernel_float4 elasped time = " << time_elapsed/TEST_TIMES << std::endl;

    cudaEventRecord(start,0);    //记录当前时间
    for (int i = 0; i < TEST_TIMES; ++i) {
        vector_add_kernel_float4_unroll<unroll_num><<<grid_dim_u, block_dim_u>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop,0);    //记录当前时间
    cudaEventSynchronize(start);    //Waits for an event to complete.
    cudaEventSynchronize(stop);     //Waits for an event to complete.Record之前的任务
    cudaEventElapsedTime(&time_elapsed, start, stop);    //计算时间差
    std::cout << "vector_add_kernel_float4_unroll elasped time = " << time_elapsed/TEST_TIMES << std::endl;

    // destroy
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_ref);
    return 0;

}

