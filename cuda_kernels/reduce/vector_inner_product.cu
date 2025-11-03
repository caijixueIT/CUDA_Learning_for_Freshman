/*
 * @Description: 计算大数组的累加和
 * @LastEditors: Bruce Li
 * @LastEditTime: 2025-07-30 16:41
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
 
 #define THREADS_PER_BLOCK 256


 float random_float(float min=-10.f, float max=10.f) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}

void init_vector(float* tensor, int n) {
    for (int i = 0; i < n; i++) {
        tensor[i] = random_float();
    }
}

 __global__ void vector_inner_product(float* vec_a, float* vec_b, float* inner_product, int n) { 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float smem[THREADS_PER_BLOCK];

    float val_a = idx < n ? vec_a[idx] : 0.0f;
    float val_b = idx < n ? vec_b[idx] : 0.0f;
    float product = val_a * val_b;
    smem[threadIdx.x] = product;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(inner_product, smem[0]);
    }
 }

 void launch_inner_product_kernel(float* d_vec_a, float* d_vec_b, float* d_inn, float* h_inn, int n) { 
    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vector_inner_product<<<num_blocks, THREADS_PER_BLOCK>>>(d_vec_a, d_vec_b, d_inn, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_inn, d_inn, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
 }

 float cpu_inner_product(float* vec_a, float* vec_b, int n) {
    float inn = 0.0f;
    for (int i = 0; i < n; i++) {
        inn += vec_a[i] * vec_b[i];
    }
    return inn;
 }

 int main(int argc, char** argv) {
    int N = 10000;
    if (argc == 2) {
        N = atoi(argv[1]);
    }
    printf("向量长度 = %d\n", N);

    float* h_vec_a = (float*)malloc(N * sizeof(float));
    float* h_vec_b = (float*)malloc(N * sizeof(float));
    init_vector(h_vec_a, N);
    init_vector(h_vec_b, N);
    float h_inn;
    float h_ref =cpu_inner_product(h_vec_a, h_vec_b, N);

    float* d_vec_a;
    cudaMalloc((void**)&d_vec_a, N * sizeof(float));
    cudaMemcpy(d_vec_a, h_vec_a, N * sizeof(float), cudaMemcpyHostToDevice);
    float* d_vec_b;
    cudaMalloc((void**)&d_vec_b, N * sizeof(float));
    cudaMemcpy(d_vec_b, h_vec_b, N * sizeof(float), cudaMemcpyHostToDevice);
    float* d_inn;
    cudaMalloc((void**)&d_inn, sizeof(float));

    launch_inner_product_kernel(d_vec_a, d_vec_b, d_inn, &h_inn, N);
    if (fabs(h_inn - h_ref) < 1e-2) {
        printf("PASS\n");
    } else {
        printf("FAIL, gpu = %f, cpu = %f\n", h_inn, h_ref);

    }
    return 0;
 }