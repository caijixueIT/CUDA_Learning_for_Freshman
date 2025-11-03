/*
 * @Description: 矩阵转置
 * @LastEditors: Bruce Li
 * @LastEditTime: 2025-07-30 17:58
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

#define BLOCK 32
#define WARMING_TIMES 50
#define PROFILING_TIMES 100

#define CUDA_CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

bool check(float* matrix_trans, float* matrix_trans_cpu, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (fabs(matrix_trans[i * col + j] - matrix_trans_cpu[i * col + j]) > 1e-10) {
                return false;
            }
        }
    }
    return true;
}

void cpu_transpose(float* matrix, float* matrix_trans, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            matrix_trans[j * row + i] = matrix[i * col + j];
        }
    }
}

template <typename Kernel, typename... Args, int WARMUP=50, int PROFILE=100>
void kernel_profiling(Kernel kernel, dim3 grid, dim3 block, std::string fun_name, 
                        float* gpu_trans, float* cpu_trans, int row, int col, Args&&... args) {
    std::cout << std::endl << std::string(100, '=') << std::endl;
    // check
    if (check(gpu_trans, cpu_trans, row, col)) {
        printf("%s PASS\n", fun_name.c_str());
    } else {
        printf("%s FAIL\n", fun_name.c_str());
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
// base line
template <int BLOCK_SIZE=32>
__global__ void transpose_0(float *matrix, float *matrix_trans, int row, int col) {
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index
    if (x < col && y < row) temp[threadIdx.y][threadIdx.x] = matrix[y * col + x];
    __syncthreads();
    if (x < col && y < row) matrix_trans[x * row + y] = temp[threadIdx.y][threadIdx.x];
}

void launch_transpose_kernel_0(float* d_matrix, float* d_matrix_trans, float* h_matrix_trans, 
                                float* h_matrix_trans_cpu, int row, int col) {
    dim3 block(BLOCK, BLOCK);
    dim3 grid(ceil(col / (float)block.x), ceil(row / (float)block.y));
    transpose_0<BLOCK><<<grid, block>>>(d_matrix, d_matrix_trans, row, col);
    cudaMemcpy(h_matrix_trans, d_matrix_trans, row * col * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    kernel_profiling(transpose_0<BLOCK>, grid, block, std::string(__func__), h_matrix_trans, h_matrix_trans_cpu, row, col,
                        d_matrix, d_matrix_trans, row, col);
}

// solve bank conflict
template <int BLOCK_SIZE=32>
__global__ void transpose_1(float *matrix, float *matrix_trans, int row, int col) {
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE + 1];
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index
    if (x < col && y < row) temp[threadIdx.y][threadIdx.x] = matrix[y * col + x];
    __syncthreads();
    if (x < col && y < row) matrix_trans[x * row + y] = temp[threadIdx.y][threadIdx.x];
}

void launch_transpose_kernel_1(float* d_matrix, float* d_matrix_trans, float* h_matrix_trans, 
                                float* h_matrix_trans_cpu, int row, int col) {
    dim3 block(BLOCK, BLOCK);
    dim3 grid(ceil(col / (float)block.x), ceil(row / (float)block.y));
    transpose_1<BLOCK><<<grid, block>>>(d_matrix, d_matrix_trans, row, col);
    cudaMemcpy(h_matrix_trans, d_matrix_trans, row * col * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    kernel_profiling(transpose_1<BLOCK>, grid, block, std::string(__func__), h_matrix_trans, h_matrix_trans_cpu, row, col,
                        d_matrix, d_matrix_trans, row, col);
}

// Coalesced Memory Access
template <int BLOCK_SIZE=32>
__global__ void transpose_2(float *matrix, float *matrix_trans, int row, int col) {
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE + 1];
    int r = threadIdx.x / BLOCK_SIZE;
    int c = threadIdx.x % BLOCK_SIZE;
    int x = blockIdx.x * blockDim.x + c; // col index
    int y = blockIdx.y * blockDim.y + r; // row index
    if (x < col && y < row) temp[c][r] = matrix[y * col + x];
    __syncthreads();
    if (x < col && y < row) matrix_trans[x * row + y] = temp[c][r];
}

void launch_transpose_kernel_2(float* d_matrix, float* d_matrix_trans, float* h_matrix_trans, 
                                float* h_matrix_trans_cpu, int row, int col) {
    dim3 block(BLOCK * BLOCK);
    dim3 grid(ceil(col / BLOCK), ceil(row / BLOCK));
    transpose_2<BLOCK><<<grid, block>>>(d_matrix, d_matrix_trans, row, col);
    cudaMemcpy(h_matrix_trans, d_matrix_trans, row * col * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    kernel_profiling(transpose_2<BLOCK>, grid, block, std::string(__func__), h_matrix_trans, h_matrix_trans_cpu, row, col,
                        d_matrix, d_matrix_trans, row, col);
}

// Coalesced Memory Access
template <int BLOCK_SIZE=32>
__global__ void transpose_3(float *matrix, float *matrix_trans, int row, int col) {
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
    int r = threadIdx.x; // 0, 1, ..., BLOCK_SIZE - 1
    int c = threadIdx.y; // 0, 1, ..., BLOCK_SIZE / 4 - 1
    int x = blockIdx.x * blockDim.x + c * 4; // col index
    int y = blockIdx.y * blockDim.y + r; // row index

    if (y < row) {
        if (x + 3 < col) {
            float4 temp_float4 = *((float4*)(&(matrix[y * col + x])));
            temp[4 * c][r] = temp_float4.x;
            temp[4 * c + 1][r] = temp_float4.y;
            temp[4 * c + 2][r] = temp_float4.z;
            temp[4 * c + 3][r] = temp_float4.w;
        } else if (x + 3 == col) {
            float2 temp_float2 = *((float2*)(&(matrix[y * col + x])));
            temp[4 * c][r] = temp_float2.x;
            temp[4 * c + 1][r] = temp_float2.y;
            temp[4 * c + 2][r] = matrix[y * col + x + 2];
        } else if (x + 2 == col) {
            float2 temp_float2 = *((float2*)(&(matrix[y * col + x])));
            temp[4 * c][r] = temp_float2.x;
            temp[4 * c + 1][r] = temp_float2.y;
        } else if (x + 1 == col) {
            temp[4 * c][r] = matrix[y * col + x];
        }
    }
    __syncthreads();

    if (y < row) {
        if (x < col) {
            matrix_trans[x * row + y] = temp[4 * c][r];
        }
        if (x + 1 < col) {
            matrix_trans[(x + 1) * row + y] = temp[4 * c + 1][r];
        }
        if (x + 2 < col) {
            matrix_trans[(x + 2) * row + y] = temp[4 * c + 2][r];
        }
        if (x + 3 < col) {
            matrix_trans[(x + 3) * row + y] = temp[4 * c + 3][r];
        }
    }
}

void launch_transpose_kernel_3(float* d_matrix, float* d_matrix_trans, float* h_matrix_trans, 
                                float* h_matrix_trans_cpu, int row, int col) {
    dim3 block(BLOCK, BLOCK / 4);
    dim3 grid(ceil(col / BLOCK), ceil(row / BLOCK));
    transpose_3<BLOCK><<<grid, block>>>(d_matrix, d_matrix_trans, row, col);
    cudaMemcpy(h_matrix_trans, d_matrix_trans, row * col * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    kernel_profiling(transpose_3<BLOCK>, grid, block, std::string(__func__), h_matrix_trans, h_matrix_trans_cpu, row, col,
                        d_matrix, d_matrix_trans, row, col);
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main(int argc, char** argv) { 
    int row = 1024;
    int col = 1024;
    if (argc == 3) {
        row = atoi(argv[1]);
        col = atoi(argv[2]);
    }

    float* h_matrix = (float*)malloc(row * col * sizeof(float));
    float* h_matrix_trans = (float*)malloc(row * col * sizeof(float));
    float* h_matrix_trans_cpu = (float*)malloc(row * col * sizeof(float));
    float* d_matrix;
    float* d_matrix_trans;
    cudaMalloc((void**)&d_matrix, row * col * sizeof(float));
    cudaMalloc((void**)&d_matrix_trans, row * col * sizeof(float));
    init_vector(h_matrix, row * col);
    cudaMemcpy(d_matrix, h_matrix, row * col * sizeof(float), cudaMemcpyHostToDevice);

    cpu_transpose(h_matrix, h_matrix_trans_cpu, row, col);
    launch_transpose_kernel_0(d_matrix, d_matrix_trans, h_matrix_trans, h_matrix_trans_cpu, row, col);
    launch_transpose_kernel_1(d_matrix, d_matrix_trans, h_matrix_trans, h_matrix_trans_cpu, row, col);
    launch_transpose_kernel_2(d_matrix, d_matrix_trans, h_matrix_trans, h_matrix_trans_cpu, row, col);
    launch_transpose_kernel_3(d_matrix, d_matrix_trans, h_matrix_trans, h_matrix_trans_cpu, row, col);
    return 0;
}