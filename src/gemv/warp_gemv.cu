#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)

template <typename T, unsigned int WarpSize>
__device__ __forceinline__ T warpReduceSum(T sum) {
    if (WarpSize >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (WarpSize >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (WarpSize >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (WarpSize >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (WarpSize >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

/*
    A : M * N
    x : N * 1
    y : M * 1
    一个 warp 处理多行
*/
template <int ROWS_PER_WARP = 4>
__global__ void gemv_warp(float *A, float *x, float *y, int M, int N) {
    int tid = threadIdx.x;
    int laneId = tid % 32;
    int warpId = tid / 32;

    constexpr int kWarpSize = 32 / ROWS_PER_WARP; // 把一个线程 warp 分成更小的 group
    int kLaneId = laneId % kWarpSize;  // group 内的线程编号
    int kWarpId = laneId / kWarpSize;  // group id

    int row = blockIdx.x * WARPS_PER_BLOCK * ROWS_PER_WARP + warpId * ROWS_PER_WARP + kWarpId;
    if (row >= M) return;

    int col = kLaneId * 4;
    float sum = 0;
    for (int i = col; i < N; i += kWarpSize * 4) {
        float4 a = *(float4*)(A + row * N + i);
        float4 b = *(float4*)(x + i);
        sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    sum = warpReduceSum<float, kWarpSize>(sum);
    if (kLaneId == 0) y[row] = sum;
}

void gemv_cpu(float *A, float *x, float *y, int M, int N) {
    for (int i = 0; i < M; i++) {
        float sum = 0;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

int check(float* y, float *y_ref, int M) {
    for (int i = 0; i < M; i++) {
        if (fabs(y_ref[i] - y[i]) > 1e-5) {
            printf("error at %d: %f %f\n", i, y_ref[i], y[i]);
            return 0;
        }
    }
    return 1;
}

int main() {
    int M = 4096;
    int N = 1024;
    float *A = (float*)malloc(M * N * sizeof(float));
    float *x = (float*)malloc(N * sizeof(float));
    float *y = (float*)malloc(M * sizeof(float));
    float *y_ref = (float*)malloc(M * sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = rand() % 11;
        }
    }

    for (int i = 0; i < N; i++) {
        x[i] = rand() % 13;
    }

    gemv_cpu(A, x, y_ref, M, N);

    float *d_A, *d_x, *d_y;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));
    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int ROWS_PER_WARP = 4;
    constexpr int ROWS_PER_BLOCK = ROWS_PER_WARP * WARPS_PER_BLOCK;

    const int blocks = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    gemv_warp<ROWS_PER_WARP><<<blocks, THREADS_PER_BLOCK>>>(d_A, d_x, d_y, M, N);
    cudaMemcpy(y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if (check(y, y_ref, M)) printf("passed\n");
    else printf("failed\n");

    int TEST_TIMES = 100;
    for (int i = 0; i < TEST_TIMES; ++i) {
        gemv_warp<ROWS_PER_WARP><<<blocks, THREADS_PER_BLOCK>>>(d_A, d_x, d_y, M, N);
    }
    float time_elapsed=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);                               // 记录开始时间
    for (int i = 0; i < TEST_TIMES; ++i) {
        gemv_warp<ROWS_PER_WARP><<<blocks, THREADS_PER_BLOCK>>>(d_A, d_x, d_y, M, N);
    }
    cudaEventRecord(stop,0);                                // 记录结束时间
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);       // 计算时间差
    std::cout << "elasped time = " << time_elapsed/TEST_TIMES << "ms" << std::endl;

    return 0;
}