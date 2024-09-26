#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define BLOCK_SIZE 32


__global__ void gemm(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float sA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE * BLOCK_SIZE];

    int tRow = threadIdx.x / BLOCK_SIZE;
    int tCol = threadIdx.x % BLOCK_SIZE;

    int rowA = blockIdx.x * BLOCK_SIZE + tRow;
    int colB = blockIdx.y * BLOCK_SIZE + tCol;

    int kTileSize = K / BLOCK_SIZE;
    float val = 0;
    for (int kidx = 0; kidx < kTileSize; kidx++) {
        int colA = kidx * BLOCK_SIZE + tCol;
        int rowB = kidx * BLOCK_SIZE + tRow;
        sA[tRow * BLOCK_SIZE + tCol] = A[rowA * K + colA];
        sB[tRow * BLOCK_SIZE + tCol] = B[rowB * N + colB];
        __syncthreads();
        for (int i = 0; i < BLOCK_SIZE; i++) {
            val += sA[tRow * BLOCK_SIZE + i] * sB[i * BLOCK_SIZE + tCol];
        }
        __syncthreads();
    }

    C[rowA * N + colB] = val;
}

void gemm_cpu(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float val = 0;
            for (int k = 0; k < K; k++) {
                val += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = val;
        }
    }
}

int check(float* C, float* C_ref, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // if (C[i * N + j] != C_ref[i * N + j]) {
            //     return 0;
            // }
            if (fabs(C[i * N + j] - C_ref[i * N + j]) > 0.001) {
                return 0;
            }
        }
    }
    return 1;
}

int main() {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    float* A = (float*)malloc(M * K * sizeof(float));
    float* B = (float*)malloc(K * N * sizeof(float));
    float* C = (float*)malloc(M * N * sizeof(float));
    float* C_ref = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * K; i++) {
        A[i] = rand() % 17;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = rand() % 23;
    }
    for (int i = 0; i < M * N; i++) {
        C[i] = 0;
    }
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(M / BLOCK_SIZE, N / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE * BLOCK_SIZE);
    gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    gemm_cpu(A, B, C_ref, M, N, K);

    if (check(C, C_ref, M, N)) {
        printf("Correct!\n");
    } else {
        printf("Wrong!\n");
    }

    int TEST_TIMES = 100;
    for (int i = 0; i < TEST_TIMES; ++i) {
        gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    float time_elapsed=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);                               // 记录开始时间
    for (int i = 0; i < TEST_TIMES; ++i) {
        gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop,0);                                // 记录结束时间
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);       // 计算时间差
    std::cout << "elasped time = " << time_elapsed/TEST_TIMES << "ms" << std::endl;


    return 0;

}