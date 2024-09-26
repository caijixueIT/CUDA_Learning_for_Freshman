#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <cublas_v2.h>

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
            // if (i < 10 && j < 10) {
            //     printf("%f %f\n", C[i * N + j], C_ref[i * N + j]);
            // }
            if (fabs(C[i * N + j] - C_ref[i * N + j]) > 0.0001) {
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
        A[i] = (rand() % 17) / 23.3;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (rand() % 23) / 17.7;
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

    cudaError_t cudaStat;  // cudaMalloc status
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle; // cuBLAS context

    float alpha = 1.0;
    float beta = 0.0;
    stat = cublasCreate(&handle); // initialize CUBLAS context
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    gemm_cpu(A, B, C_ref, M, N, K);

    if (check(C, C_ref, M, N)) {
        printf("Correct!\n");
    } else {
        printf("Wrong!\n");
    }

    int TEST_TIMES = 100;
    for (int i = 0; i < TEST_TIMES; ++i) {
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    float time_elapsed=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);                               // 记录开始时间
    for (int i = 0; i < TEST_TIMES; ++i) {
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    cudaEventRecord(stop,0);                                // 记录结束时间
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);       // 计算时间差
    std::cout << "elasped time = " << time_elapsed/TEST_TIMES << "ms" << std::endl;
    return 0;
}