#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];
    
    // 将结果 M*N 的矩阵分块成 BM*BM 的矩阵，每个线程块负责计算一个 BM*TN 的矩阵
    // 这里得到当前线程块负责的 BM*BN 的矩阵的索引
    int bRow = blockIdx.x;
    int bCol = blockIdx.y;
    
    // 将当前线程快负责的 BM*BN 的结果子矩阵再次分块为 TM*TN 的小块，每个线程负责计算一个 TM*TN 的矩阵
    // 这里得到当前线程负责的 TM*TN 的矩阵的索引
    int tRow = threadIdx.x / (BN / TN);
    int tCol = threadIdx.x % (BN / TN);

    // 线程块内线程数量
    constexpr int numThreadsBlocktile = (BM * BN) / (TM * TN);

    // 每个线程负责拷贝 A 中的 BM * BK / numThreadsBlocktile 个元素，从 global memory 到 shared memory
    constexpr int valsPerThreadA = BM * BK / numThreadsBlocktile;
    int sRowA = threadIdx.x / (BK / valsPerThreadA);
    int sColA = threadIdx.x % (BK / valsPerThreadA);

    // 每个线程负责拷贝 B 中的 BK * BN / numThreadsBlocktile 个元素，从 global memory 到 shared memory
    constexpr int valsPerThreadB = BK * BN / numThreadsBlocktile;
    int sRowB = threadIdx.x / (BN / valsPerThreadB);
    int sColB = threadIdx.x % (BN / valsPerThreadB);

    // 当前线程所需寄存器
    float threadResult[TM * TN] = {0};
    float regA[TM];
    float regB[TN];

    // A B C 的起始位置
    A += bRow * BM * K;
    B += bCol * BN;
    C += bRow * BM * N + bCol * BN;
    
    // K 维度循环
    for (int kidx = 0; kidx < K / BK; kidx++) {
        for (int i = 0; i < valsPerThreadA; i += 4) {
            FETCH_FLOAT4(sA[sRowA * BK + sColA * valsPerThreadA + i]) = FETCH_FLOAT4(A[sRowA * K + kidx * BK + sColA * valsPerThreadA + i]);
        }
        for (int j = 0; j < valsPerThreadB ; j += 4) {
            FETCH_FLOAT4(sB[sRowB * BN + sColB * valsPerThreadB + j]) = FETCH_FLOAT4(B[kidx * BK * N + sRowB * N + sColB * valsPerThreadB + j]);
        }
        // for (int i = 0; i < valsPerThreadA; i++) {
        //     sA[sRowA * BK + sColA * valsPerThreadA + i] = A[sRowA * K + kidx * BK + sColA * valsPerThreadA + i];
        // }
        // for (int j = 0; j < valsPerThreadB; j++) {
        //     sB[sRowB * BN + sColB * valsPerThreadB + j] = B[kidx * BK * N + sRowB * N + sColB * valsPerThreadB + j];
        // }
        __syncthreads();
        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i++) {
                regA[i] = sA[(tRow * TM + i) * BK + k];
            }
            for (int j = 0; j < TN; j++) {
                regB[j] = sB[k * BN + tCol * TN + j];
            }
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    threadResult[i * TN + j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; j += 4) {
            FETCH_FLOAT4(C[(tRow * TM + i) * N + tCol * TN + j]) = FETCH_FLOAT4(threadResult[i * TN + j]);
        }
    }
    // for (int i = 0; i < TM; ++i) {
    //     for (int j = 0; j < TN; ++j) {
    //         C[(tRow * TM + i) * N + tCol * TN + j] = threadResult[i * TN + j];
    //     }
    // }
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

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 grid(M / BM, N / BN); // 每个 block 负责计算结果中一个 BM*BN 的子矩阵
    dim3 block((BM / TM) * (BN / TN)); // 每个线程负责计算结果中一个 TM*TN 的子矩阵
    gemm<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    gemm_cpu(A, B, C_ref, M, N, K);

    if (check(C, C_ref, M, N)) {
        printf("Correct!\n");
    } else {
        printf("Wrong!\n");
    }

    int TEST_TIMES = 100;
    for (int i = 0; i < TEST_TIMES; ++i) {
        gemm<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    float time_elapsed=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);                               // 记录开始时间
    for (int i = 0; i < TEST_TIMES; ++i) {
        gemm<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop,0);                                // 记录结束时间
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);       // 计算时间差
    std::cout << "elasped time = " << time_elapsed/TEST_TIMES << "ms" << std::endl;
    return 0;
}