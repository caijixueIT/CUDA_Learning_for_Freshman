#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>


namespace random_utils {
std::random_device rd;
std::mt19937 gen(rd());
// 定义浮点数分布范围（例如 0.0 到 1.0）
std::uniform_real_distribution<> dis(-10.f, 10.f);

float rand_float() {
    return dis(gen);
}
};


// 每个 block 计算 C 中大小为 BM * BN 的子矩阵, block 个数 = (M / BM) * (N / BN)
// 每个 thread 计算 C 中大小为 TM * TN 的子矩阵, thread 个数 = (BM / TM) * (BN / TN)
// grid(M / BM, N / BN)
// block(BM / TM, BN / TN)
// A [M, K]
// B [K, N]
// C [M, N]
template<int BM=32, int BN=32, int BK=4, int TM=4, int TN=4>
__global__ void gemm(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];
    
    float vals[TM][TN] = {0.f};
    for (int k = 0; k < K / BK; ++k) {
        for (int i = threadIdx.x; i < BM; i += TM) {
            for (int j = threadIdx.y; j < BN; j += TN) {
                int A_row = blockIdx.x * BM + i;
                int B_col = blockIdx.y * BN + j;

                for (int l = threadIdx.y; l < BK; l +=TN) {
                    int A_col = k * BK + l;
                    sA[i][l] = A[A_row * K + A_col];
                }

                for (int l = threadIdx.x; l < BK; l += TM) {
                    int B_row = k * BK + l;
                    sB[l][j] = B[B_row * N + B_col];
                }
            }
        }
        __syncthreads();
        
        for (int i = 0; i < TM; ++i) {
            for (int j = 0; j < TN; ++j) {
                for (int l = 0; l < BK; l++) {
                    vals[i][j] += sA[i * BM / TM + threadIdx.x][l] * sB[l][j * BN / TN + threadIdx.y];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int C_row = blockIdx.x * BM + i * (BM / TM) + threadIdx.x;
            int C_col = blockIdx.y * BN + j * (BN / TN) + threadIdx.y;
            C[C_row * N + C_col] = vals[i][j];
        }
    }
}

void gemm_cpu(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float val = 0.f;
            for (int k = 0; k < K; ++k) {
                val += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = val;
        }
    }
}

void init_matric(float* A, int row, int col) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            A[i * col + j] = random_utils::rand_float();
        }
    }
}

void init_identity_matric(float* A, int row, int col) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            A[i * col + j] = (i == j ? 1.f : 0.f);
        }
    }
}

bool check(float* res_gpu, float* res_cpu, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (fabs(res_gpu[i * N + j] - res_cpu[i * N + j]) > 1e-2) {
                printf("(%d, %d) %f %f\n", i, j, res_gpu[i * N + j], res_cpu[i * N + j]);
                return false;
            }
        }
    }
    return true;
}

void print_matric(float* A, int row, int col) {
    return;
    // for (int i = 0; i < row; ++i) {
    //     for (int j = 0; j < col; ++j) {
    //         printf("%f ", A[i * col + j]);
    //     }
    //     printf("\n");
    // }
}

int main() {
    int M = 1024 * 2;
    int N = 1024 * 2;
    int K = 1024 * 1;
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 8;
    constexpr int TM = 4;
    constexpr int TN = 4;

    // allocate memory in cpu
    float* A = (float*)malloc(M * K * sizeof(float));
    float* B = (float*)malloc(K * N * sizeof(float));
    float* C = (float*)malloc(M * N * sizeof(float));
    float* C_ref = (float*)malloc(M * N * sizeof(float));

    // init cpu data
    init_matric(A, M, K);
    init_matric(B, K, N);
    print_matric(A, M, K);
    print_matric(B, K, N);

    // allocate memory in gpu
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // init gpu data
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // gemm by gpu
    dim3 grid(M/BM, N/BN);
    dim3 block(BM/TM, BN/TN);
    gemm<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    print_matric(C, M, N);

    // gemm by cpu
    gemm_cpu(A, B, C_ref, M, N, K);
    print_matric(C_ref, M, N);

    // check
    if (check(C, C_ref, M, N)) {
        printf("pass\n");
    } else {
        printf("fail\n");
    }

    // profile
    int TEST_TIMES = 100;
    for (int i = 0; i < TEST_TIMES; ++i) {
        gemm<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    float time_elapsed=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0); // 记录开始时间
    for (int i = 0; i < TEST_TIMES; ++i) {
        gemm<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop, 0); // 记录结束时间
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop); // 计算时间差
    std::cout << "elasped time = " << time_elapsed / TEST_TIMES << "ms" << std::endl;

    return 0;
}