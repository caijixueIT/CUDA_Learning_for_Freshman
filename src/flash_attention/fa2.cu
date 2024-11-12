#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <cublas_v2.h>

#define INFINITY 3.402823466e+38F

/*
    Q,K,V,O shape = [batch_size, num_heads, seq_len, head_dim]
    grid (batch_size * num_heads, (seq_len + BM - 1) / BM)
    block (BN, BM)
    assume seq_len % BM == 0, seq_len % BN == 0
*/
template <typename T, int BM=16, int BN=16, int kHeadDim=128>
__global__ void fa2(T* Q, T* K, T* V, T* O, int seq_len, float softmax_scale) {

    // (0, 1, ..., BN-1)
    int tx = threadIdx.x;
    // (0, 1, ..., BM-1)
    int ty = threadIdx.y;
    int row = blockIdx.y * BM + threadIdx.y;
    if (row >= seq_len) return;

    int base_offset = blockIdx.x * seq_len * kHeadDim;
    Q += base_offset;
    K += base_offset;
    V += base_offset;
    O += base_offset;

    __shared__ T sQ[BM][kHeadDim];
    __shared__ T sK[BN][kHeadDim];
    __shared__ T sV[BN][kHeadDim];
    __shared__ T sO[BM][kHeadDim];
    __shared__ T sQK[BM][BN];
    __shared__ T sSafeE[BM][BN];
    __shared__ T sDenom[BM];
    __shared__ T sMax[BM];

    // load Q, O from global memory to shared memory
    for (int i = tx; i < kHeadDim; i +=BN) {
        sQ[ty][i] = Q[row * kHeadDim + i];
        sO[ty][i] = 0;
    }

    sMax[ty] = -INFINITY;
    sDenom[ty] = 0.f;
    
    // iteration over KV blocks
    for (int kv_row = tx; kv_row < seq_len; kv_row += BN) {
        // load K, V from global memory to shared memory
        for (int i = ty; i < kHeadDim; i +=BM) {
            sK[tx][i] = K[kv_row * kHeadDim + i];
            sV[tx][i] = V[kv_row * kHeadDim + i];
        }
        __syncthreads();

        // compute QK
        T qk = 0.f;
        for (int i = 0; i < kHeadDim; i++) {
            qk += sQ[ty][i] * sK[tx][i];
        }
        sQK[ty][tx] = qk * softmax_scale;
        __syncthreads();
        
        // local max
        T local_max = -INFINITY;
        for (int i = 0; i < BN; i++) {
            local_max = max(local_max, sQK[ty][i]);
        }
        __syncthreads();

        T global_max = max(local_max, sMax[ty]);
        sSafeE[ty][tx] = exp(sQK[ty][tx] - global_max);
        __syncthreads();

        // local denom
        T local_denom = 0.f;
        for (int i = 0; i < BN; i++) {
            local_denom += sSafeE[ty][i];
        }
        __syncthreads();

        T rescale_factor = exp(sMax[tx] - global_max);
        T global_denom = sDenom[ty] * rescale_factor + local_denom;
        __syncthreads();

        // update output
        for (int i = tx; i < kHeadDim; i +=BN) {
            sO[ty][i] *= rescale_factor;
            for (int j = 0; j < BN; j++) {
                sO[ty][i] += sSafeE[ty][j] * sV[j][i];
            }
        }
        
        // update max and denom
        sMax[ty] = global_max;
        sDenom[ty] = global_denom;
        __syncthreads();
    }
    // result storing s2g
    for (int i = tx; i < kHeadDim; i +=BN) {
        O[row * kHeadDim + i] = sO[ty][i] / sDenom[ty];
    }
}


int main() {
    int batch = 1;
    int num_heads = 32;
    constexpr int head_dim = 64;
    int seq_len = 1024;

    int N = batch * num_heads * seq_len * head_dim;
    float* q = (float*)malloc(N * sizeof(float));
    float* k = (float*)malloc(N * sizeof(float));
    float* v = (float*)malloc(N * sizeof(float));
    float* o = (float*)malloc(N * sizeof(float));
    float* ref_o = (float*)malloc(N * sizeof(float));

    float *d_q, *d_k, *d_v, *d_o;
    cudaMalloc((void**)&d_q, N * sizeof(float));
    cudaMalloc((void**)&d_q, N * sizeof(float));
    cudaMalloc((void**)&d_q, N * sizeof(float));
    cudaMalloc((void**)&d_o, N * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        q[i] = rand() % 11;
        k[i] = rand() % 5;
        v[i] = rand() % 7;
    }

    cudaMemcpy(d_q, q, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, N * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int BM = 16;
    constexpr int BN = 16;
    dim3 grid(batch * num_heads, (seq_len + BM - 1) / BM);
    dim3 block(BN, BM);
    float softmax_scale = 1.f / sqrt(head_dim);
    fa2<float, BM, BN, head_dim><<<grid, block>>>(d_q, d_k, d_v, d_o, seq_len, softmax_scale);
    cudaDeviceSynchronize();
    cudaMemcpy(o, d_o, N * sizeof(float), cudaMemcpyDeviceToHost);
    return 0;
}

