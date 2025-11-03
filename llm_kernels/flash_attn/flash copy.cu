#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define INFINITY 1e9

// Q、K、V shape：[batch_size, num_heads, seq_len, hidden_dim]
// grid(batch * num_heads, (seq_len + BM - 1) / BM)
// block(THREADS_PER_BLOCK)
template<int BM=16, int BN=16, int HIDDEN_DIM=128, int THREADS_PER_BLOCK=256>
__global__ void flash(float* Q, float* K, float* V, float* O, int seq_len, float softmax_scale) {
    Q += blockIdx.x * HIDDEN_DIM * seq_len + blockIdx.y * HIDDEN_DIM * BM;
    O += blockIdx.x * HIDDEN_DIM * seq_len + blockIdx.y * HIDDEN_DIM * BM;

    K += blockIdx.x * HIDDEN_DIM * seq_len;
    V += blockIdx.x * HIDDEN_DIM * seq_len;
    
    int tx = threadIdx.x;
    int row = tx / BN;
    int col = tx % BN;
    __shared__ float s_Q[BM][HIDDEN_DIM];
    __shared__ float s_O[BM][HIDDEN_DIM];

    __shared__ float s_K[BN][HIDDEN_DIM];
    __shared__ float s_V[BN][HIDDEN_DIM];

    __shared__ float sQK[BM][BN];
    __shared__ float sSafeExp[BM][BN];
    __shared__ float sDenom[BM];
    __shared__ float sMax[BM];

    for (int i = tx; i < BM * HIDDEN_DIM; i += THREADS_PER_BLOCK) {
        s_Q[tx / HIDDEN_DIM][tx % HIDDEN_DIM] = Q[i];
        s_O[tx / HIDDEN_DIM][tx % HIDDEN_DIM] = 0.f;
    }

    for (int i = tx; i < BM; i += THREADS_PER_BLOCK) {
        sMax[i] = -INFINITY;
        sDenom[i] = 0.f;

    }

    int num_kv_blocks = (seq_len + BN - 1) / BN;
    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
        // 1. load K and V
        for (int i = tx; i < BN * HIDDEN_DIM; i += THREADS_PER_BLOCK) {
            s_K[tx / HIDDEN_DIM][tx % HIDDEN_DIM] = K[kv_block_idx * BN * HIDDEN_DIM + i];
            s_V[tx / HIDDEN_DIM][tx % HIDDEN_DIM] = V[kv_block_idx * BN * HIDDEN_DIM + i];
        }
        __syncthreads();

        // 2. compute QK
        float val = 0.f;
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            val += s_Q[row][j] * s_K[col][j];
        }
        sQK[row][col] = val * softmax_scale;
        __syncthreads();

        // 3. local row max
        float local_row_max = -INFINITY;
        for (int i = 0; i < BN; ++i) {
            local_row_max = max(local_row_max, sQK[row][i]);
        }
        __syncthreads();

        // 4. global row max
        float global_row_max = max(local_row_max, sMax[row]);
        sExp[][]






        // 3. get local max
        for (int i = tx; i < BM * BN; i += THREADS_PER_BLOCK) {
            int qk_row = i / BN;
            float 
        }


        







    }
}


void launch_flash_kernel(float* Q, float* K, float* V, float* O, int batch_size, int num_heads, int seq_len, int hidden_dim) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int THREADS_PER_BLOCK = 128;
    constexpr int HIDDEN_DIM = 128;

    dim3 grid(batch_size * num_heads, (seq_len + BM - 1) / BM);
    dim3 block(THREADS_PER_BLOCK);

    flash<BM, BN, HIDDEN_DIM, THREADS_PER_BLOCK><<<grid, block>>>(Q, K, V, O, batch_size, num_heads, seq_len, hidden_dim);
}