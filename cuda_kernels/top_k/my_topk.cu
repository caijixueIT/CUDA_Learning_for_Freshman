#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <random>
#include <chrono>
#include <queue>
#include <vector>

#define BLOCK_SISE 256
#define TOP_K 8
#define MIN_FLOAT -1e20

/**
 * @brief: top_k kernel, 在输入的 dim 维度上进行 top_k 操作，每个 thread block 处理行
 * @output: [batch_size, TopK]
 * @indices: [batch_size, TopK]
 * @input: [batch_size, dim]
 * @batch_size:
 * @dim:
 */
template <int TopK=TOP_K, int BlockSize=BLOCK_SISE, int RowsPerBlock=1>
__global__ void top_k(float* output, int* indices, float* input, int batch_size, int dim) {
    __shared__ float s_topk[RowsPerBlock][TopK * BlockSize];
    __shared__ int s_topk_indices[RowsPerBlock][TopK * BlockSize];

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    float r_topk[TopK];
    float r_topk_indices[TopK];

    for (int i = bid, row = 0; i < batch_size; i += gridDim.x * RowsPerBlock, row += 1) {
        float* input_row = input + i * dim;
        for (int ii = 0; ii < TopK; ++ii) {
            r_topk[ii] = MIN_FLOAT;
            r_topk_indices[ii] = -1;
        }

        for (int j = tid; j < dim; j += BlockSize) { 
            float val = input_row[j];

            int k = -1;
            while (k + 1 < TopK && val > r_topk[k + 1]) k++;
            if (k >= 0) {
                for (int kk = 0; kk < k; ++kk) {
                    r_topk[kk] = r_topk[kk + 1];
                    r_topk_indices[kk] = r_topk_indices[kk + 1];
                }
                r_topk[k] = val;
                r_topk_indices[k] = j;
            }
        }

        for (int kk = 0; kk < TopK; ++kk) {
            s_topk[row][tid * TopK + kk] = r_topk[kk];
            s_topk_indices[row][tid * TopK + kk] = r_topk_indices[kk];
        }
        __syncthreads();

        for (int stride = BlockSize / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                float* s_topk_a = s_topk[row] + tid * TopK;
                float* s_topk_b = s_topk[row] + (tid + stride) * TopK;
                int* s_topk_indices_a = s_topk_indices[row] + tid * TopK;
                int* s_topk_indices_b = s_topk_indices[row] + (tid + stride) * TopK;
                int a = TopK - 1, b = TopK - 1, c = TopK - 1;

                float r_topk_merge[TopK];
                int r_topk_indices_merge[TopK];

                while (a >= 0 && b >= 0 && c >= 0) {
                    if (s_topk_a[a] > s_topk_b[b]) {
                        r_topk_merge[c] = s_topk_a[a];
                        r_topk_indices_merge[c] = s_topk_indices_a[a];
                        a--;
                    } else {
                        r_topk_merge[c] = s_topk_b[b];
                        r_topk_indices_merge[c] = s_topk_indices_b[b];
                        b--;
                    }
                    c--;
                }

                for (int kk = 0; kk < TopK; ++kk) {
                    s_topk_a[kk] = r_topk_merge[kk];
                    s_topk_indices_a[kk] = r_topk_indices_merge[kk];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            for (int kk = 0; kk < TopK; ++kk) {
                output[i * TopK + kk] = s_topk[row][tid * TopK + kk];
                indices[i * TopK + kk] = s_topk_indices[row][tid * TopK + kk];
            }
        }
    }
}