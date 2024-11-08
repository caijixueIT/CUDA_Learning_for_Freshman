#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <cooperative_groups.h>
#include <random>
#include <iostream>

#define WARP_SIZE 32
#define MAX_WARP_NUM 32
#define THREADS_PER_BLOCK 256
#define MAX_REG_SIZE 8
#define minus_infinity -10000.0

unsigned int next_pow2(unsigned int n) {
    return std::pow(2, std::ceil(std::log2(n)));
}

bool check(float *out, float *res, int N, int row, int col) {
    // 只打印第一个 batch 第一个 head 的结果的左上角 8x8 的子矩阵
    for (int i = 0; i < row && i < 8; i++) {
        for (int j = 0; j < col && j < 8; j++) {
            printf("%f ", out[i * col + j]);
        }
        printf("\n");
    }
    printf("\n\n");
    for (int i = 0; i < row && i < 8; i++) {
        for (int j = 0; j < col && j < 8; j++) {
            printf("%f ", res[i * col + j]);
        }
        printf("\n");
    }
    printf("\n\n");
    for(int i=0; i<N; i++){
        if(abs(out[i] - res[i]) > 1e-4) {
            printf("out[%d]=%f, res[%d]=%f", i, out[i], i, res[i]);
            return false;
        }
    }
    return true;
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
//   #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Warp Reduce Max
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max(float val) {
//   #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// Block reduce sum/max/min device helper for Layer/RMS Norm/Softmax etc.
// grid 1D block 1D, grid(N/128), block(128)
template<const int NUM_THREADS=128>
__device__ __forceinline__ float block_reduce_sum(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  
  val = warp_reduce_sum<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  val = warp_reduce_sum<NUM_WARPS>(val);
  return val;
}

template<const int NUM_THREADS=128>
__device__ __forceinline__ float block_reduce_max(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  
  val = warp_reduce_max<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : -FLT_MAX;
  val = warp_reduce_max<NUM_WARPS>(val);
  return val;
}

template <int iterations>
__global__ void attn_softmax_v2(float* vals,
                                float* attn_mask,
                                int total_count,
                                int heads,
                                int sequence_length,
                                int num_seq,
                                int reduceWidth)
{
    float4 data[MAX_REG_SIZE];

    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    int reduce_blocks = reduceWidth >> 5;
    int seq_lane = threadIdx.x % reduceWidth;

    __shared__ float partialSum[MAX_WARP_NUM];

    int iter_offset = blockIdx.x * (warp_num / reduce_blocks) + (wid / reduce_blocks);
    if (iter_offset < total_count) {
        vals += (iter_offset * sequence_length);
        int seq_id = iter_offset % num_seq;
        int mask_offset = seq_id * sequence_length;

        float max_val = minus_infinity;
        for (int i = 0; i < iterations; i++) {
            int data_id = i * (reduceWidth << 2) + (seq_lane);
            
            bool x_check = (data_id < sequence_length);
            bool y_check = ((data_id + reduceWidth) < sequence_length);
            bool z_check = ((data_id + reduceWidth * 2) < sequence_length);
            bool w_check = ((data_id + reduceWidth * 3) < sequence_length);

            float* attn_mask_ptr = attn_mask ? (attn_mask + data_id + mask_offset) : attn_mask;

            if (attn_mask_ptr) {
                data[i].x = x_check ? vals[data_id] + attn_mask_ptr[0]: minus_infinity;
                data[i].y = y_check ? vals[data_id + reduceWidth] + attn_mask_ptr[reduceWidth]
                                    : minus_infinity;
                data[i].z = z_check ? vals[data_id + reduceWidth * 2] + attn_mask_ptr[reduceWidth * 2]
                                    : minus_infinity;
                data[i].w = w_check ? vals[data_id + reduceWidth * 3] + attn_mask_ptr[reduceWidth * 3]
                                    : minus_infinity;
            } else {
                data[i].x = x_check ? vals[data_id] : minus_infinity;
                data[i].y = y_check ? vals[data_id + reduceWidth] : minus_infinity;
                data[i].z = z_check ? vals[data_id + reduceWidth * 2] : minus_infinity;
                data[i].w = w_check ? vals[data_id + reduceWidth * 3] : minus_infinity;
            }

            max_val = (data[i].x > max_val ? data[i].x : max_val);
            max_val = (data[i].y > max_val ? data[i].y : max_val);
            max_val = (data[i].z > max_val ? data[i].z : max_val);
            max_val = (data[i].w > max_val ? data[i].w : max_val);
        }

        max_val = warp_reduce_max<WARP_SIZE>(max_val);
        if (reduceWidth > WARP_SIZE) {
            if (lane == 0) partialSum[wid] = max_val;
            __syncthreads();

            if (lane < warp_num) max_val = partialSum[lane];
            __syncthreads();

            for (int i = 1; i < reduce_blocks; i *= 2) {
                max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, i));
            }

            max_val = __shfl_sync(0xffffffff, max_val, wid);
        }

        float sum = 0;
        for (int i = 0; i < iterations; i++) {
            data[i].x = __expf(data[i].x - max_val);
            data[i].y = __expf(data[i].y - max_val);
            data[i].z = __expf(data[i].z - max_val);
            data[i].w = __expf(data[i].w - max_val);

            sum += (data[i].x + data[i].y + data[i].z + data[i].w);
        }

        sum = warp_reduce_sum<WARP_SIZE>(sum);
        if (reduceWidth > WARP_SIZE) {
            if (lane == 0) partialSum[wid] = sum;
            __syncthreads();

            if (lane < warp_num) sum = partialSum[lane];
            __syncthreads();

            for (int i = 1; i < reduce_blocks; i *= 2) { 
                sum += __shfl_xor_sync(0xffffffff, sum, i); 
            }

            sum = __shfl_sync(0xffffffff, sum, wid);
        }
        // sum += 1e-6;

        for (int i = 0; i < iterations; i++) {
            int data_id = i * (reduceWidth << 2) + (seq_lane);
            if (data_id < sequence_length) {
                vals[data_id] = data[i].x / sum;
                if ((data_id + reduceWidth) < sequence_length)
                    vals[data_id + reduceWidth] = data[i].y / sum;
                if ((data_id + reduceWidth * 2) < sequence_length)
                    vals[data_id + reduceWidth * 2] = data[i].z / sum;
                if ((data_id + reduceWidth * 3) < sequence_length)
                    vals[data_id + reduceWidth * 3] = data[i].w / sum;
            }
        }
    }
}

#define LAUNCH_ATTN_SOFTMAX_V2(iterations)                                      \
    attn_softmax_v2<iterations><<<grid, block, 0, stream>>>(vals,               \
                                                               mask,            \
                                                               total_count,     \
                                                               heads,           \
                                                               sequence_length, \
                                                               num_seq,         \
                                                               reduce_width);

void launch_attn_softmax_v2(float* vals,
                            float* mask,
                            int batch_size,
                            int heads,
                            int num_seq,
                            int sequence_length,
                            cudaStream_t stream)
{
    const int total_count = batch_size * heads * num_seq;
    constexpr int attn_threads = THREADS_PER_BLOCK; // 256
    constexpr int min_reduce_width = WARP_SIZE;     // 32
    constexpr int internal_unroll = 16 / sizeof(float);
    
    const int thread_steps_rounded =
        next_pow2((sequence_length + internal_unroll - 1) / internal_unroll);
    const int thread_steps_schedule =
        (thread_steps_rounded < min_reduce_width) ? min_reduce_width : thread_steps_rounded;

    // 取值范围 32, 64, 128, 256, 表示多少个线程进行规约处理 score 一行
    const int reduce_width = (thread_steps_schedule < attn_threads) ? thread_steps_schedule
                                                                    : attn_threads;
    // 当一行元素足够多, 需要通过多次迭代才能处理完一行
    const int iterations = thread_steps_schedule / reduce_width;
    // 当一行元素足够少, reduce_width 个线程就可以处理一行, 所以一个 block 可以处理多行
    const int partitions = attn_threads / reduce_width;

    // Launch params
    dim3 grid((total_count + partitions - 1) / partitions);
    dim3 block(attn_threads);

    if (iterations == 1) {
        LAUNCH_ATTN_SOFTMAX_V2(1);
    } else if (iterations == 2) {
        LAUNCH_ATTN_SOFTMAX_V2(2);
    } else if (iterations == 4) {
        LAUNCH_ATTN_SOFTMAX_V2(4);
    } else if (iterations == 8) {
        LAUNCH_ATTN_SOFTMAX_V2(8);
    } else {
        throw std::runtime_error("Unsupport Seq_Length!");
    }
}

int main() {
    int batch_size = 2;
    int num_heads = 8;
    int num_seqs = 128;
    int seq_len = 128;

    int N = batch_size * num_heads * num_seqs * seq_len;
    float* h_val = (float*)malloc(N * sizeof(float));
    float* cpu_res = (float*)malloc(N * sizeof(float));
    float* cuda_res = (float*)malloc(N * sizeof(float));
    float* h_mask = (float*)malloc(num_seqs * seq_len * sizeof(float));

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-5.0, 5.0);
    for (int i = 0; i < N; ++i) {
        h_val[i] = static_cast<float>(dis(gen));
    }

    int mask_begin = 1;
    for (int i = 0; i < num_seqs; ++i, ++mask_begin) {
        for (int j = 0; j < mask_begin; ++j) {
            h_mask[i * seq_len + j] = 0.f;
        }
        for (int j = mask_begin; j < seq_len; ++j) {
            h_mask[i * seq_len + j] = minus_infinity;
        }
    }

    float* d_val;
    cudaMalloc((void**)&d_val, N * sizeof(float));
    cudaMemcpy(d_val, h_val, N * sizeof(float), cudaMemcpyHostToDevice);

    float* d_mask;
    cudaMalloc((void**)&d_mask, num_seqs * seq_len * sizeof(float));
    cudaMemcpy(d_mask, h_mask, num_seqs * seq_len * sizeof(float), cudaMemcpyHostToDevice);

    launch_attn_softmax_v2(d_val, d_mask, batch_size, num_heads, num_seqs, seq_len, 0);
    cudaMemcpy(cuda_res, d_val, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < batch_size * num_heads; ++i) {
        float* ptr = h_val + i * num_seqs * seq_len;
        float* res_ptr = cpu_res + i * num_seqs * seq_len;
        for (int j = 0; j < num_seqs; ++j) {
            float row_max = minus_infinity;
            for (int k = 0; k < seq_len; ++k) {
                row_max = max(row_max, ptr[j * seq_len + k] + h_mask[j * seq_len + k]);
            }

            float row_sum = 0.f;
            for (int k = 0; k < seq_len; ++k) {
                row_sum += expf(ptr[j * seq_len + k] + h_mask[j * seq_len + k] - row_max);
            }

            for (int k = 0; k < seq_len; ++k) {
                res_ptr[j * seq_len + k] = expf(ptr[j * seq_len + k] + h_mask[j * seq_len + k] - row_max) / row_sum;
            }
        }
    }

    if (check(cpu_res, cuda_res, N, num_seqs, seq_len)) {
        printf("pass\n");
    } else {
        printf("error\n");
    }    
    return 0;
}
