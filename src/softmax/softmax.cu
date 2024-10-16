#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define WARP_SIZE 32
#define THREADS_PER_BLOCK 128
#define FLT_MAX ((float)(1e10))

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

// 经典 attention, Q 的形状为 [batch_size, num_heads, seq_len_q, head_dim]
// K, V 形状为 [batch_size, num_heads, seq_len_kv, head_dim]
// 则 softmax 的输入形状为 [batch_size, num_heads, seq_len_q, seq_len_kv]
__global__ void softmax_kernel(float* input, const int batch_size, const int num_heads, 
        const int seq_len_q, const int seq_len_kv, const float scale, const bool causal=true) {
    int offset = blockIdx.x * seq_len_kv;
    __shared__ float s_max, s_sum;

    // reduce max
    float thread_max = -FLT_MAX;
    for (int i = threadIdx.x; i < seq_len_kv; i += blockDim.x) {
        thread_max = max(thread_max, input[offset + i]);
    }
    float row_max = block_reduce_max<THREADS_PER_BLOCK>(thread_max);
    __syncthreads();
    if (threadIdx.x == 0) s_max = row_max;

    // reduce sum
    float thread_sum = 0.f;
    for (int i = threadIdx.x; i < seq_len_kv; i += blockDim.x) {
        thread_sum += exp(input[offset + i] - s_max);
    }
    float row_sum = block_reduce_sum<THREADS_PER_BLOCK>(thread_sum);
    __syncthreads();
    if (threadIdx.x == 0) s_sum = row_sum;

    // write result to global mem
    for (int i = threadIdx.x; i < seq_len_kv; i += blockDim.x) {
        input[offset + i] = (exp(input[offset + i] - s_max) / s_sum);
    }
}

void softmax_cpu(float* input, int row, int col) {
    for (int r = 0; r < row; r++) {
        float row_max = -FLT_MAX;
        for (int c = 0; c < col; c++) {
            row_max = max(input[r * col + c], row_max);
        }
        float row_sum = 0.f;
        for (int c = 0; c < col; c++) {
            row_sum += expf(input[r * col + c] - row_max);
        }
        for (int c = 0; c < col; c++) {
            input[r * col + c] = expf(input[r * col + c] - row_max) / row_sum;
        }
    }
}

bool check(float *out, float *res, int N){
    int count = 0;
    for(int i=0; i<N; i++){
        if(out[i]!= res[i]) {
            printf("out[%d]=%f, ref[%d]=%f\n", i, out[i], i, res[i]);
            count++;
        }       
    }
    printf("%d not match\n", count);
    return count == 0;
}

int main() {
    const int batch_size = 8;
    const int num_heads = 1;
    const int seq_len_q = 11; // prefill, seq_len_q == seq_len_kv; decoding, seq_len_q == 1
    const int seq_len_kv = 1024;
    const int head_dim = 64;

    const int N = batch_size * num_heads * seq_len_q * seq_len_kv;

    float* host_input = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        host_input[i] = 1.;
    }
    float* host_input_res = (float*)malloc(N * sizeof(float));
    float* device_input;
    cudaMalloc((void**)(&device_input), N * sizeof(float));
    cudaMemcpy(device_input, host_input, N * sizeof(float), cudaMemcpyHostToDevice);

    int block_size;

    dim3 grid_dim(batch_size * num_heads * seq_len_q);
    dim3 block_dim(THREADS_PER_BLOCK); // 一个 block 处理一行

    float scale = sqrt((float)head_dim);
    softmax_kernel<<<grid_dim, block_dim>>>(device_input, batch_size, num_heads, seq_len_q, seq_len_kv, scale);
    cudaMemcpy(host_input_res, device_input, N * sizeof(float), cudaMemcpyDeviceToHost);

    softmax_cpu(host_input, batch_size * num_heads * seq_len_q, seq_len_kv);
    bool right = check(host_input, host_input_res, N);
    if (!right) printf("not right!\n");
    else printf("passed\n");
}