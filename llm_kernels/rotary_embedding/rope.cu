#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <random>
#include "cooperative_groups.h"

namespace cg = cooperative_groups;
#define WARP_SIZE 32

#define THREADS_PER_BLOCK 256

// input 的形状为 [batch_size, seq_len, num_heads * head_dim]
// 每个 block 负责对 n 个 head 进行旋转位置编码，每个 block 256 个线程
// 则负责每个 head 的线程个数为 256 / n
template<typename T, int RotaryDim = 64, int HEADS_PER_BLOCK = 8>
__global__ void apply_rotary_embedding(T* input, int batch_size, int seq_len, int num_heads, int head_dim) {
    constexpr int THREADS_PER_HEAD = THREADS_PER_BLOCK / HEADS_PER_BLOCK;
    int head_idx = blockIdx.x * HEADS_PER_BLOCK + threadIdx.x / THREADS_PER_HEAD;

    constexpr int VALUES_PER_THREAD = 2 * RotaryDim / THREADS_PER_HEAD;

    T vals[VALUES_PER_THREAD];
    input += head_idx * head_dim + (threadIdx.x % THREADS_PER_HEAD) * VALUES_PER_THREAD;
    int seq_idx = head_idx % num_heads;
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
        vals[i] = input[i];
    }

    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
        
    }


}

template<int group_size = 16>
__global__ void test(float* input, int n) {
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<group_size> head_group = cg::tiled_partition<group_size>(tb);

    int tid = threadIdx.x;
    float val = input[tid];

    int group_idx = tid % group_size;
    const int target_lane = (group_idx < group_size / 2)
                                            ? head_group.thread_rank() + group_size / 2
                                            : head_group.thread_rank() - group_size / 2;

    const float rot = head_group.shfl(val, target_lane);
    printf("tid = %d, %f, %f\n", tid, val, rot);
    input[tid] = rot;

}

int main() {
    int n = 32;
    constexpr int group_size = 16;
    float* input = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i) input[i] = i + 1;
    float* d_input;
    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    test<group_size><<<1, n>>>(d_input, n);
    cudaMemcpy(input, d_input, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        std::cout << input[i] << " ";
        if (i % group_size == (group_size - 1)) std::cout << std::endl;
    }

    return 0;

}