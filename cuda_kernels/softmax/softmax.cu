#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define WARP_SIZE 32
#define FLT_MAX ((float)(1e10))

template <typename T>
struct Add {
    __device__ __forceinline__ T operator() (const T& x, const T&y) {
        return x + y;
    }
};

template <typename T>
struct Max {
    __device__ __forceinline__ T operator() (const T& x, const T&y) {
        return x > y ? x : y;
    }
};


template <typename T, int REDUCE_SIZE, template<typename> class OP>
__forceinline__ __device__ T warp_reduce(T val) {
    for (int i = REDUCE_SIZE / 2; i >= 1; i >>= 1) {
        val = OP<T>()(val, __shfl_xor_sync(0xffffffff, val, i));
    }
    return val;
}

template <typename T, int THREADS_PER_BLOCK, template<typename> class OP>
__forceinline__ __device__ T block_reduce(T val) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    constexpr int WARP_NUM = THREADS_PER_BLOCK / WARP_SIZE;

    __shared__ T s_data[WARP_NUM];

    val = warp_reduce<T, WARP_SIZE, OP>(val);
    if (lane == 0) {
        s_data[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < WARP_NUM) ? s_data[lane] : T(0);
    val = warp_reduce<T, WARP_NUM, OP>(val);
    
    return val;
}

/*
 *  softmax kernel
 *  one block for one row
*/
template <int threads_per_block=128>
__global__ void softmax_kernel(float* input, int row, int col) {
    input += blockIdx.x * col;
    __shared__ float s_sum, s_max;

    float max = -FLT_MAX;
    for (int i = threadIdx.x; i < col; i += blockDim.x) {
        max = max > input[i] ? max : input[i];
    }
    max = block_reduce<float, threads_per_block, Max>(max);
    if (threadIdx.x == 0) s_max = max;
    __syncthreads();

    float sum = 0.f;
    for (int i = threadIdx.x; i < col; i += blockDim.x) {
        input[i] = expf(input[i] - s_max);
        sum += input[i];
    }
    sum = block_reduce<float, threads_per_block, Add>(sum);
    if (threadIdx.x == 0) s_sum = 1.f / sum;
    __syncthreads();

    for (int i = threadIdx.x; i < col; i += blockDim.x) {
        input[i] *= s_sum;
    }
}

/**
 *  softmax kernel v2
 *  one warp for one row
*/
template <int threads_per_block=128>
__global__ void softmax_kernel_v2(float* input, int row, int col) {
    constexpr int warp_num = threads_per_block / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int cur_row = blockIdx.x * warp_num + warp_id;
    if (cur_row >= row) return;
    input += cur_row * col;
    int lane_id = threadIdx.x % WARP_SIZE;

    float max = -FLT_MAX;
    for (int i = lane_id; i < col; i += WARP_SIZE) {
        max = max > input[i] ? max : input[i];
    }
    max = warp_reduce<float, WARP_SIZE, Max>(max);

    float sum = 0.f;
    for (int i = lane_id; i < col; i += WARP_SIZE) {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }
    sum = warp_reduce<float, WARP_SIZE, Add>(sum);

    for (int i = lane_id; i < col; i += WARP_SIZE) {
        input[i] /= sum;
    }
}

void softmax_cpu(float* input, float* output, int row, int col) {
    for (int i = 0; i < row; ++i) {
        float max = -FLT_MAX;
        for (int j = 0; j < col; ++j) {
            max = max > input[i * col + j] ? max : input[i * col + j];
        }
        float sum = 0.f;
        for (int j = 0; j < col; ++j) {
            sum += expf(input[i * col + j] - max);
        }
        for (int j = 0; j < col; ++j) {
            output[i * col + j] = expf(input[i * col + j] - max) / sum;
        }
    }
}

bool check(float *out, float *res, int N){
    for (int i = 0; i < N; ++i) {
        if (fabs(out[i] - res[i]) > 1e-5) {
            printf("error\n");
            return false;
        }
    }
    printf("success\n");
    return true;
}

void print(float* val, int row, int col) {
    printf("======================\n");
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            printf("%.3f ", val[i * col + j]);
        }
        printf("\n");
    }
}

int main() {
    int row = 127;
    int col = 1024;
    constexpr int threads_per_block = 128;
    constexpr int warp_num = threads_per_block / WARP_SIZE;

    float* input = (float*)malloc(row * col * sizeof(float));
    float* output = (float*)malloc(row * col * sizeof(float));
    float* output_ref = (float*)malloc(row * col * sizeof(float));
    float* d_input;
    cudaMalloc((void**)&d_input, row * col * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < row * col; ++i) {
        input[i] = rand() % 17;
    }
    softmax_cpu(input, output_ref, row, col);
    // one block for one row
    cudaMemcpy(d_input, input, row * col * sizeof(float), cudaMemcpyHostToDevice);
    softmax_kernel<threads_per_block><<<row, threads_per_block>>>(d_input, row, col);
    cudaMemcpy(output, d_input, row * col * sizeof(float), cudaMemcpyDeviceToHost);
    check(output, output_ref, row * col);
    // one warp for one row
    cudaMemcpy(d_input, input, row * col * sizeof(float), cudaMemcpyHostToDevice);
    softmax_kernel_v2<threads_per_block><<<(row + warp_num - 1) / warp_num, threads_per_block>>>(d_input, row, col);
    cudaMemcpy(output, d_input, row * col * sizeof(float), cudaMemcpyDeviceToHost);
    check(output, output_ref, row * col);
    return 0;
}