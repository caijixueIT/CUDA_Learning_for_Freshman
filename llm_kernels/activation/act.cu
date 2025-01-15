#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <random>

#define WARP_SIZE 32

// 定义常见的激活函数
namespace activations {
// ReLU
template <typename T>
struct relu {
    __device__ __host__ T operator() (const T& x) {
        return ((x > 0) ? x : 0);
    };
};

// GELU (近似实现)
template <typename T>
struct gelu {
    static constexpr T sqrt_2_over_pi = 0.7978845608028654; // sqrt(2 / pi)
    static constexpr T alpha = 0.044715;

    __device__ __host__ T operator() (const T& x) {
        T x_cubed = x * x * x;
        T tanh_value = tanh(sqrt_2_over_pi * (x + alpha * x_cubed));
        return 0.5 * x * (1 + tanh_value);
    };
};

// SiLU
template<typename T>
struct silu {
    __device__ __host__ T operator() (const T& x) {
        return x / (1 + exp(-x));
    }
};
}; // end of namespace activations

template<typename T, template<typename> class ACT>
__global__ void activation_kernel_v1(T* input, T* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        output[i] = ACT<T>()(input[i]);
    }
}

template<typename T, template<typename> class ACT>
__global__ void activation_kernel_v2(T* input, T* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int pack_size = 16 / sizeof(T);

    for (int i = tid * pack_size; i < n; i += blockDim.x * gridDim.x * pack_size) {
        float4 input_val = *(float4*)(input + i);
        T* pack = (T*)&(input_val);
        for (int j = 0; j < pack_size; ++j) {
            pack[j] = ACT<T>()(pack[j]);
        }
        *(float4*)(output + i) = input_val;
    }
}

template<typename T, template<typename> class ACT>
void activation_cpu(T* input, T* output, int n) {
    for (int i = 0; i < n; ++i) {
        output[i] = ACT<T>()(input[i]);
    }
}

template<typename T>
bool check(T* res, T* ref, int n) {
    for (int i = 0; i < n; ++i) {
        if (i < 10) printf("%d, %f, %f\n", i, res[i], ref[i]);
        if (fabs((float)res[i] - (float)ref[i]) > 1e-2) return false;
    }
    return true;
}

int main() {
    int num_tokens = 1024;
    int hidden_size = 1024;
    int n = num_tokens * hidden_size;
    constexpr int THREADS_PER_BLOCK = 128;

    using dtype = float;
    dtype* input = (dtype*)malloc(n * sizeof(dtype));
    dtype* output = (dtype*)malloc(n * sizeof(dtype));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-100.f, 100.f);
    for (int i = 0; i < n; ++i) {
        input[i] = dis(gen);
        if (i < 10) printf("input %d, %f\n", i, input[i]);
    }

    dtype *d_input, *d_output;
    cudaMalloc((void**)&d_input, n * sizeof(dtype));
    cudaMalloc((void**)&d_output, n * sizeof(dtype));

    cudaMemcpy(d_input, input, n * sizeof(dtype), cudaMemcpyHostToDevice);
    activation_kernel_v2<dtype, activations::relu><<<128, THREADS_PER_BLOCK>>>(d_input, d_output, n);
    cudaMemcpy(output, d_output, n * sizeof(dtype), cudaMemcpyDeviceToHost);

    dtype* output_ref = (dtype*)malloc(n * sizeof(dtype));
    activation_cpu<dtype, activations::relu>(input, output_ref, n);

    if (check(output, output_ref, n)) std::cout << "passed" << std::endl;
    else std::cout << "failed" << std::endl;

    return 0;
}