#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define WARP_SIZE 32
namespace cuda_reduce {
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

}; // end of namespace cuda_reduce

// 一个 block 处理一行 grid(num_tokens), block(THREADS_PER_BLOCK)
template<typename T, int THREADS_PER_BLOCK>
__global__ void rms_norm_v1(T* input, T* output, T* gamma, int num_tokens, int hidden_size, float eps) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    __shared__ float s_var;
    float var = 0.0f;
    for (int i = tid; i < hidden_size; i += THREADS_PER_BLOCK) {
        float x = (float)input[bid * hidden_size + i];
        var += x * x;
    }

    var = cuda_reduce::block_reduce<float, THREADS_PER_BLOCK, cuda_reduce::Add>(var);
    if (tid == 0) {
        s_var = var + eps;
        s_var = sqrtf(s_var);
        s_var = 1.0f / s_var;
    }
    __syncthreads();

    for (int i = tid; i < hidden_size; i += THREADS_PER_BLOCK) {
        float x = (float)input[bid * hidden_size + i];
        float y = (float)gamma[i] * x * s_var;
        output[bid * hidden_size + i] = (T)y;
    }
}

template <typename T, int VEC_SIZE>
struct vec_pack {
    T data[VEC_SIZE];
    __forceinline__ __device__ float get_square_sum() {
        float sum = 0.0f;
        for (int i = 0; i < VEC_SIZE; ++i) {
            float v = (float)data[i];
            sum += v * v;
        }
        return sum;
    }

    __forceinline__ __device__ auto do_norm(const vec_pack<T, VEC_SIZE>& gamma, 
                                            const float& var) {
        vec_pack<T, VEC_SIZE> norm;
        for (int i = 0; i < VEC_SIZE; ++i) {
            norm.data[i] = (T)(((float)gamma.data[i]) * ((float)data[i]) * var);
        }
        return norm;
    }

    __forceinline__ __device__ auto do_norm_in_place(const vec_pack<T, VEC_SIZE>& gamma, 
                                            const float& var) {
        for (int i = 0; i < VEC_SIZE; ++i) {
            data[i] = (T)(((float)gamma.data[i]) * ((float)data[i]) * var);
        }
        return *this;
    }
};



// 一个 block 处理一行, 采用向量化访存 grid(num_tokens), block(THREADS_PER_BLOCK)
template<typename T, int THREADS_PER_BLOCK>
__global__ void rms_norm_v2(T* input, T* output, T* gamma, int num_tokens, int hidden_size, float eps) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    constexpr int VEC_SIZE = 16 / sizeof(T);

    __shared__ float s_var;
    using vec_pack_t = vec_pack<T, VEC_SIZE>;
    float var = 0.0f;
    for (int i = tid * VEC_SIZE; i < hidden_size; i += THREADS_PER_BLOCK * VEC_SIZE) {
        auto& vec = *(vec_pack_t*)(input + bid * hidden_size + i);
        var += vec.get_square_sum();
    }

    var = cuda_reduce::block_reduce<float, THREADS_PER_BLOCK, cuda_reduce::Add>(var);
    if (tid == 0) {
        s_var = var + eps;
        s_var = sqrtf(s_var);
        s_var = 1.0f / s_var;
    }
    __syncthreads();

    for (int i = tid * VEC_SIZE; i < hidden_size; i += THREADS_PER_BLOCK * VEC_SIZE) {
        auto& vec = *(vec_pack_t*)(input + bid * hidden_size + i);
        const auto& gamma_vec = *(vec_pack_t*)(gamma + i);
        const auto& norm = vec.do_norm(gamma_vec, s_var);
        *(vec_pack<T, VEC_SIZE>*)(output + bid * hidden_size + i) = norm;
    }
}

// 一个 block 处理一行, 采用向量化访存, 采用寄存器存储读取的结果，减少重复从 global_memory读取输入, grid(num_tokens), block(THREADS_PER_BLOCK)
template<typename T, int THREADS_PER_BLOCK, int HIDDEN_SIZE>
__global__ void rms_norm_v3(T* input, T* output, T* gamma, int num_tokens, int hidden_size, float eps) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    constexpr int VEC_SIZE = 16 / sizeof(T);
    constexpr int REG_SIZE = HIDDEN_SIZE / VEC_SIZE / THREADS_PER_BLOCK;
    using VEC = vec_pack<T, VEC_SIZE>;
    VEC reg[REG_SIZE];

    int reg_idx = 0;
    for (int i = tid * VEC_SIZE; i < hidden_size; i += THREADS_PER_BLOCK * VEC_SIZE, ++reg_idx) {
        reg[reg_idx] = *(vec_pack<T, VEC_SIZE>*)(input + bid * hidden_size + i);
    }
    
    __shared__ float s_var;
    float var = 0.0f;
    for (reg_idx = 0; reg_idx < REG_SIZE; reg_idx++) {
        var += reg[reg_idx].get_square_sum();
    }

    var = cuda_reduce::block_reduce<float, THREADS_PER_BLOCK, cuda_reduce::Add>(var);
    if (tid == 0) {
        s_var = var + eps;
        s_var = sqrtf(s_var);
        s_var = 1.0f / s_var;
    }
    __syncthreads();

    reg_idx = 0;
    for (int i = tid * VEC_SIZE; i < hidden_size; i += THREADS_PER_BLOCK * VEC_SIZE, reg_idx++) {
        auto& vec = *(vec_pack<T, VEC_SIZE>*)(input + bid * hidden_size + i);
        const auto& gamma_vec = *(vec_pack<T, VEC_SIZE>*)(gamma + i);
        vec.do_norm_in_place(gamma_vec, s_var);
        *(vec_pack<T, VEC_SIZE>*)(output + bid * hidden_size + i) = vec;
    }
}

template<typename T>
void rms_norm_cpu(T* input, T* output, T* gamma, int num_tokens, int hidden_size, float eps) {
    for (int i = 0; i < num_tokens; ++i) {
        float var;
        for (int j = 0; j < hidden_size; ++j) {
            var += ((float)input[i * hidden_size + j]) * ((float)input[i * hidden_size + j]);
        }
        var = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < hidden_size; ++j) {
            float res = ((float)input[i * hidden_size + j]) * var * (float)gamma[j];
            output[i * hidden_size + j] = (T)res;
        }
    }
}

template<typename T>
bool check(T *out, T *res, int N){
    for (int i = 0; i < N; ++i) {
        // printf("%d, %f, %f\n", i, (float)out[i], (float)res[i]);
        if (fabs((float)out[i] - (float)res[i]) > 1e-1) {
            printf("%d, %f, %f\n", i, (float)out[i], (float)res[i]);
            return false;
        }
    }
    return true;
}


int main() {
    constexpr int num_tokens = 1024;
    constexpr int hidden_size = 128;
    constexpr int THREADS_PER_BLOCK = 32;
    using dtype = float;

    constexpr float eps = 1e-5f;

    dtype* input = (dtype*)malloc(sizeof(dtype) * num_tokens * hidden_size);
    dtype* output = (dtype*)malloc(sizeof(dtype) * num_tokens * hidden_size);
    dtype* gamma = (dtype*)malloc(sizeof(dtype) * hidden_size);

    std::random_device rd;
    std::mt19937 gen(rd());

    // 定义浮点数分布范围（例如 0.0 到 1.0）
    std::uniform_real_distribution<> dis(-1.f, 1.f);

    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        input[i] = dis(gen);
    }
    for (int i = 0; i < hidden_size; ++i) {
        gamma[i] = dis(gen);
    }

    dtype *d_input, *d_output, *d_gamma;
    cudaMalloc((void**)&d_input, sizeof(dtype) * num_tokens * hidden_size);
    cudaMalloc((void**)&d_output, sizeof(dtype) * num_tokens * hidden_size);
    cudaMalloc((void**)&d_gamma, sizeof(dtype) * hidden_size);

    cudaMemcpy(d_input, input, sizeof(dtype) * num_tokens * hidden_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, sizeof(dtype) * hidden_size, cudaMemcpyHostToDevice);
    
    // rms_norm_v1<dtype, THREADS_PER_BLOCK><<<num_tokens, THREADS_PER_BLOCK>>>(d_input, d_output, d_gamma, num_tokens, hidden_size, eps);
    // rms_norm_v2<dtype, THREADS_PER_BLOCK><<<num_tokens, THREADS_PER_BLOCK>>>(d_input, d_output, d_gamma, num_tokens, hidden_size, eps);
    rms_norm_v3<dtype, THREADS_PER_BLOCK, hidden_size><<<num_tokens, THREADS_PER_BLOCK>>>(d_input, d_output, d_gamma, num_tokens, hidden_size, eps);
    cudaMemcpy(output, d_output, sizeof(dtype) * num_tokens * hidden_size, cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // check by cpu
    dtype* output_ref = (dtype*)malloc(sizeof(dtype) * num_tokens * hidden_size);
    rms_norm_cpu<dtype>(input, output_ref, gamma, num_tokens, hidden_size, eps);
    if (check(output, output_ref, num_tokens * hidden_size)) {
        printf("pass\n");
    } else {
        printf("fail\n");
    }

    free(input);
    free(gamma);
    free(output);
    free(output_ref);
    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_output);

    return 0;
}