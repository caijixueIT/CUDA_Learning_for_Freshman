/**
 * @Description: 沿着给定高维张量的某一个维度进行规约，求最大值
 * @LastEditors: Bruce Li
 * @LastEditTime: 2025-07-29 18:08
 */

#include <assert.h>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <chrono>

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32
#define FLT_MIN -3.402823466e+38F

#define CUDA_CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

float* h_tensor;
float* h_tensor_max;
float* h_tensor_max_ref;
float* d_tensor;
float* d_tensor_max;
long long N = 1;
long long M = 1;

bool check(float* cpu_res, float* gpu_res, int N) {
    for (int i = 0; i < N; i++) {
        if (abs(cpu_res[i] - gpu_res[i]) > 1e-10) {
            return false;
        }
    }
    return true;
}

float random_float(float min=-100.f, float max=100.f) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}

void init_tensor(float* tensor, int n) {
    for (int i = 0; i < n; i++) {
        tensor[i] = random_float();
    }
}

void create_tensor(int* input_shape, int input_dims, int reduce_dim) { 
    printf("输入是一个 %d 维张量, 要沿着第 %d 维取最大值, 输入各维度的具体维数是\n", input_dims, reduce_dim);
    for (int i = 0; i < input_dims; i++) {
        printf("第 %d 维: %d\n", i, input_shape[i]);
        N *= input_shape[i];
        if (i != reduce_dim) M *= input_shape[i];
    }
    printf("输出是一个 %d 维张量, 输出各维度的具体维数是\n", input_dims - 1);
    for (int i = 0, j = 0; i < input_dims; i++) {
        if (i == reduce_dim) continue;
        printf("第 %d 维: %d\n", ++j, input_shape[i]);
    }
    printf("则输入张量包含 %lld 个浮点数, 输出张量包含 %lld 个浮点数\n", N, M);

    h_tensor = (float*)malloc(N * sizeof(float));
    h_tensor_max = (float*)malloc(M * sizeof(float));
    h_tensor_max_ref = (float*)malloc(M * sizeof(float));

    cudaMalloc((void**)&d_tensor, N * sizeof(float));
    cudaMalloc((void**)&d_tensor_max, M * sizeof(float));

    init_tensor(h_tensor, N);
    cudaMemcpy(d_tensor, h_tensor, N * sizeof(float), cudaMemcpyHostToDevice);
}

void destroy_tensor() {
    free(h_tensor);
    free(h_tensor_max);
    free(h_tensor_max_ref);
    cudaFree(d_tensor);
    cudaFree(d_tensor_max);
    h_tensor = nullptr;
    h_tensor_max = nullptr;
    h_tensor_max_ref = nullptr;
    d_tensor = nullptr;
    d_tensor_max = nullptr;
}

void cpu_reduce_tensor_max(float* tensor, float* tensor_out_ref, int* input_shape, int input_dims, int reduce_dim) {
    int stride_out = 1;
    for (int i = reduce_dim + 1; i < input_dims; i++) {
        stride_out *= input_shape[i];
    }
    int num = 1;
    for (int i = 0; i < reduce_dim; i++) {
        num *= input_shape[i];
    }

    for (int i = 0; i < num; i++) {
        float* input_ptr = tensor + i * stride_out * input_shape[reduce_dim];
        float* output_ptr = tensor_out_ref + i * stride_out;
        
        for (int j = 0; j < stride_out; j++) {
            float max_val = FLT_MIN;
            for (int k = 0; k < input_shape[reduce_dim]; k++) {
                max_val = max(max_val, input_ptr[k * stride_out + j]);
            }
            output_ptr[j] = max_val;
        }
    }
}

template <int KWARP_SIZE = WARP_SIZE>
__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int i = KWARP_SIZE / 2; i >= 1; i >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, i));
    }
    return val;
}

template <int KBLOCK_SIZE = THREAD_PER_BLOCK>
__device__ __forceinline__ float block_reduce(float val) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    constexpr int NUM_WARPS = KBLOCK_SIZE / WARP_SIZE;
    static __shared__ float smem[NUM_WARPS];
    
    val = warp_reduce<WARP_SIZE>(val);
    if (lane == 0) {
        smem[wid] = val;
    }
    __syncthreads();
    val = (lane < NUM_WARPS) ? smem[lane] : 0;
    val = warp_reduce<NUM_WARPS>(val);
    return val;
}

template <int NUM_THREADS_PER_BLOCK=THREAD_PER_BLOCK>
__global__ void kernel_0(float* d_tensor, float* d_tensor_max, int reduce_dim_size) {
    int idx = blockIdx.x;
    int offset = blockIdx.y;
    int tid = threadIdx.x;

    float max_val = FLT_MIN;
    for (int i = tid; i < reduce_dim_size; i += blockDim.x) {
        int index = idx * reduce_dim_size * gridDim.y + i * gridDim.y + offset;
        max_val = fmaxf(max_val, d_tensor[index]);
    }
    max_val = block_reduce<NUM_THREADS_PER_BLOCK>(max_val);
    if (tid == 0) {
        d_tensor_max[idx * gridDim.y + offset] = max_val;
    }
}

void launch_kernel_0(float* d_tensor, float* d_tensor_max, float* h_tensor_max, int* input_shape, int input_dims, int reduce_dim) {
    int stride_out = 1;
    for (int i = reduce_dim + 1; i < input_dims; i++) {
        stride_out *= input_shape[i];
    }
    int num = 1;
    for (int i = 0; i < reduce_dim; i++) {
        num *= input_shape[i];
    }
    dim3 grid_dim(num, stride_out);
    dim3 block_dim(THREAD_PER_BLOCK);
    kernel_0<<<grid_dim, block_dim>>>(d_tensor, d_tensor_max, input_shape[reduce_dim]);
    CUDA_CHECK(cudaMemcpy(h_tensor_max, d_tensor_max, num * stride_out * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
}

void print_result(float* tensor, float* tensor_max, int* input_shape, int input_dims, int reduce_dim) {
    if (input_dims == 3 && reduce_dim == 1) {
        for (int i = 0; i < input_shape[0]; i++) {
            for (int j = 0; j < input_shape[1]; j++) {
                for (int k = 0; k < input_shape[2]; k++) {
                    int index = i * input_shape[1] * input_shape[2] + j * input_shape[2] + k;
                    printf("%f ", tensor[index]);
                }
                printf("\n");
            }
            printf("\n");
        }

        for (int i = 0; i < input_shape[0]; i++) {
            for (int j = 0; j < input_shape[2]; j++) {
                int index = i * input_shape[2] + j;
                printf("%f ", tensor_max[index]);
            }
            printf("\n");
        }
    }
    printf("\n");
}

int main(int argc, char** argv) {
    int input_dims;
    int reduce_dim;
    int* input_shape = nullptr;
    if (argc > 1) {
        input_dims = atoi(argv[1]);
        assert(argc == input_dims + 3);
        input_shape = new int[input_dims];
        for (int i = 0; i < input_dims; i++) {
            int dim = atoi(argv[i + 2]);
            assert(dim > 0);
            input_shape[i] = dim;
        }
        reduce_dim = atoi(argv[argc - 1]);
        assert(reduce_dim >= 0 && reduce_dim < input_dims);
    } else {
        input_dims = 3;
        int a[] = {2, 3, 4};
        input_shape = a;
        reduce_dim = 1;
    }

    create_tensor(input_shape, input_dims, reduce_dim);

    cpu_reduce_tensor_max(h_tensor, h_tensor_max_ref, input_shape, input_dims, reduce_dim);
    launch_kernel_0(d_tensor, d_tensor_max, h_tensor_max, input_shape, input_dims, reduce_dim);
    if (check(h_tensor_max_ref, h_tensor_max, M)) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }

    print_result(h_tensor, h_tensor_max, input_shape, input_dims, reduce_dim);
    print_result(h_tensor, h_tensor_max_ref, input_shape, input_dims, reduce_dim);
    destroy_tensor();

    return 0;
}

// shell 编译并执行
// nvcc -arch=sm_89 max_along_some_dim.cu && ./a.out 7 12 34 5678 3 2 5 3 2