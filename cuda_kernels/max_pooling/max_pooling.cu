#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <assert.h>
#include <iostream>

#define FLT_MAX 3.402823466e+38F
#define WARMING_TIMES 50
#define PROFILING_TIMES 100

#define CUDA_CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

float* h_input;
float* h_output;
float* h_output_ref;
float* d_input;
float* d_output;

float random_float(float min=-100.f, float max=100.f) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}

void init_tensor(float* tensor, int n) {
    for (int i = 0; i < n; i++) {
        tensor[i] = i;//random_float();
    }
}

void create_tensor(int batch_size, int channels, int height, int width, int out_height, int out_width) { 
    int N = batch_size * channels * height * width;
    int M = batch_size * channels * out_height * out_width;
    h_input = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(M * sizeof(float));
    h_output_ref = (float*)malloc(M * sizeof(float));
    init_tensor(h_input, N);
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, M * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
}

void destroy_tensor() {
    free(h_input);
    free(h_output);
    free(h_output_ref);
    cudaFree(d_input);
    cudaFree(d_output);
    h_input = nullptr;
    h_output = nullptr;
    h_output_ref = nullptr;
    d_input = nullptr;
    d_output = nullptr;
}

bool check(float* cpu_res, float* gpu_res, int N) {
    for (int i = 0; i < N; i++) {
        if (abs(cpu_res[i] - gpu_res[i]) > 1e-10) {
            return false;
        }
    }
    return true;
}

void cpu_max_pool2d(float* input, float* output, int batch, int channel, int height, int width, 
    int pool_height, int pool_width, int height_stride, int width_stride, int out_height, int out_width) { 
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channel; c++) {
            for (int h = 0; h < out_height; h++) {
                for (int w = 0; w < out_width; w++) {
                    int input_h_start = h * height_stride;
                    int input_w_start = w * width_stride;
                    int input_h_end = min(input_h_start + pool_height, height);
                    int input_w_end = min(input_w_start + pool_width, width);
                    float max_val = -FLT_MAX;
                    for (int input_h = input_h_start; input_h < input_h_end; input_h++) {
                        for (int input_w = input_w_start; input_w < input_w_end; input_w++) {
                            int input_index = (b * channel + c) * height * width + input_h * width + input_w;
                            max_val = max(max_val, input[input_index]);
                        }
                    }
                    output[(b * channel + c) * out_height * out_width + h * out_width + w] = max_val;
                }
            }
        }
    }
}


template <typename Kernel, typename... Args, int WARMUP=50, int PROFILE=100>
void kernel_profiling(Kernel kernel, dim3 grid, dim3 block, std::string fun_name, float* gpu_res, float* cpu_res, int n, Args&&... args) {
    std::cout << std::endl << std::string(100, '=') << std::endl;
    // check
    if (check(cpu_res, gpu_res, n)) {
        printf("%s PASS\n", fun_name.c_str());
    } else {
        printf("%s FAIL\n", fun_name.c_str());
        return;
    }
    // warming up
    for (int i = 0; i < WARMUP; i++) {
        kernel<<<grid, block>>>(std::forward<Args>(args)...);
    }
    // profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventSynchronize(start);
    for (int i = 0; i < PROFILE; i++) {
        kernel<<<grid, block>>>(std::forward<Args>(args)...);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("%s Average elasped time = %f ms\n", fun_name.c_str(), elapsed_time / PROFILING_TIMES);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

__global__ void max_pool2d_kernel_0(
    const float* input, 
    float* output, 
    int batch_size, 
    int channels, 
    int height, 
    int width, 
    int pool_height, 
    int pool_width, 
    int height_stride, 
    int width_stride,
    int out_height,
    int out_width
) { 
    int batch_idx = blockIdx.z / channels;
    int channel_idx = blockIdx.z % channels;
    int height_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int width_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx >= batch_size || channel_idx >= channels || height_idx >= out_height || width_idx >= out_width) return;

    int in_height_start_idx = height_idx * height_stride;
    int in_width_start_idx = width_idx * width_stride;
    int in_height_end_idx = min(in_height_start_idx + pool_height - 1, height - 1);
    int in_width_end_idx = min(in_width_start_idx + pool_width - 1, width - 1);
    float max_val = -FLT_MAX;
    for (int in_height_idx = in_height_start_idx; in_height_idx <= in_height_end_idx; in_height_idx++) {
        for (int in_width_idx = in_width_start_idx; in_width_idx <= in_width_end_idx; in_width_idx++) {
            int in_idx = batch_idx * channels * height * width + 
                         channel_idx * height * width +  
                         in_height_idx * width + 
                         in_width_idx;
            max_val = max(max_val, input[in_idx]);
        }
    }
    output[batch_idx * channels * out_height * out_width + 
        channel_idx * out_height * out_width + height_idx * out_width + width_idx] = max_val;
}

void launch_kernel_0(
    const float* d_input, 
    float* d_output, 
    float* h_output,
    float* h_output_ref,
    int batch_size, 
    int channels, 
    int height, 
    int width, 
    int pool_height, 
    int pool_width, 
    int height_stride, 
    int width_stride,
    int out_height,
    int out_width
) {
    dim3 block(16, 16);  // 每个线程块有16x16个线程
    dim3 grid((out_height + block.x - 1) / block.x,
              (out_width + block.y - 1) / block.y,
              batch_size * channels);
    
    max_pool2d_kernel_0<<<grid, block>>>(
        d_input, d_output, 
        batch_size, channels, height, width,
        pool_height, pool_width, 
        height_stride, width_stride,
        out_height, out_width);
    
    int out_size = out_height * out_width * batch_size * channels;  
    cudaMemcpy(h_output, d_output, sizeof(float) * out_size, cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel_profiling(max_pool2d_kernel_0, grid, block, std::string(__func__), h_output, h_output_ref, out_size, 
        d_input, d_output, batch_size, channels, height, width, pool_height, pool_width, height_stride, width_stride, out_height, out_width);
}

__global__ void max_pool2d_kernel_1(
    const float* input, 
    float* output, 
    int batch_size, 
    int channels, 
    int height, 
    int width, 
    int pool_height, 
    int pool_width, 
    int height_stride, 
    int width_stride,
    int out_height,
    int out_width
) { 
    int batch_idx = blockIdx.z / channels;
    int channel_idx = blockIdx.z % channels;
    int height_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int width_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx >= batch_size || channel_idx >= channels /* || height_idx >= out_height || width_idx >= out_width */) return;

    extern __shared__ float shared_mem[];
    const int shared_height = height_stride * blockDim.x + pool_height - 1;
    const int shared_width = width_stride * blockDim.y + pool_width - 1;
    const int block_start_h = blockIdx.x * blockDim.x * height_stride;
    const int block_start_w = blockIdx.y * blockDim.y * width_stride;
    // const int block_end_h = min(block_start_h + shared_height, height);
    // const int block_end_w = min(block_start_w + shared_width, width);

    for (int i = threadIdx.x; i < shared_height; i += blockDim.x) {
        for (int j = threadIdx.y; j < shared_width; j += blockDim.y) {
            const int h = block_start_h + i;
            const int w = block_start_w + j;
            if (h < height && w < width) {
                shared_mem[i * shared_width + j] = input[((batch_idx * channels + channel_idx) * height + h) * width + w];
            } else {
                shared_mem[i * shared_width + j] = -FLT_MAX;
            }
        }
    }
    
    __syncthreads();
    // if (threadIdx.x == 0 && threadIdx.y == 0) {
    //     for (int i = 0; i < shared_height; i++) {
    //         for (int j = 0; j < shared_width; j++) {
    //             printf("%.1f ", shared_mem[i * shared_width + j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n\n");
    // }

    float max_val = -FLT_MAX;
    
    for (int h = 0; h < pool_height; ++h) {
        const int shared_height_idx = h + threadIdx.x * height_stride;
        if (shared_height_idx >= shared_height) continue;
        for (int w = 0; w < pool_width; ++w) {
            const int shared_width_idx = w + threadIdx.y * width_stride;
            if (shared_width_idx >= shared_width) continue;
            max_val = max(max_val, shared_mem[shared_height_idx * shared_width + shared_width_idx]);
        }
    }
    output[((batch_idx * channels + channel_idx ) * out_height + height_idx) * out_width + width_idx] = max_val;
}

void launch_kernel_1(
    const float* d_input, 
    float* d_output, 
    float* h_output,
    float* h_output_ref,
    int batch_size, 
    int channels, 
    int height, 
    int width, 
    int pool_height, 
    int pool_width, 
    int height_stride, 
    int width_stride,
    int out_height,
    int out_width
) {
    dim3 block(16, 16);  // 每个线程块有16x16个线程
    dim3 grid((out_height + block.x - 1) / block.x,
              (out_width + block.y - 1) / block.y,
              batch_size * channels);

    int shmem_size = (block.x * height_stride + pool_height - 1) * (block.y * width_stride + pool_width - 1);
    
    max_pool2d_kernel_1<<<grid, block, shmem_size * sizeof(float)>>>(
        d_input, d_output, 
        batch_size, channels, height, width,
        pool_height, pool_width, 
        height_stride, width_stride,
        out_height, out_width);
    
    int out_size = out_height * out_width * batch_size * channels;  
    cudaMemcpy(h_output, d_output, sizeof(float) * out_size, cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel_profiling(max_pool2d_kernel_1, grid, block, std::string(__func__), h_output, h_output_ref, out_size, 
        d_input, d_output, batch_size, channels, height, width, pool_height, pool_width, height_stride, width_stride, out_height, out_width);
}

void print_tensor(float* tensor, int height, int width) {
    std::cout << std::endl << std::string(100, '=') << std::endl;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f ", tensor[i * width + j]);
        }
        printf("\n");
    }
}

// 测试函数
int main(int argc, char** argv) {
    int batch_size = 1;
    int channels = 1;
    int height = 9;
    int width = 9;
    int pool_height = 2;
    int pool_width = 2;
    int height_stride = 1;
    int width_stride = 1;
    if (argc == 9) {
        batch_size = atoi(argv[1]);
        channels = atoi(argv[2]);
        height = atoi(argv[3]);
        width = atoi(argv[4]);
        pool_height = atoi(argv[5]);
        pool_width = atoi(argv[6]);
        height_stride = atoi(argv[7]);
        width_stride = atoi(argv[8]);
        assert(batch_size > 0 && channels > 0 && height > 0 && width > 0);
        assert(pool_height > 0 && pool_width > 0);
        assert(height_stride > 0 && width_stride > 0);
        assert(height >= pool_height && width >= pool_width);
    }
    int out_height = (height - pool_height) / height_stride + 1;
    int out_width = (width - pool_width) / width_stride + 1;

    create_tensor(batch_size, channels, height, width, out_height, out_width);
    cpu_max_pool2d(h_input, h_output_ref, batch_size, channels, height, width, pool_height, pool_width, 
        height_stride, width_stride, out_height, out_width);
    launch_kernel_0(d_input, d_output, h_output, h_output_ref, batch_size, channels, height, width, pool_height, pool_width, 
        height_stride, width_stride, out_height, out_width);
    launch_kernel_1(d_input, d_output, h_output, h_output_ref, batch_size, channels, height, width, pool_height, pool_width, 
        height_stride, width_stride, out_height, out_width);
    
    if (batch_size == 1 && channels == 1) {
        print_tensor(h_input, height, width);
        print_tensor(h_output, out_height, out_width);
        print_tensor(h_output_ref, out_height, out_width);
    }
    

    return 0;
}