#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Kernel to compute max along the channel dimension (C) using shared memory
__global__ void maxAlongChannelsSharedMemory(float* input, float* output, int N, int C, int H, int W) {
    extern __shared__ float sharedMemory[];  // Dynamic shared memory

    int n = blockIdx.x;  // Batch index
    int h = blockIdx.y;  // Height index
    int w = blockIdx.z;  // Width index
    int tid = threadIdx.x;  // Thread index within the block

    // Initialize shared memory
    sharedMemory[tid] = input[n * C * H * W + tid * H * W + h * W + w];
    __syncthreads();  // Synchronize threads to ensure shared memory is populated

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (sharedMemory[tid + stride] > sharedMemory[tid]) {
                sharedMemory[tid] = sharedMemory[tid + stride];
            }
        }
        __syncthreads();  // Synchronize threads after each reduction step
    }

    // Write the result to output
    if (tid == 0) {
        output[n * H * W + h * W + w] = sharedMemory[0];
    }
}

int main() {
    // Tensor dimensions
    int N = 2;  // Batch size
    int C = 32; // Channels (must be a power of 2 for simplicity)
    int H = 2;  // Height
    int W = 2;  // Width

    // Host data
    float* h_input = new float[N * C * H * W];
    float* h_output = new float[N * H * W];

    // Initialize input data
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    h_input[n * C * H * W + c * H * W + h * W + w] = static_cast<float>(c + 1);  // Fill with channel index
                }
            }
        }
    }

    // Device data
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input, N * C * H * W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, N * H * W * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blocks(N, H, W);  // Each block handles one (N, H, W) position
    int threadsPerBlock = C;  // One thread per channel
    size_t sharedMemorySize = threadsPerBlock * sizeof(float);  // Shared memory size
    maxAlongChannelsSharedMemory<<<blocks, threadsPerBlock, sharedMemorySize>>>(d_input, d_output, N, C, H, W);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * H * W * sizeof(float), cudaMemcpyDeviceToHost));

    // Print result
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                std::cout << "Max value at (n=" << n << ", h=" << h << ", w=" << w << "): "
                          << h_output[n * H * W + h * W + w] << std::endl;
            }
        }
    }

    // Free memory
    delete[] h_input;
    delete[] h_output;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}