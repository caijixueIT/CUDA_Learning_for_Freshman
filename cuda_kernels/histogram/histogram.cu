#include <stdio.h>
#include <cuda_runtime.h>
#include <random>
#include <chrono>

#define BINS 256
#define THREADS_PER_BLOCK 256

unsigned char random_unsigned_char() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);
    return static_cast<unsigned char>(dis(gen));
}

__global__ void histogram_kernel(unsigned char* image, unsigned int* histo, 
                                int width, int height) {
    // 共享内存中的局部直方图
    __shared__ unsigned int temp_histo[BINS];

    int tid = threadIdx.x * blockDim.y + threadIdx.y;
    // 初始化共享内存
    for (int i = tid; i < BINS; i += blockDim.x * blockDim.y) {
        temp_histo[i] = 0;
    }
    __syncthreads();
    
    // 计算线程的全局索引
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int pixel_value = image[y * width + x];
        atomicAdd(&temp_histo[pixel_value], 1);
    }
    __syncthreads();
    
    // 将共享内存中的结果合并到全局内存
    for (int i = tid; i < BINS; i += blockDim.x * blockDim.y) {
        atomicAdd(&histo[i], temp_histo[i]);
    }
}

void compute_histogram(unsigned char* d_image, unsigned int* d_histo, 
                      int width, int height) {
    // 设置网格和块维度
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y);
    
    // 调用核函数
    histogram_kernel<<<grid, block>>>(d_image, d_histo, width, height);
}

bool check(unsigned int* h_histo, unsigned int* h_histo_ref, int n) {
    for (int i = 0; i < n; i++) {
        if (h_histo[i] != h_histo_ref[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    const int width = 1024;
    const int height = 768;
    const int image_size = width * height;
    
    // 分配主机内存
    unsigned char* h_image = new unsigned char[image_size];
    unsigned int h_histo[BINS] = {0};
    unsigned int h_histo_ref[BINS] = {0};
    
    // 初始化图像数据 (这里简单示例，实际应从文件读取)
    for (int i = 0; i < image_size; i++) {
        h_image[i] = random_unsigned_char();
        h_histo_ref[h_image[i]]++;
    }
    
    // 分配设备内存
    unsigned char* d_image;
    unsigned int* d_histo;
    cudaMalloc((void**)&d_image, image_size * sizeof(unsigned char));
    cudaMalloc((void**)&d_histo, BINS * sizeof(unsigned int));
    
    // 复制数据到设备
    cudaMemcpy(d_image, h_image, image_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_histo, 0, BINS * sizeof(unsigned int));
    
    // 计算直方图
    compute_histogram(d_image, d_histo, width, height);
    
    // 将结果复制回主机
    cudaMemcpy(h_histo, d_histo, BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (check(h_histo, h_histo_ref, BINS)) {
        printf("Passed\n");
    } else {    
        printf("Failed\n");
        // 打印结果 (示例)
        for (int i = 0; i < BINS; i++) {
            printf("%d: [%u][%u]\n", i, h_histo[i], h_histo_ref[i]);
        }
    }
    
    // 释放内存
    delete[] h_image;
    cudaFree(d_image);
    cudaFree(d_histo);
    return 0;
}