#include <iostream>
#include <cuda_runtime.h>

template<int block_size=16, bool conflict_free=true>
__global__ void matrixTransposeSharedMemory(const float* input, float* output, int width, int height) {
    // 定义共享内存
    __shared__ float smem[block_size][block_size + (conflict_free ? 1 : 0)];

    int tx = threadIdx.x / block_size;
    int ty = threadIdx.x % block_size;

    // 计算线程的全局坐标
    int x = blockIdx.x * block_size + tx; // 原矩阵列 id
    int y = blockIdx.y * block_size + ty; // 原矩阵行 id

    // 将数据从全局内存加载到共享内存, 加载过程中同时把这一个子矩阵进行转置
    if (x < width && y < height) {
        smem[tx][ty] = input[y * width + x];
        // printf("smem[%d][%d] = %d\n", tx, ty, (int)smem[tx][ty]);
    }

    // 等待所有线程完成数据加载
    __syncthreads();

    // 将转置后的数据从共享内存写回全局内存
    if (x < width && y < height) {
        output[x * height + y] = smem[tx][ty];
    }
}

void print_matrix(float* matrix, int rows, int cols, const char* description) {
    printf("====== %s ======\n", description);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%d ", int(matrix[i * cols + j]));
        }
        printf("\n");
    }
}

void transpose_cpu(const float* input, float* output, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            output[j * height + i] = input[i * width + j];
        }
    }
}

void check(const float* a, const float* b, int N) {
    for (int i = 0; i < N; ++i) {
        if (fabs(a[i] - b[i]) > 1e-9) {printf("failed\n"); return;}
    }
    printf("passed\n");
}

int main() {
    // 定义矩阵大小
    const int width = 1024;   // 列
    const int height = 2048;  // 行

    // 申请主机内存
    float* h_input = (float*)malloc(width * height * sizeof(float));
    float* h_output = (float*)malloc(width * height * sizeof(float));
    float* h_output_ref = (float*)malloc(width * height * sizeof(float));
    // 初始化矩阵数据
    for (int i = 0; i < width * height; ++i) {
        h_input[i] = i % width; // 不同行相同列的元素都一样
    }
    // 打印原始矩阵
    // print_matrix(h_input, height, width, "原始矩阵");

    // 分配设备内存
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, width * height * sizeof(float));
    cudaMalloc((void**)&d_output, height * width * sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // 定义线程块和网格大小
    constexpr int block_size = 16;
    dim3 blockSize(block_size * block_size); // 每个线程块的大小
    dim3 gridSize((width + block_size - 1) / block_size, (height + block_size - 1) / block_size);

    // 调用核函数
    matrixTransposeSharedMemory<block_size><<<gridSize, blockSize>>>(d_input, d_output, width, height);

    // 将结果从设备复制回主机
    cudaMemcpy(h_output, d_output, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    // print_matrix(h_output, width, height, "转置矩阵");

    transpose_cpu(h_input, h_output_ref, width, height);
    check(h_output, h_output_ref, width * height);

    
    // profile
    int TEST_TIMES = 100;
    // warming up
    for (int i = 0; i < TEST_TIMES; ++i) {
        matrixTransposeSharedMemory<block_size><<<gridSize, blockSize>>>(d_input, d_output, width, height);
    }

    float time_elapsed = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);    //记录开始时间
    for (int i = 0; i < TEST_TIMES; ++i) {
        matrixTransposeSharedMemory<block_size><<<gridSize, blockSize>>>(d_input, d_output, width, height);
    }
    cudaEventRecord(stop,0);    //记录结束时间
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);    //计算时间差
    std::cout << "kernel elasped time = " << time_elapsed/TEST_TIMES << std::endl;
    
    // 释放主机内存
    free(h_input);
    free(h_output);
    free(h_output_ref);

    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}