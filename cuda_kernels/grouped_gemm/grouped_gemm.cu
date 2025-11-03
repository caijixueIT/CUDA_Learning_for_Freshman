#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// 定义分块大小
#define BLOCK_SIZE 16
#define TILE_SIZE 16

__global__ void grouped_gemm_shared_kernel(
    int* m, int* n, int* k,    // 每个GEMM的维度数组
    float** A_array,            // A矩阵指针数组
    float** B_array,            // B矩阵指针数组
    float** C_array,            // C矩阵指针数组
    int group_size,             // GEMM操作的数量
    int* lda, int* ldb, int* ldc // 每个矩阵的leading dimension
) {
    // 共享内存声明 - 用于缓存A和B的块
    __shared__ float As[BLOCK_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][BLOCK_SIZE];
    
    int group_id = blockIdx.x;
    if (group_id >= group_size) return;
    
    // 获取当前GEMM的参数
    int curr_m = m[group_id];
    int curr_n = n[group_id];
    int curr_k = k[group_id];
    
    float* A = A_array[group_id];
    float* B = B_array[group_id];
    float* C = C_array[group_id];
    
    int curr_lda = lda[group_id];
    int curr_ldb = ldb[group_id];
    int curr_ldc = ldc[group_id];
    
    // 每个线程计算的输出位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.z * blockDim.z + threadIdx.x;
    
    float sum = 0.0f;
    
    // 分块处理矩阵乘法
    for (int t = 0; t < (curr_k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 协作加载A的块到共享内存
        int A_row = row;
        int A_col = t * TILE_SIZE + threadIdx.x;
        if (A_row < curr_m && A_col < curr_k) {
            As[threadIdx.y][threadIdx.x] = A[A_row + A_col * curr_lda];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // 协作加载B的块到共享内存
        int B_row = t * TILE_SIZE + threadIdx.y;
        int B_col = col;
        if (B_row < curr_k && B_col < curr_n) {
            Bs[threadIdx.y][threadIdx.x] = B[B_row + B_col * curr_ldb];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();  // 确保所有线程完成数据加载
        
        // 计算当前块的乘积
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        
        __syncthreads();  // 确保所有线程完成计算
    }
    
    // 写入结果
    if (row < curr_m && col < curr_n) {
        C[row + col * curr_ldc] = sum;
    }
}

// 包装函数
void grouped_gemm_shared(
    const std::vector<int>& m_vec,
    const std::vector<int>& n_vec,
    const std::vector<int>& k_vec,
    const std::vector<float*>& A_vec,
    const std::vector<float*>& B_vec,
    std::vector<float*>& C_vec,
    const std::vector<int>& lda_vec,
    const std::vector<int>& ldb_vec,
    const std::vector<int>& ldc_vec
) {
    int group_size = m_vec.size();
    
    // 在设备上分配内存并拷贝数据
    int *d_m, *d_n, *d_k;
    float **d_A_array, **d_B_array, **d_C_array;
    int *d_lda, *d_ldb, *d_ldc;
    
    // 分配设备内存
    cudaMalloc(&d_m, group_size * sizeof(int));
    cudaMalloc(&d_n, group_size * sizeof(int));
    cudaMalloc(&d_k, group_size * sizeof(int));
    cudaMalloc(&d_lda, group_size * sizeof(int));
    cudaMalloc(&d_ldb, group_size * sizeof(int));
    cudaMalloc(&d_ldc, group_size * sizeof(int));
    
    // 分配指针数组
    cudaMalloc(&d_A_array, group_size * sizeof(float*));
    cudaMalloc(&d_B_array, group_size * sizeof(float*));
    cudaMalloc(&d_C_array, group_size * sizeof(float*));
    
    // 拷贝数据到设备
    cudaMemcpy(d_m, m_vec.data(), group_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, n_vec.data(), group_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k_vec.data(), group_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lda, lda_vec.data(), group_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ldb, ldb_vec.data(), group_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ldc, ldc_vec.data(), group_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // 创建主机端的指针数组
    std::vector<float*> h_A_array(group_size);
    std::vector<float*> h_B_array(group_size);
    std::vector<float*> h_C_array(group_size);
    
    for (int i = 0; i < group_size; ++i) {
        h_A_array[i] = A_vec[i];
        h_B_array[i] = B_vec[i];
        h_C_array[i] = C_vec[i];
    }
    
    // 拷贝指针数组到设备
    cudaMemcpy(d_A_array, h_A_array.data(), group_size * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_array, h_B_array.data(), group_size * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_array, h_C_array.data(), group_size * sizeof(float*), cudaMemcpyHostToDevice);
    
    // 配置内核参数
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    
    // 为每个GEMM操作计算需要的grid维度
    for (int i = 0; i < group_size; ++i) {
        int grid_x = 1;  // 每个GEMM一个block.x
        int grid_y = (m_vec[i] + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int grid_z = (n_vec[i] + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        dim3 grid(grid_x, grid_y, grid_z);
        
        // 启动内核
        grouped_gemm_shared_kernel<<<grid, block>>>(
            d_m + i, d_n + i, d_k + i,
            d_A_array + i, d_B_array + i, d_C_array + i,
            1,  // 每次处理一个GEMM
            d_lda + i, d_ldb + i, d_ldc + i
        );
    }
    
    // 释放设备内存
    cudaFree(d_m);
    cudaFree(d_n);
    cudaFree(d_k);
    cudaFree(d_A_array);
    cudaFree(d_B_array);
    cudaFree(d_C_array);
    cudaFree(d_lda);
    cudaFree(d_ldb);
    cudaFree(d_ldc);
}

int main() {
    const int group_size = 3;
    
    // 示例数据
    std::vector<int> m = {128, 256, 64};
    std::vector<int> n = {128, 128, 256};
    std::vector<int> k = {256, 128, 128};
    
    std::vector<int> lda = {128, 256, 64};
    std::vector<int> ldb = {256, 128, 128};
    std::vector<int> ldc = {128, 256, 64};
    
    // 分配和初始化矩阵
    std::vector<float*> A(group_size), B(group_size), C(group_size);
    for (int i = 0; i < group_size; ++i) {
        cudaMalloc(&A[i], m[i] * k[i] * sizeof(float));
        cudaMalloc(&B[i], k[i] * n[i] * sizeof(float));
        cudaMalloc(&C[i], m[i] * n[i] * sizeof(float));
        
        // 这里应该初始化矩阵数据...
    }
    
    // 调用Grouped GEMM
    grouped_gemm_shared(m, n, k, A, B, C, lda, ldb, ldc);
    
    // 同步设备
    cudaDeviceSynchronize();
    
    // 清理
    for (int i = 0; i < group_size; ++i) {
        cudaFree(A[i]);
        cudaFree(B[i]);
        cudaFree(C[i]);
    }
    
    return 0;
}