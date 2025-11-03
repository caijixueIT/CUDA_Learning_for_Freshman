#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <random>
#include <chrono>
#include <queue>
#include <vector>

#define BLOCK_SISE 256
#define MAX_BLOCK_SIZE 512
#define TOP_K 10

#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
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

template <int THREADS_PER_BLOCK = 256, int TopK = 8>
__global__ void topK_reduction(float* input, float* output, int* indices, int n, int k, int* index_map=nullptr) {
    assert(blockDim.x == THREADS_PER_BLOCK);
    assert(TopK == k);

    extern __shared__ float shared[];
    float* shared_values = shared;
    int* shared_indices = (int*)&shared_values[k * blockDim.x];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float thread_top_k[TopK];
    int thread_top_k_indices[TopK];
    for (int i = 0; i < k; i++) {
        thread_top_k[i] = -FLT_MAX;
        thread_top_k_indices[i] = -1;
    }

    // 插入排序，获得每个线程负责的多个元素中的 top k
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float val = input[i];
        
        // 如果值大于线程内最小值，则插入
        if (val > thread_top_k[k-1]) {
            // 找到插入位置
            int pos = k-1;
            while (pos > 0 && val > thread_top_k[pos-1]) {
                pos--;
            }

            // 移动元素
            for (int j = k-1; j > pos; j--) {
                thread_top_k[j] = thread_top_k[j-1];
                thread_top_k_indices[j] = thread_top_k_indices[j-1];
            }
            
            // 插入新值
            thread_top_k[pos] = val;
            thread_top_k_indices[pos] = i;
        }
    }

    // 将线程的 top k 结果写到共享内存
    for (int i = 0; i < k; i++) {
        shared_values[tid * k + i] = thread_top_k[i];
        shared_indices[tid * k + i] = thread_top_k_indices[i];
    }
    __syncthreads();

    // 在共享内存中规约，得到线程块的 top k
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            // 合并两个由大到小排列的长度为 k 的有序数组，取出前 k 个元素
            int l = 0, m = 0, n = 0;
            float* val_a = shared_values + tid * k;
            float* val_b = shared_values + (tid + stride) * k;
            int* indices_a = shared_indices + tid * k;
            int* indices_b = shared_indices + (tid + stride) * k;

            while (l < k && m < k && n < k) {
                if (val_a[m] >= val_b[n]) {
                    thread_top_k[l] = val_a[m];
                    thread_top_k_indices[l] = indices_a[m];
                    l++;
                    m++;
                } else {
                    thread_top_k[l] = val_b[n];
                    thread_top_k_indices[l] = indices_b[n];
                    l++;
                    n++;
                }
            }

            for (int i = 0; i < k; i++) {
                shared_values[tid * k + i] = thread_top_k[i];
                shared_indices[tid * k + i] = thread_top_k_indices[i];
            }
        }
        __syncthreads();
    }

    // 将线程块的 top k 拷贝到全局内存
    if (tid == 0) {
        for (int i = 0; i < k; i++) {
            output[blockIdx.x * k + i] = shared_values[tid * k + i];
            indices[blockIdx.x * k + i] = index_map ? index_map[shared_indices[tid * k + i]] : shared_indices[tid * k + i];
            // if (THREADS_PER_BLOCK > 256 && blockIdx.x < 2) {
            //     printf("output[%d]: %f, indices[%d]: %d\n", i, output[blockIdx.x * k + i], i, indices[blockIdx.x * k + i]);
            // }
        }
    }
}

// 需要第二次内核调用或主机代码来合并各块的结果
// 两阶段TopK实现
template <int TopK = 8>
void topK_cuda(float* d_input, float* d_output, int* d_indices, int n, int k) {
    assert(k == TopK);
    // 第一阶段：每个块计算局部TopK
    constexpr int block_size = BLOCK_SISE;
    int grid_size = (n + block_size - 1) / block_size;
    
    // 分配临时存储空间
    float* d_temp_values;
    int* d_temp_indices;
    cudaMalloc(&d_temp_values, grid_size * k * sizeof(float));
    cudaMalloc(&d_temp_indices, grid_size * k * sizeof(int));
    
    // 调用第一阶段内核
    topK_reduction<block_size, TopK><<<grid_size, block_size, block_size*2*k*sizeof(float)>>>(
        d_input, d_temp_values, d_temp_indices, n, k);
    
    // 第二阶段：合并各块结果
    constexpr int max_block_size = MAX_BLOCK_SIZE;
    topK_reduction<max_block_size, TopK><<<1, max_block_size, max_block_size*2*k*sizeof(float)>>>(
        d_temp_values, d_output, d_indices, grid_size*k, k, d_temp_indices);

    
    CHECK(cudaDeviceSynchronize());
    
    // 清理
    cudaFree(d_temp_values);
    cudaFree(d_temp_indices);
}

struct TopKItem {
    float value;
    int index;
    bool operator<(const TopKItem& other) const {
        return value > other.value;
    }
};

void topK_cpu(const float* input, float* output, int* indices, int n, int k) { 
    std::vector<TopKItem> data(n);
    for (int i = 0; i < n; i++) {
        data[i] = {input[i], i};
    }

    std::priority_queue<TopKItem> pq;
    for (int i = 0; i < k; i++) {
        pq.push(data[i]);
    }
    for (int i = k; i < n; i++) {
        if (data[i] < pq.top()) {
            pq.pop();
            pq.push(data[i]);
        }
    }
    int i = k - 1;
    while (!pq.empty()) {
        output[i] = pq.top().value;
        indices[i] = pq.top().index;
        pq.pop();
        i--;
    }
}

bool check(float* output, float* output_ref, int* indices, int* indices_ref, int k) {
    // for (int i = 0; i < k; i++) {
    //     printf("output[%d]: %f, indices[%d]: %d\n", i, output[i], i, indices[i]);
    //     printf("output_ref[%d]: %f, indices_ref[%d]: %d\n", i, output_ref[i], i, indices_ref[i]);
    //     printf("\n");
    // }
    for (int i = 0; i < k; i++) {
        if (output[i] != output_ref[i] || indices[i] != indices_ref[i]) {
            return false;
        }
    }
    return true;
}

int main() { 
    int n = 100*1000;
    constexpr int k = TOP_K;
    float* h_input = (float*)malloc(n * sizeof(float));
    float* h_output = (float*)malloc(k * sizeof(float));
    int* h_indices = (int*)malloc(k * sizeof(int));
    float* h_output_ref = (float*)malloc(k * sizeof(float));
    int* h_indices_ref = (int*)malloc(k * sizeof(int));

    init_tensor(h_input, n);
    topK_cpu(h_input, h_output_ref, h_indices_ref, n, k);

    float* d_input;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    float* d_output;
    cudaMalloc(&d_output, k * sizeof(float));
    int* d_indices;
    cudaMalloc(&d_indices, k * sizeof(int));
    topK_cuda<k>(d_input, d_output, d_indices, n, k);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices, d_indices, k * sizeof(int), cudaMemcpyDeviceToHost);

    if (check(h_output, h_output_ref, h_indices, h_indices_ref, k)) {
        printf("TopK test passed!\n");
    } else {
        printf("TopK test failed!\n");
    }

    return 0;
}
