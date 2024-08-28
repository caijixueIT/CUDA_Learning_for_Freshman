#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template<typename T>
bool check(T *out, T *res, int N){
    for(int i=0; i<N; i++){
        if(out[i]!= res[i])
            return false;
    }
    return true;
}

template <typename T, int THREADS_PER_BLOCK=256, int UNROLL=8, int GRANULARITY=16>
struct ele_wise_op_config {
    using element_type = T;
    static constexpr int threads = THREADS_PER_BLOCK;
    static constexpr int granularity = GRANULARITY;
    static constexpr int unroll = UNROLL;
    static constexpr int T_per_access = GRANULARITY / sizeof(T);
};

struct params {
    int N;
};

struct vec_add_params : params {
    void* a;
    void* b;
    void* c;
};

template<typename T, int num>
struct packed_vals {
    T data[num];
    __host__ __device__ __forceinline__ packed_vals& operator= (const packed_vals& rhs) {
        for (int i = 0; i < num; ++i) {
            this->data[i] = rhs.data[i];
        }
        return *this;
    }
};


template <typename cfg>
__global__ void vector_add_kernel_template(vec_add_params& param) {
    int offset = (blockIdx.x * cfg::threads * cfg::unroll + threadIdx.x) * cfg::T_per_access;
    using T = typename cfg::element_type;
    
    #pragma unroll
    for (int i = 0; i < cfg::unroll; ++i) {
        int idx = offset + (i * cfg::threads) * cfg::T_per_access;
        using packed = packed_vals<T, cfg::T_per_access>;
        packed aa = ((packed*)(((T*)param.a) + idx))[0];
        packed bb = ((packed*)(((T*)param.b) + idx))[0];
        packed cc;
        #pragma unroll
        for (int i = 0; i < cfg::T_per_access; ++i) {
            cc.data[i] = aa.data[i] + bb.data[i];
        }
        // ((packed*)(((T*)param.c) + idx))[0] = cc; // 此处赋值会报错 ？？？
    }
}

template <typename cfg>
void launch_vector_add_kernel(vec_add_params& param) {
    int num_blocks = param.N / cfg::threads / cfg::unroll / cfg::T_per_access;
    dim3 grid_dim(num_blocks);
    dim3 block_dim(cfg::threads);
    // vector_add_kernel_template<cfg><<<grid_dim, block_dim>>>(param);
    cudaDeviceSynchronize();//这个同步不能省略
    printf("after kernel function : %s\n",cudaGetErrorString(cudaGetLastError()));
}

int main() {
    using T = __nv_bfloat16; // half // float
    constexpr int N = 32 * 1024 * 1024;

    T* h_a = (T*)malloc(N * sizeof(T));
    T* h_b = (T*)malloc(N * sizeof(T));
    T* h_c = (T*)malloc(N * sizeof(T));
    T* h_c_ref = (T*)malloc(N * sizeof(T));

    T* d_a;
    T* d_b;
    T* d_c;
    cudaMalloc((void**)&d_a, N * sizeof(T));
    cudaMalloc((void**)&d_b, N * sizeof(T));
    cudaMalloc((void**)&d_c, N * sizeof(T));
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // reference by CPU
    for (int i = 0; i < N; ++i) {
        h_c_ref[i] = h_a[i] + h_b[i];
    }

    vec_add_params param;
    param.N = N;
    param.a = d_a;
    param.b = d_b;
    param.c = d_c;

    launch_vector_add_kernel<ele_wise_op_config<T>>(param);
    cudaMemcpy(d_c, h_c, N * sizeof(T), cudaMemcpyDeviceToHost);
    if (!check(h_c, h_c_ref, N)) {
        std::cout << "vector_add_kernel error!" <<std::endl;
    } else {
        std::cout << "vector_add_kernel pass!" <<std::endl;
    }

    int TEST_TIMES = 1;
    float time_elapsed=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warming up
    for (int i = 0; i < TEST_TIMES; ++i) {
        launch_vector_add_kernel<ele_wise_op_config<T, 256, 32, 16>>(param);
    }
    // profile
    cudaEventRecord(start,0);
    for (int i = 0; i < TEST_TIMES; ++i) {
        launch_vector_add_kernel<ele_wise_op_config<T, 256, 32, 16>>(param);
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);
    std::cout << "vector_add_kernel elasped time = " << time_elapsed/TEST_TIMES << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_ref);
    return 0;

}

