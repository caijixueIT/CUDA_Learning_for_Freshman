#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

__global__ void shfl(float* arr, int n) {
    int tid = threadIdx.x;
    float val = tid < n ? arr[tid] : 0.f;
    val = __shfl_down_sync(0xffffffff, val, 4, 16);
    arr[tid] = val;
}

int main() {
    constexpr int N = 32;
    float* arr = (float*)malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) {
        arr[i] = i + 1;
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            printf("%.2f  ", arr[i * 8 + j]);
        }
        printf("\n");
    }
    printf("########################################\n");
    float* d_arr;
    cudaMalloc((void**)&d_arr, sizeof(float) * N);
    cudaMemcpy(d_arr, arr, sizeof(float) * N, cudaMemcpyHostToDevice);
    shfl<<<1, N>>>(d_arr, N);
    cudaDeviceSynchronize();
    cudaMemcpy(arr, d_arr, sizeof(float) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            printf("%.2f  ", arr[i * 8 + j]);
        }
        printf("\n");
    }
    
    return 0;
}