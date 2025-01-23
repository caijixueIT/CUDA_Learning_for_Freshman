Running the kernels on a NVIDIA RTX 4090 (Ada, SM89):

GFLOPs at matrix size M=4096, N=4096, K=4096:
<!-- benchmark_results -->
| Kernel                              |   GFLOPs/s | Performance relative to cuBLAS   |
|:------------------------------------|-----------:|:---------------------------------|
| 1: Naive                            |      673.5 | 1.3%                             |
| 2: GMEM Coalescing                  |     5217.8 | 10.4%                            |
| 3: SMEM Caching                     |     6481.4 | 12.9%                            |
| 4: 1D Blocktiling                   |    19308   | 38.6%                            |
| 5: 2D Blocktiling                   |    35636.5 | 71.2%                            |
| 7: Avoid Bank Conflicts (Linearize) |    36609.9 | 73.1%                            |
| 8: Avoid Bank Conflicts (Offset)    |    37057.6 | 74.0%                            |
| 12: Double Buffering                |    41475.7 | 82.8%                            |
| 6: Vectorized Mem Access            |    42464.2 | 84.8%                            |
| 9: Autotuning                       |    42827.2 | 85.5%                            |
| 10: Warptiling                      |    45319.9 | 90.5%                            |
| 0: cuBLAS                           |    50067.4 | 100.0%                           |
<!-- benchmark_results -->

## Setup

1. Install dependencies: CUDA toolkit 12, Python (+ Seaborn), CMake, Ninja.
2. Configure NVCC compilation parameters. Look up your GPUs compute
   capability [here](https://developer.nvidia.com/cuda-gpus). Then configure the `CMakeLists.txt` and change:
    ```cmake
    set(CUDA_COMPUTE_CAPABILITY 89)
    ```
3. Build: `mkdir build && cd build && cmake .. && cmake --build .`
4. Run one of the kernels: `DEVICE=<device_id> ./sgemm <kernel number>`
5. Profiling via [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) (ncu): `make profile KERNEL=<kernel number>`


参考博客[如何优化 CUDA 矩阵乘内核以获得类似 cuBLAS 的性能](https://blog.csdn.net/LostUnravel/article/details/138034380)