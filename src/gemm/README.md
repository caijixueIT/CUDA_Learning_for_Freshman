Running the kernels on a NVIDIA RTX 4090 (Ada, SM89):

GFLOPs at matrix size M=4096, N=4096, K=4096:
<!-- benchmark_results -->
| Kernel                              |   GFLOPs/s | Performance relative to cuBLAS   |
|:------------------------------------|-----------:|:---------------------------------|
| 1: Naive                            |      681.3 | 1.4%                             |
| 2: GMEM Coalescing                  |     4834   | 9.8%                             |
| 3: SMEM Caching                     |     6552   | 13.3%                            |
| 4: 1D Blocktiling                   |    19282.1 | 39.1%                            |
| 5: 2D Blocktiling                   |    36052.5 | 73.1%                            |
| 7: Avoid Bank Conflicts (Linearize) |    36745.5 | 74.5%                            |
| 8: Avoid Bank Conflicts (Offset)    |    37437.6 | 75.9%                            |
| 12: Double Buffering                |    41475.7 | 84.1%                            |
| 9: Autotuning                       |    42945.9 | 87.1%                            |
| 6: Vectorized Mem Access            |    43326.3 | 87.9%                            |
| 10: Warptiling                      |    46349.9 | 94.0%                            |
| 0: cuBLAS                           |    49305.6 | 100.0%                           |
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