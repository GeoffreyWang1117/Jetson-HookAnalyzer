#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "kernels/optimized/kernels.h"
#include "profiler/profiler.h"

using namespace hook_analyzer;

void benchmark_gemm(int M, int N, int K, int iterations = 100) {
    std::cout << "\n=== GEMM Benchmark (M=" << M << ", N=" << N << ", K=" << K << ") ===" << std::endl;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Initialize with random data
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);
    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

    // Warmup
    kernels::gemmFloatOptimized(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Benchmark custom kernel
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        kernels::gemmFloatOptimized(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float avg_time_custom = duration.count() / (float)iterations / 1000.0f;

    // Benchmark cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha, d_B, N, d_A, K,
                    &beta, d_C, N);
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float avg_time_cublas = duration.count() / (float)iterations / 1000.0f;

    // Calculate GFLOPS
    float gflops_custom = (2.0f * M * N * K) / (avg_time_custom * 1e6);
    float gflops_cublas = (2.0f * M * N * K) / (avg_time_cublas * 1e6);

    std::cout << "Custom Kernel: " << avg_time_custom << " ms, " << gflops_custom << " GFLOPS" << std::endl;
    std::cout << "cuBLAS:        " << avg_time_cublas << " ms, " << gflops_cublas << " GFLOPS" << std::endl;
    std::cout << "Performance ratio: " << (gflops_custom / gflops_cublas * 100) << "%" << std::endl;

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void benchmark_elementwise(size_t N, int iterations = 1000) {
    std::cout << "\n=== Element-wise Operations Benchmark (N=" << N << ") ===" << std::endl;

    size_t bytes = N * sizeof(float);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Test addition
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        kernels::addFloat(d_a, d_b, d_c, N);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float avg_time = duration.count() / (float)iterations / 1000.0f;

    float bandwidth = (3.0f * bytes) / (avg_time * 1e6);  // GB/s
    std::cout << "Add: " << avg_time << " ms, " << bandwidth << " GB/s" << std::endl;

    // Test ReLU
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        kernels::reluFloat(d_a, d_c, N);
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    avg_time = duration.count() / (float)iterations / 1000.0f;

    bandwidth = (2.0f * bytes) / (avg_time * 1e6);
    std::cout << "ReLU: " << avg_time << " ms, " << bandwidth << " GB/s" << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    std::cout << "==================================" << std::endl;
    std::cout << "HookAnalyzer Kernel Benchmarks" << std::endl;
    std::cout << "==================================" << std::endl;

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "\nGPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;

    // Run benchmarks
    benchmark_gemm(512, 512, 512);
    benchmark_gemm(1024, 1024, 1024);
    benchmark_elementwise(1024 * 1024);
    benchmark_elementwise(16 * 1024 * 1024);

    std::cout << "\n==================================" << std::endl;
    std::cout << "Benchmarks completed!" << std::endl;
    std::cout << "==================================" << std::endl;

    return 0;
}
