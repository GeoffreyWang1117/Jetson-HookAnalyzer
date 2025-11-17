#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include "../kernels/optimized/kernels.h"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error" << std::endl; \
            exit(1); \
        } \
    } while(0)

struct BenchmarkResult {
    std::string name;
    float time_ms;
    float gflops;
    float speedup;
    float percent_of_cublas;
};

float benchmarkKernel(
    std::function<cudaError_t(const float*, const float*, float*, int, int, int, cudaStream_t)> kernel_func,
    int M, int N, int K, int warmup = 3, int iterations = 20)
{
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel_func(d_A, d_B, d_C, M, N, K, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel_func(d_A, d_B, d_C, M, N, K, 0);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iterations;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return avg_ms;
}

float benchmarkCuBLAS(int M, int N, int K, int warmup = 3, int iterations = 20) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iterations;

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return avg_ms;
}

void printHeader() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                                      ║\n";
    std::cout << "║         GEMM Optimization Comparison - Experiment 1 Results         ║\n";
    std::cout << "║                                                                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n\n";
}

void printResults(const std::vector<BenchmarkResult>& results, int M, int N, int K) {
    std::cout << "\n";
    std::cout << "Matrix Size: " << M << " x " << N << " x " << K << "\n";
    std::cout << std::string(100, '=') << "\n";
    std::cout << std::setw(30) << "Kernel Version" << " | "
              << std::setw(12) << "Time (ms)" << " | "
              << std::setw(12) << "GFLOPS" << " | "
              << std::setw(12) << "Speedup" << " | "
              << std::setw(15) << "% of cuBLAS" << "\n";
    std::cout << std::string(100, '-') << "\n";

    for (const auto& result : results) {
        std::cout << std::setw(30) << result.name << " | "
                  << std::setw(12) << std::fixed << std::setprecision(4) << result.time_ms << " | "
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.gflops << " | "
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.speedup << "x | "
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.percent_of_cublas << "%\n";
    }
    std::cout << std::string(100, '=') << "\n";
}

void findBestKernel(const std::vector<BenchmarkResult>& results) {
    auto best = std::max_element(results.begin() + 1, results.end(),  // Skip cuBLAS (first element)
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.gflops < b.gflops;
        });

    std::cout << "\n";
    std::cout << "🏆 Best Custom Kernel: " << best->name << "\n";
    std::cout << "   Performance: " << best->gflops << " GFLOPS\n";
    std::cout << "   vs cuBLAS: " << best->percent_of_cublas << "%\n";
    std::cout << "   Speedup over baseline: " << best->speedup << "x\n\n";
}

int main() {
    printHeader();

    // Test matrix size (1024x1024 - the problematic size)
    int M = 1024, N = 1024, K = 1024;

    std::cout << "Running benchmarks for " << M << "x" << N << "x" << K << " matrices...\n";
    std::cout << "(This may take 1-2 minutes)\n\n";

    // Benchmark cuBLAS (baseline)
    std::cout << "[1/6] Benchmarking cuBLAS...               ";
    std::cout.flush();
    float cublas_time = benchmarkCuBLAS(M, N, K);
    double flops = 2.0 * M * N * K;
    float cublas_gflops = (flops / (cublas_time / 1000.0)) / 1e9;
    std::cout << "✓ " << cublas_gflops << " GFLOPS\n";

    // Benchmark original 16x16 kernel
    std::cout << "[2/6] Benchmarking 16x16 tile (original)... ";
    std::cout.flush();
    auto kernel_16 = [](const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
        return hook_analyzer::kernels::gemmFloatOptimized(A, B, C, M, N, K, s);
    };
    float time_16 = benchmarkKernel(kernel_16, M, N, K);
    float gflops_16 = (flops / (time_16 / 1000.0)) / 1e9;
    std::cout << "✓ " << gflops_16 << " GFLOPS\n";

    // Benchmark 32x32 kernel
    std::cout << "[3/6] Benchmarking 32x32 tile...            ";
    std::cout.flush();
    auto kernel_32 = [](const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
        return hook_analyzer::kernels::gemmFloatOptimized32(A, B, C, M, N, K, s);
    };
    float time_32 = benchmarkKernel(kernel_32, M, N, K);
    float gflops_32 = (flops / (time_32 / 1000.0)) / 1e9;
    std::cout << "✓ " << gflops_32 << " GFLOPS\n";

    // Benchmark 32x32 vectorized kernel
    std::cout << "[4/6] Benchmarking 32x32 vectorized...      ";
    std::cout.flush();
    auto kernel_32v = [](const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
        return hook_analyzer::kernels::gemmFloatOptimized32Vectorized(A, B, C, M, N, K, s);
    };
    float time_32v = benchmarkKernel(kernel_32v, M, N, K);
    float gflops_32v = (flops / (time_32v / 1000.0)) / 1e9;
    std::cout << "✓ " << gflops_32v << " GFLOPS\n";

    // Benchmark 64x64 kernel
    std::cout << "[5/6] Benchmarking 64x64 tile...            ";
    std::cout.flush();
    auto kernel_64 = [](const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
        return hook_analyzer::kernels::gemmFloatOptimized64(A, B, C, M, N, K, s);
    };
    float time_64 = benchmarkKernel(kernel_64, M, N, K);
    float gflops_64 = (flops / (time_64 / 1000.0)) / 1e9;
    std::cout << "✓ " << gflops_64 << " GFLOPS\n";

    // Benchmark 32x32 double buffer kernel
    std::cout << "[6/6] Benchmarking 32x32 double buffer...   ";
    std::cout.flush();
    auto kernel_32db = [](const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
        return hook_analyzer::kernels::gemmFloatOptimized32DoubleBuffer(A, B, C, M, N, K, s);
    };
    float time_32db = benchmarkKernel(kernel_32db, M, N, K);
    float gflops_32db = (flops / (time_32db / 1000.0)) / 1e9;
    std::cout << "✓ " << gflops_32db << " GFLOPS\n";

    // Compile results
    std::vector<BenchmarkResult> results;
    results.push_back({"cuBLAS (baseline)", cublas_time, cublas_gflops, 1.0f, 100.0f});
    results.push_back({"16x16 tile (original)", time_16, gflops_16,
                       gflops_16 / gflops_16, (gflops_16 / cublas_gflops) * 100.0f});
    results.push_back({"32x32 tile", time_32, gflops_32,
                       gflops_32 / gflops_16, (gflops_32 / cublas_gflops) * 100.0f});
    results.push_back({"32x32 vectorized", time_32v, gflops_32v,
                       gflops_32v / gflops_16, (gflops_32v / cublas_gflops) * 100.0f});
    results.push_back({"64x64 tile", time_64, gflops_64,
                       gflops_64 / gflops_16, (gflops_64 / cublas_gflops) * 100.0f});
    results.push_back({"32x32 double buffer", time_32db, gflops_32db,
                       gflops_32db / gflops_16, (gflops_32db / cublas_gflops) * 100.0f});

    // Print results
    printResults(results, M, N, K);
    findBestKernel(results);

    // Summary
    std::cout << "╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    Optimization Summary                           ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n\n";

    float best_gflops = std::max(std::max(gflops_32, gflops_32v), std::max(gflops_64, gflops_32db));
    float improvement = (best_gflops - gflops_16) / gflops_16 * 100.0f;

    std::cout << "Baseline Performance (16x16):  " << gflops_16 << " GFLOPS "
              << "(" << (gflops_16/cublas_gflops)*100 << "% cuBLAS)\n";
    std::cout << "Best Optimized Performance:    " << best_gflops << " GFLOPS "
              << "(" << (best_gflops/cublas_gflops)*100 << "% cuBLAS)\n";
    std::cout << "\n";
    std::cout << "Improvement: +" << improvement << "%\n";
    std::cout << "Speedup: " << (best_gflops / gflops_16) << "x\n";
    std::cout << "\n";

    if ((best_gflops / cublas_gflops) > 0.50) {
        std::cout << "✓ SUCCESS: Achieved >50% of cuBLAS performance!\n";
    } else if ((best_gflops / cublas_gflops) > 0.30) {
        std::cout << "⚠  GOOD: Achieved >30% of cuBLAS performance.\n";
        std::cout << "   Further optimizations needed to reach 50% target.\n";
    } else {
        std::cout << "❌ More optimization needed to reach target performance.\n";
    }

    std::cout << "\n";
    return 0;
}
