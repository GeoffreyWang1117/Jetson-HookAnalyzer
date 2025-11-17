#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include "../kernels/optimized/kernels.h"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// Utility function to measure kernel performance
struct KernelMetrics {
    float avg_time_ms;
    float gflops;
    float bandwidth_gb_s;
    int occupancy_percent;
};

KernelMetrics benchmarkGEMM(int M, int N, int K, int warmup_runs = 5, int bench_runs = 20) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    // Initialize with random data
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        hook_analyzer::kernels::gemmFloatOptimized(d_A, d_B, d_C, M, N, K, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < bench_runs; i++) {
        hook_analyzer::kernels::gemmFloatOptimized(d_A, d_B, d_C, M, N, K, 0);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / bench_runs;

    // Calculate GFLOPS
    double flops = 2.0 * M * N * K;  // GEMM: 2*M*N*K operations
    double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

    // Calculate bandwidth (bytes read + written per operation)
    double bytes = (size_A + size_B + size_C);
    double bandwidth = (bytes / (avg_ms / 1000.0)) / 1e9;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    KernelMetrics metrics;
    metrics.avg_time_ms = avg_ms;
    metrics.gflops = gflops;
    metrics.bandwidth_gb_s = bandwidth;
    metrics.occupancy_percent = 0;  // Will calculate separately

    return metrics;
}

void analyzeOccupancy() {
    // Get device properties
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║            GPU Architecture Analysis                       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Shared Memory per SM: " << prop.sharedMemPerMultiprocessor / 1024 << " KB\n";
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB\n";
    std::cout << "Registers per SM: " << prop.regsPerMultiprocessor << "\n";
    std::cout << "Registers per Block: " << prop.regsPerBlock << "\n";
    std::cout << "Warp Size: " << prop.warpSize << "\n";
    std::cout << "L2 Cache Size: " << prop.l2CacheSize / 1024 / 1024 << " MB\n";
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz\n";
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits\n";

    // Calculate theoretical peak memory bandwidth
    double peak_bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
    std::cout << "Peak Memory Bandwidth: " << std::fixed << std::setprecision(1)
              << peak_bandwidth << " GB/s\n";

    std::cout << "\n";
}

void analyzeGEMMKernel() {
    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         Current GEMM Kernel Analysis (Tile 16x16)         ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    // Current kernel configuration
    const int TILE_SIZE = 16;
    const int THREADS_PER_BLOCK = TILE_SIZE * TILE_SIZE;  // 256
    const int SHARED_MEM_BYTES = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);  // 2 tiles

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    std::cout << "Kernel Configuration:\n";
    std::cout << "  Tile Size: " << TILE_SIZE << "x" << TILE_SIZE << "\n";
    std::cout << "  Threads per Block: " << THREADS_PER_BLOCK << "\n";
    std::cout << "  Shared Memory per Block: " << SHARED_MEM_BYTES << " bytes ("
              << SHARED_MEM_BYTES / 1024.0 << " KB)\n\n";

    // Calculate theoretical occupancy
    int max_blocks_per_sm_threads = prop.maxThreadsPerMultiProcessor / THREADS_PER_BLOCK;
    int max_blocks_per_sm_shared = prop.sharedMemPerMultiprocessor / SHARED_MEM_BYTES;
    int actual_blocks_per_sm = std::min(max_blocks_per_sm_threads, max_blocks_per_sm_shared);
    int active_warps_per_sm = actual_blocks_per_sm * (THREADS_PER_BLOCK / prop.warpSize);
    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    float occupancy = (float)active_warps_per_sm / max_warps_per_sm * 100.0f;

    std::cout << "Theoretical Occupancy Analysis:\n";
    std::cout << "  Max Blocks/SM (by threads): " << max_blocks_per_sm_threads << "\n";
    std::cout << "  Max Blocks/SM (by shared mem): " << max_blocks_per_sm_shared << "\n";
    std::cout << "  Actual Blocks/SM: " << actual_blocks_per_sm << "\n";
    std::cout << "  Active Warps/SM: " << active_warps_per_sm << " / " << max_warps_per_sm << "\n";
    std::cout << "  Occupancy: " << std::fixed << std::setprecision(1) << occupancy << "%\n\n";

    if (occupancy < 50) {
        std::cout << "⚠️  WARNING: Low occupancy! This limits performance.\n\n";
    }

    std::cout << "Potential Issues:\n";
    std::cout << "  1. Small tile size (16x16) → Less work per thread block\n";
    std::cout << "  2. No double buffering → Memory latency not hidden\n";
    std::cout << "  3. No vectorized loads → Underutilizing memory bandwidth\n";
    std::cout << "  4. Possible bank conflicts in shared memory access\n\n";
}

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║           GEMM Performance Analysis & Profiling              ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";

    // 1. GPU Architecture Analysis
    analyzeOccupancy();

    // 2. Current Kernel Analysis
    analyzeGEMMKernel();

    // 3. Performance Benchmarks at different sizes
    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              Performance Benchmarks                        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    struct TestCase {
        int M, N, K;
        const char* name;
    };

    std::vector<TestCase> test_cases = {
        {256, 256, 256, "Small (256x256)"},
        {512, 512, 512, "Medium (512x512)"},
        {1024, 1024, 1024, "Large (1024x1024)"},
        {2048, 2048, 2048, "Very Large (2048x2048)"},
    };

    std::cout << std::setw(20) << "Matrix Size" << " | "
              << std::setw(12) << "Time (ms)" << " | "
              << std::setw(12) << "GFLOPS" << " | "
              << std::setw(15) << "Bandwidth (GB/s)" << "\n";
    std::cout << std::string(75, '-') << "\n";

    for (const auto& tc : test_cases) {
        auto metrics = benchmarkGEMM(tc.M, tc.N, tc.K, 3, 10);

        std::cout << std::setw(20) << tc.name << " | "
                  << std::setw(12) << std::fixed << std::setprecision(4) << metrics.avg_time_ms << " | "
                  << std::setw(12) << std::fixed << std::setprecision(2) << metrics.gflops << " | "
                  << std::setw(15) << std::fixed << std::setprecision(2) << metrics.bandwidth_gb_s << "\n";
    }

    // 4. Compare with cuBLAS for 1024x1024
    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         Comparison with cuBLAS (1024x1024)                ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    int M = 1024, N = 1024, K = 1024;
    auto custom_metrics = benchmarkGEMM(M, N, K, 5, 20);

    // cuBLAS benchmark
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
    for (int i = 0; i < 5; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 d_B, N,
                                 d_A, K,
                                 &beta,
                                 d_C, N));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 20; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 d_B, N,
                                 d_A, K,
                                 &beta,
                                 d_C, N));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float cublas_total_ms;
    CHECK_CUDA(cudaEventElapsedTime(&cublas_total_ms, start, stop));
    float cublas_avg_ms = cublas_total_ms / 20;

    double flops = 2.0 * M * N * K;
    double cublas_gflops = (flops / (cublas_avg_ms / 1000.0)) / 1e9;

    std::cout << "Custom Kernel:\n";
    std::cout << "  Time: " << custom_metrics.avg_time_ms << " ms\n";
    std::cout << "  Performance: " << custom_metrics.gflops << " GFLOPS\n\n";

    std::cout << "cuBLAS:\n";
    std::cout << "  Time: " << cublas_avg_ms << " ms\n";
    std::cout << "  Performance: " << cublas_gflops << " GFLOPS\n\n";

    float ratio = (custom_metrics.gflops / cublas_gflops) * 100.0f;
    std::cout << "Performance Ratio: " << std::fixed << std::setprecision(2)
              << ratio << "% of cuBLAS\n";
    std::cout << "Speedup potential: " << std::fixed << std::setprecision(2)
              << (100.0f / ratio) << "x\n\n";

    if (ratio < 30) {
        std::cout << "❌ CRITICAL: Performance is very poor (<30% of cuBLAS)\n";
        std::cout << "   → Large optimization opportunity!\n\n";
    } else if (ratio < 60) {
        std::cout << "⚠️  WARNING: Performance is suboptimal (<60% of cuBLAS)\n";
        std::cout << "   → Significant optimization opportunity\n\n";
    } else {
        std::cout << "✓  Good performance (>60% of cuBLAS)\n\n";
    }

    // 5. Optimization Recommendations
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           Optimization Recommendations                    ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Priority 1 (Immediate Impact):\n";
    std::cout << "  1. Increase tile size to 32x32 or 64x64\n";
    std::cout << "     → More work per block, better amortization\n";
    std::cout << "  2. Implement vectorized loads (float4)\n";
    std::cout << "     → Improve memory throughput by 4x\n\n";

    std::cout << "Priority 2 (Advanced):\n";
    std::cout << "  3. Double buffering with shared memory\n";
    std::cout << "     → Hide memory latency with computation\n";
    std::cout << "  4. Rectangular tiles for non-square matrices\n";
    std::cout << "  5. Warp-level primitives (__shfl_sync)\n\n";

    std::cout << "Priority 3 (Expert):\n";
    std::cout << "  6. Use TensorCore WMMA API (FP16)\n";
    std::cout << "  7. Autotuning for different matrix sizes\n\n";

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    std::cout << "\n";
    return 0;
}
