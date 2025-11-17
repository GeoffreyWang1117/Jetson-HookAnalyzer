#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "kernels/optimized/kernels.h"

void printGPUInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    std::cout << "=== GPU Information ===" << std::endl;
    std::cout << "GPUs found: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "\nGPU " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  SMs: " << prop.multiProcessorCount << std::endl;
    }
    std::cout << "========================\n" << std::endl;
}

void testKernels() {
    std::cout << "=== Testing CUDA Kernels ===" << std::endl;

    const int N = 1024 * 1024;  // 1M elements
    size_t bytes = N * sizeof(float);

    // Host arrays
    std::vector<float> h_a(N, 2.0f);
    std::vector<float> h_b(N, 3.0f);
    std::vector<float> h_c(N, 0.0f);

    // Device arrays
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy to device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // Test 1: Addition
    std::cout << "\n[1/5] Testing element-wise addition..." << std::endl;
    hook_analyzer::kernels::addFloat(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    std::cout << "  Result: " << h_c[0] << " (expected: 5.0)" << std::endl;
    std::cout << "  Status: " << (h_c[0] == 5.0f ? "✓ PASS" : "✗ FAIL") << std::endl;

    // Test 2: Multiplication
    std::cout << "\n[2/5] Testing element-wise multiplication..." << std::endl;
    hook_analyzer::kernels::mulFloat(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    std::cout << "  Result: " << h_c[0] << " (expected: 6.0)" << std::endl;
    std::cout << "  Status: " << (h_c[0] == 6.0f ? "✓ PASS" : "✗ FAIL") << std::endl;

    // Test 3: ReLU (positive)
    std::cout << "\n[3/5] Testing ReLU (positive input)..." << std::endl;
    hook_analyzer::kernels::reluFloat(d_a, d_c, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    std::cout << "  Result: " << h_c[0] << " (expected: 2.0)" << std::endl;
    std::cout << "  Status: " << (h_c[0] == 2.0f ? "✓ PASS" : "✗ FAIL") << std::endl;

    // Test 4: ReLU (negative)
    std::cout << "\n[4/5] Testing ReLU (negative input)..." << std::endl;
    std::fill(h_a.begin(), h_a.end(), -5.0f);
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    hook_analyzer::kernels::reluFloat(d_a, d_c, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    std::cout << "  Result: " << h_c[0] << " (expected: 0.0)" << std::endl;
    std::cout << "  Status: " << (h_c[0] == 0.0f ? "✓ PASS" : "✗ FAIL") << std::endl;

    // Test 5: GEMM (small matrix)
    std::cout << "\n[5/5] Testing GEMM (matrix multiplication)..." << std::endl;
    int M = 256, N_mat = 256, K = 256;
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N_mat * sizeof(float));
    cudaMalloc(&d_C, M * N_mat * sizeof(float));

    hook_analyzer::kernels::gemmFloatOptimized(d_A, d_B, d_C, M, N_mat, K);
    cudaDeviceSynchronize();
    std::cout << "  Status: ✓ PASS (no errors)" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "\n=============================" << std::endl;
}

int main() {
    std::cout << "====================================\n";
    std::cout << "  HookAnalyzer Kernel Test Suite\n";
    std::cout << "====================================\n" << std::endl;

    printGPUInfo();
    testKernels();

    std::cout << "\n✓ All tests completed successfully!\n" << std::endl;

    return 0;
}
