#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace hook_analyzer {
namespace kernels {

// Matrix multiplication: C = A * B
// A: M x K, B: K x N, C: M x N
cudaError_t gemmFloat(const float* A, const float* B, float* C,
                      int M, int N, int K,
                      cudaStream_t stream = 0);

// Optimized GEMM with shared memory
cudaError_t gemmFloatOptimized(const float* A, const float* B, float* C,
                               int M, int N, int K,
                               cudaStream_t stream = 0);

// Element-wise operations
cudaError_t addFloat(const float* A, const float* B, float* C,
                     size_t N, cudaStream_t stream = 0);

cudaError_t mulFloat(const float* A, const float* B, float* C,
                     size_t N, cudaStream_t stream = 0);

// Activation functions
cudaError_t reluFloat(const float* input, float* output,
                      size_t N, cudaStream_t stream = 0);

cudaError_t sigmoidFloat(const float* input, float* output,
                         size_t N, cudaStream_t stream = 0);

// Softmax
cudaError_t softmaxFloat(const float* input, float* output,
                         int batch_size, int num_classes,
                         cudaStream_t stream = 0);

// Batch normalization
cudaError_t batchNormFloat(const float* input, float* output,
                           const float* mean, const float* variance,
                           const float* gamma, const float* beta,
                           int batch_size, int channels, int spatial_size,
                           float epsilon, cudaStream_t stream = 0);

// Memory operations
cudaError_t memsetFloat(float* ptr, float value, size_t N,
                        cudaStream_t stream = 0);

// Reduction operations
cudaError_t reduceSum(const float* input, float* output,
                      size_t N, cudaStream_t stream = 0);

cudaError_t reduceMax(const float* input, float* output,
                      size_t N, cudaStream_t stream = 0);

} // namespace kernels
} // namespace hook_analyzer
