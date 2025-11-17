#include "kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace hook_analyzer {
namespace kernels {

// ============================================================================
// Matrix Multiplication Kernels
// ============================================================================

__global__ void gemmKernel(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

#define TILE_SIZE 16

__global__ void gemmKernelOptimized(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && (t * TILE_SIZE + tx) < K)
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if ((t * TILE_SIZE + ty) < K && col < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

cudaError_t gemmFloat(const float* A, const float* B, float* C,
                      int M, int N, int K, cudaStream_t stream) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    gemmKernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);

    return cudaGetLastError();
}

cudaError_t gemmFloatOptimized(const float* A, const float* B, float* C,
                               int M, int N, int K, cudaStream_t stream) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    gemmKernelOptimized<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);

    return cudaGetLastError();
}

// ============================================================================
// Element-wise Operations
// ============================================================================

__global__ void addKernel(const float* A, const float* B, float* C, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void mulKernel(const float* A, const float* B, float* C, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * B[idx];
    }
}

cudaError_t addFloat(const float* A, const float* B, float* C,
                     size_t N, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    addKernel<<<gridSize, blockSize, 0, stream>>>(A, B, C, N);
    return cudaGetLastError();
}

cudaError_t mulFloat(const float* A, const float* B, float* C,
                     size_t N, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    mulKernel<<<gridSize, blockSize, 0, stream>>>(A, B, C, N);
    return cudaGetLastError();
}

// ============================================================================
// Activation Functions
// ============================================================================

__global__ void reluKernel(const float* input, float* output, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void sigmoidKernel(const float* input, float* output, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

cudaError_t reluFloat(const float* input, float* output,
                      size_t N, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    reluKernel<<<gridSize, blockSize, 0, stream>>>(input, output, N);
    return cudaGetLastError();
}

cudaError_t sigmoidFloat(const float* input, float* output,
                         size_t N, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    sigmoidKernel<<<gridSize, blockSize, 0, stream>>>(input, output, N);
    return cudaGetLastError();
}

// ============================================================================
// Softmax
// ============================================================================

__global__ void softmaxKernel(const float* input, float* output,
                               int batch_size, int num_classes) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float* input_row = input + batch_idx * num_classes;
    float* output_row = output + batch_idx * num_classes;

    // Find max for numerical stability
    float max_val = input_row[0];
    for (int i = 1; i < num_classes; i++) {
        max_val = fmaxf(max_val, input_row[i]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        float exp_val = expf(input_row[i] - max_val);
        output_row[i] = exp_val;
        atomicAdd(&sum, exp_val);
    }

    __syncthreads();

    // Normalize
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        output_row[i] /= sum;
    }
}

cudaError_t softmaxFloat(const float* input, float* output,
                         int batch_size, int num_classes,
                         cudaStream_t stream) {
    int blockSize = 256;
    softmaxKernel<<<batch_size, blockSize, 0, stream>>>(
        input, output, batch_size, num_classes);
    return cudaGetLastError();
}

// ============================================================================
// Batch Normalization
// ============================================================================

__global__ void batchNormKernel(const float* input, float* output,
                                const float* mean, const float* variance,
                                const float* gamma, const float* beta,
                                int batch_size, int channels, int spatial_size,
                                float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * spatial_size;

    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;

        float normalized = (input[idx] - mean[c]) / sqrtf(variance[c] + epsilon);
        output[idx] = gamma[c] * normalized + beta[c];
    }
}

cudaError_t batchNormFloat(const float* input, float* output,
                           const float* mean, const float* variance,
                           const float* gamma, const float* beta,
                           int batch_size, int channels, int spatial_size,
                           float epsilon, cudaStream_t stream) {
    int total_size = batch_size * channels * spatial_size;
    int blockSize = 256;
    int gridSize = (total_size + blockSize - 1) / blockSize;

    batchNormKernel<<<gridSize, blockSize, 0, stream>>>(
        input, output, mean, variance, gamma, beta,
        batch_size, channels, spatial_size, epsilon);

    return cudaGetLastError();
}

// ============================================================================
// Memory Operations
// ============================================================================

__global__ void memsetKernel(float* ptr, float value, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        ptr[idx] = value;
    }
}

cudaError_t memsetFloat(float* ptr, float value, size_t N,
                        cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    memsetKernel<<<gridSize, blockSize, 0, stream>>>(ptr, value, N);
    return cudaGetLastError();
}

// ============================================================================
// Reduction Operations
// ============================================================================

__global__ void reduceSumKernel(const float* input, float* output, size_t N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

cudaError_t reduceSum(const float* input, float* output,
                      size_t N, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Initialize output
    cudaMemsetAsync(output, 0, sizeof(float), stream);

    reduceSumKernel<<<gridSize, blockSize, 0, stream>>>(input, output, N);
    return cudaGetLastError();
}

__global__ void reduceMaxKernel(const float* input, float* output, size_t N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? input[idx] : -INFINITY;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((int*)output, __float_as_int(sdata[0]));
    }
}

cudaError_t reduceMax(const float* input, float* output,
                      size_t N, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Initialize output
    float init_val = -INFINITY;
    cudaMemcpyAsync(output, &init_val, sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    reduceMaxKernel<<<gridSize, blockSize, 0, stream>>>(input, output, N);
    return cudaGetLastError();
}

} // namespace kernels
} // namespace hook_analyzer
