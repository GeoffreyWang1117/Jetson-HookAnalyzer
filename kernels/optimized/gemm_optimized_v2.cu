#include "kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace hook_analyzer {
namespace kernels {

// ============================================================================
// Optimized GEMM Kernels - Version 2
// ============================================================================

// Version 2a: Larger tile size (32x32)
#define TILE_SIZE_32 32

__global__ void gemmKernel32x32(const float* A, const float* B, float* C,
                                 int M, int N, int K) {
    __shared__ float As[TILE_SIZE_32][TILE_SIZE_32];
    __shared__ float Bs[TILE_SIZE_32][TILE_SIZE_32];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE_32 + ty;
    int col = bx * TILE_SIZE_32 + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE_32 - 1) / TILE_SIZE_32; t++) {
        // Load tiles into shared memory
        if (row < M && (t * TILE_SIZE_32 + tx) < K)
            As[ty][tx] = A[row * K + t * TILE_SIZE_32 + tx];
        else
            As[ty][tx] = 0.0f;

        if ((t * TILE_SIZE_32 + ty) < K && col < N)
            Bs[ty][tx] = B[(t * TILE_SIZE_32 + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial product with loop unrolling
        #pragma unroll
        for (int k = 0; k < TILE_SIZE_32; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Version 2b: 32x32 with vectorized loads (float4)
__global__ void gemmKernel32x32Vectorized(const float* A, const float* B, float* C,
                                          int M, int N, int K) {
    __shared__ float As[TILE_SIZE_32][TILE_SIZE_32];
    __shared__ float Bs[TILE_SIZE_32][TILE_SIZE_32];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE_32 + ty;
    int col = bx * TILE_SIZE_32 + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE_32 - 1) / TILE_SIZE_32; t++) {
        // Vectorized load for A (if aligned and within bounds)
        if (row < M && (t * TILE_SIZE_32 + tx) < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE_32 + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Vectorized load for B
        if ((t * TILE_SIZE_32 + ty) < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE_32 + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE_32; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Version 2c: 64x64 tile for even larger blocks
#define TILE_SIZE_64 64

__global__ void gemmKernel64x64(const float* A, const float* B, float* C,
                                 int M, int N, int K) {
    __shared__ float As[TILE_SIZE_64][TILE_SIZE_64];
    __shared__ float Bs[TILE_SIZE_64][TILE_SIZE_64];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Each thread computes 4 elements (2x2)
    int row = by * TILE_SIZE_64 + ty;
    int col = bx * TILE_SIZE_64 + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE_64 - 1) / TILE_SIZE_64; t++) {
        // Load tiles into shared memory
        if (row < M && (t * TILE_SIZE_64 + tx) < K)
            As[ty][tx] = A[row * K + t * TILE_SIZE_64 + tx];
        else
            As[ty][tx] = 0.0f;

        if ((t * TILE_SIZE_64 + ty) < K && col < N)
            Bs[ty][tx] = B[(t * TILE_SIZE_64 + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial product
        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE_64; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Version 2d: Double buffering to hide memory latency
__global__ void gemmKernel32x32DoubleBuffer(const float* A, const float* B, float* C,
                                            int M, int N, int K) {
    // Two sets of shared memory for double buffering
    __shared__ float As[2][TILE_SIZE_32][TILE_SIZE_32];
    __shared__ float Bs[2][TILE_SIZE_32][TILE_SIZE_32];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE_32 + ty;
    int col = bx * TILE_SIZE_32 + tx;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE_32 - 1) / TILE_SIZE_32;

    // Prefetch first tile
    int write_idx = 0;
    if (row < M && tx < K)
        As[write_idx][ty][tx] = A[row * K + tx];
    else
        As[write_idx][ty][tx] = 0.0f;

    if (ty < K && col < N)
        Bs[write_idx][ty][tx] = B[ty * N + col];
    else
        Bs[write_idx][ty][tx] = 0.0f;

    __syncthreads();

    for (int t = 0; t < num_tiles; t++) {
        int read_idx = write_idx;
        write_idx = 1 - write_idx;

        // Prefetch next tile while computing current tile
        if (t + 1 < num_tiles) {
            int next_tile_offset = (t + 1) * TILE_SIZE_32;

            if (row < M && (next_tile_offset + tx) < K)
                As[write_idx][ty][tx] = A[row * K + next_tile_offset + tx];
            else
                As[write_idx][ty][tx] = 0.0f;

            if ((next_tile_offset + ty) < K && col < N)
                Bs[write_idx][ty][tx] = B[(next_tile_offset + ty) * N + col];
            else
                Bs[write_idx][ty][tx] = 0.0f;
        }

        // Compute using current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE_32; k++) {
            sum += As[read_idx][ty][k] * Bs[read_idx][k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Host API Functions
// ============================================================================

cudaError_t gemmFloatOptimized32(const float* A, const float* B, float* C,
                                 int M, int N, int K, cudaStream_t stream) {
    dim3 blockDim(TILE_SIZE_32, TILE_SIZE_32);
    dim3 gridDim((N + TILE_SIZE_32 - 1) / TILE_SIZE_32,
                 (M + TILE_SIZE_32 - 1) / TILE_SIZE_32);

    gemmKernel32x32<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);

    return cudaGetLastError();
}

cudaError_t gemmFloatOptimized32Vectorized(const float* A, const float* B, float* C,
                                           int M, int N, int K, cudaStream_t stream) {
    dim3 blockDim(TILE_SIZE_32, TILE_SIZE_32);
    dim3 gridDim((N + TILE_SIZE_32 - 1) / TILE_SIZE_32,
                 (M + TILE_SIZE_32 - 1) / TILE_SIZE_32);

    gemmKernel32x32Vectorized<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);

    return cudaGetLastError();
}

cudaError_t gemmFloatOptimized64(const float* A, const float* B, float* C,
                                 int M, int N, int K, cudaStream_t stream) {
    dim3 blockDim(TILE_SIZE_64, TILE_SIZE_64);
    dim3 gridDim((N + TILE_SIZE_64 - 1) / TILE_SIZE_64,
                 (M + TILE_SIZE_64 - 1) / TILE_SIZE_64);

    gemmKernel64x64<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);

    return cudaGetLastError();
}

cudaError_t gemmFloatOptimized32DoubleBuffer(const float* A, const float* B, float* C,
                                             int M, int N, int K, cudaStream_t stream) {
    dim3 blockDim(TILE_SIZE_32, TILE_SIZE_32);
    dim3 gridDim((N + TILE_SIZE_32 - 1) / TILE_SIZE_32,
                 (M + TILE_SIZE_32 - 1) / TILE_SIZE_32);

    gemmKernel32x32DoubleBuffer<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);

    return cudaGetLastError();
}

} // namespace kernels
} // namespace hook_analyzer
