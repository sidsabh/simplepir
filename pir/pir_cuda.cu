#include <cuda_runtime.h>
#include <cstdio>
#include "pir_cuda.h"

#ifndef TILE_COLS
#define TILE_COLS 1024 // columns processed per tile per block
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256 // threads per block (multiple of 32)
#endif

static constexpr int  BASIS        = 10;
static constexpr int  COMPRESSION  = 3;
static constexpr Elem MASK         = (1u << BASIS) - 1u;

#define CUDA_ASSERT(stmt) do { \
    cudaError_t err = (stmt);  \
    if (err != cudaSuccess) {  \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        abort();               \
    }                          \
} while (0)


// Persistent device buffers
static Elem  *d_a    = nullptr;   // [aRows * aCols]
static Elem  *d_b    = nullptr;   // [aCols * COMPRESSION]
static Elem  *d_out  = nullptr;   // [aRows]
static size_t d_aRows = 0, d_aCols = 0;

// Each block computes 1 output row (or more via grid-stride). Columns are processed in tiles.
// For each tile, the block cooperatively caches b into shared memory, then
// each thread walks its subset of columns for the target row.
__global__ void matMulVecPackedKernelTiled(Elem * __restrict__ out,
                                           const Elem * __restrict__ a,
                                           const Elem * __restrict__ b,
                                           size_t aRows, size_t aCols,
                                           size_t startRow, size_t numRows)
{
    const size_t rowLocal = blockIdx.x;
    if (rowLocal >= numRows) return;

    const size_t row = startRow + rowLocal;
    const size_t rowBase = row * aCols;

    extern __shared__ Elem s_b[];                      
    Elem* s_b0 = s_b + 0 * TILE_COLS;
    Elem* s_b1 = s_b + 1 * TILE_COLS;
    Elem* s_b2 = s_b + 2 * TILE_COLS;

    // Per-thread accumulator
    unsigned long long acc = 0ull; 

    // Process columns in tiles
    for (size_t tileBase = 0; tileBase < aCols; tileBase += TILE_COLS) {
        const int lane = threadIdx.x;
        const int blockThreads = blockDim.x;
        const size_t tileCols = min((size_t)TILE_COLS, aCols - tileBase);

        // Cooperatively load b tile into shared memory (coalesced)
        for (size_t c = lane; c < tileCols; c += blockThreads) {
            const size_t j = tileBase + c;
            s_b0[c] = __ldg(&b[j * COMPRESSION + 0]);
            s_b1[c] = __ldg(&b[j * COMPRESSION + 1]);
            s_b2[c] = __ldg(&b[j * COMPRESSION + 2]);
        }
        __syncthreads();

        // Now walk this tile’s columns for this row, striding by blockDim.x for coalesced loads
        // Threads read a[rowBase + j] at consecutive j -> coalesced
        for (size_t c = lane; c < tileCols; c += blockThreads) {
            const size_t j = tileBase + c;
            Elem db = __ldg(&a[rowBase + j]);

            // Extract packed BASIS-bit chunks and MAC with shared b
            Elem v0 = db & MASK;
            Elem v1 = (db >> BASIS) & MASK;
            Elem v2 = (db >> (2 * BASIS)) & MASK;

            // Multiply-accumulate in 64-bit
            acc += (unsigned long long)v0 * (unsigned long long)s_b0[c];
            acc += (unsigned long long)v1 * (unsigned long long)s_b1[c];
            acc += (unsigned long long)v2 * (unsigned long long)s_b2[c];
        }
        __syncthreads();
    }

    // Intra-block reduction to a single value per row (block)
    // Warp-level reduction first
    unsigned long long val = acc;
    // reduce within warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // One partial per warp -> shared memory
    __shared__ unsigned long long warpSums[BLOCK_SIZE / 32];
    const int warpId = threadIdx.x >> 5;
    const int lane   = threadIdx.x & 31;

    if (lane == 0) warpSums[warpId] = val;
    __syncthreads();

    // First warp finalizes
    if (warpId == 0) {
        unsigned long long sum = (lane < (BLOCK_SIZE / 32)) ? warpSums[lane] : 0ull;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            out[row] = (Elem)sum;
        }
    }
}

extern "C" void matMulVecPackedGPUInit(const Elem *a, size_t aRows, size_t aCols)
{
    if (d_a) CUDA_ASSERT(cudaFree(d_a));
    if (d_b) CUDA_ASSERT(cudaFree(d_b));
    if (d_out) CUDA_ASSERT(cudaFree(d_out));

    d_aRows = aRows;
    d_aCols = aCols;

    const size_t bytesA = aRows * aCols * sizeof(Elem);
    const size_t bytesB = aCols * COMPRESSION * sizeof(Elem);
    const size_t bytesO = aRows * sizeof(Elem);

    CUDA_ASSERT(cudaMalloc(&d_a, bytesA));
    CUDA_ASSERT(cudaMalloc(&d_b, bytesB));
    CUDA_ASSERT(cudaMalloc(&d_out, bytesO));

    CUDA_ASSERT(cudaMemcpy(d_a, a, bytesA, cudaMemcpyHostToDevice));

    CUDA_ASSERT(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
}

extern "C" void matMulVecPackedGPUCompute(Elem *out, const Elem *b)
{
    matMulVecPackedGPUComputeRange(out, b, 0, d_aRows);
}

extern "C" void matMulVecPackedGPUComputeRange(Elem *out, const Elem *b,
                                               size_t startRow, size_t numRows)
{
    if (!d_a || startRow >= d_aRows) return;
    if (startRow + numRows > d_aRows) numRows = d_aRows - startRow;

    const size_t bytesB = d_aCols * COMPRESSION * sizeof(Elem);
    CUDA_ASSERT(cudaMemcpy(d_b, b, bytesB, cudaMemcpyHostToDevice));

    const dim3 block(BLOCK_SIZE);
    const dim3 grid((unsigned)numRows);

    const size_t shmem = TILE_COLS * COMPRESSION * sizeof(Elem);

    matMulVecPackedKernelTiled<<<grid, block, shmem>>>(d_out, d_a, d_b,
                                                       d_aRows, d_aCols,
                                                       startRow, numRows);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    const size_t bytesOut = numRows * sizeof(Elem);
    CUDA_ASSERT(cudaMemcpy(out, d_out + startRow, bytesOut, cudaMemcpyDeviceToHost));
}

extern "C" void matMulVecPackedGPUFree()
{
    if (d_a)   { CUDA_ASSERT(cudaFree(d_a));   d_a = nullptr; }
    if (d_b)   { CUDA_ASSERT(cudaFree(d_b));   d_b = nullptr; }
    if (d_out) { CUDA_ASSERT(cudaFree(d_out)); d_out = nullptr; }
}
