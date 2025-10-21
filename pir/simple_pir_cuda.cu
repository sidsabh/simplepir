#include <cuda_runtime.h>
#include <cstdio>
#include "simple_pir_cuda.h"



#ifndef TILE_COLS
#define TILE_COLS 4096// columns processed per tile per block
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024 // threads per block (multiple of 32)
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


// super simple: matMulVecPackedKernel
// this kernel completely misuses CUDA as we don't use shared memory or coalesced loads
// tput: 8.5 GB/s
__global__ void matMulVecPackedKernel(Elem * __restrict__ out,
                                       const Elem * __restrict__ a,
                                       const Elem * __restrict__ b,
                                       size_t aRows, size_t aCols,
                                       size_t startRow, size_t numRows)
{
    const size_t rowLocal = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowLocal >= numRows) return;

    const size_t row = startRow + rowLocal;
    const size_t rowBase = row * aCols;

    unsigned long long acc = 0ull;

    for (size_t j = 0; j < aCols; ++j) {
        Elem db = __ldg(&a[rowBase + j]);

        Elem v0 = db & MASK;
        Elem v1 = (db >> BASIS) & MASK;
        Elem v2 = (db >> (2 * BASIS)) & MASK;

        acc += (unsigned long long)v0 * (unsigned long long)__ldg(&b[j * COMPRESSION + 0]);
        acc += (unsigned long long)v1 * (unsigned long long)__ldg(&b[j * COMPRESSION + 1]);
        acc += (unsigned long long)v2 * (unsigned long long)__ldg(&b[j * COMPRESSION + 2]);
    }

    out[row] = (Elem)acc;
}

// here, we cache b into shared memory per tile
// tput: 14 GB/s
__global__ void matMulVecPackedKernel_opt(Elem *__restrict__ out,
                                            const Elem *__restrict__ a,
                                            const Elem *__restrict__ b,
                                            size_t aRows, size_t aCols,
                                            size_t startRow, size_t numRows)
{
    const size_t rowLocal = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowLocal >= numRows) return;

    const size_t row = startRow + rowLocal;
    const size_t rowBase = row * aCols;

    // choose tile size so shared memory fits (≈ 48–96 KB)
    extern __shared__ Elem b_cache[]; // [TILE_COLS * COMPRESSION]


    unsigned long long acc = 0ull;

    // process b in chunks
    for (size_t jBase = 0; jBase < aCols; jBase += TILE_COLS) {
        size_t tileSize = min((size_t)TILE_COLS, aCols - jBase);
        size_t totalB = tileSize * COMPRESSION;

        // cooperative load of b into shared memory
        for (size_t i = threadIdx.x; i < totalB; i += blockDim.x) {
            b_cache[i] = __ldg(&b[jBase * COMPRESSION + i]);
        }
        __syncthreads(); // make sure all b_cache is ready

        // compute partial dot product for this tile
        for (size_t j = 0; j < tileSize; ++j) {
            Elem db = __ldg(&a[rowBase + jBase + j]);
            Elem v0 = db & MASK;
            Elem v1 = (db >> BASIS) & MASK;
            Elem v2 = (db >> (2 * BASIS)) & MASK;

            acc += (unsigned long long)v0 * (unsigned long long)b_cache[j * COMPRESSION + 0];
            acc += (unsigned long long)v1 * (unsigned long long)b_cache[j * COMPRESSION + 1];
            acc += (unsigned long long)v2 * (unsigned long long)b_cache[j * COMPRESSION + 2];
        }

        __syncthreads(); // reuse shared mem for next tile
    }

    out[row] = (Elem)acc;
}


// Each block computes 1 output row (or more via grid-stride). Columns are processed in tiles.
// If, for each tile, the block cooperatively loads b into shared memory, then we get near optimal performance! We don't do this because we want to progress from worst to best kernel.
// each thread walks its subset of columns for the target row.
// tput: 151 GB/s
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

    // Per-thread accumulator
    unsigned long long acc = 0ull; 

    // Process columns in tiles
    for (size_t tileBase = 0; tileBase < aCols; tileBase += TILE_COLS) {
        const int lane = threadIdx.x;
        const int blockThreads = blockDim.x;
        const size_t tileCols = min((size_t)TILE_COLS, aCols - tileBase);

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
            acc += (unsigned long long)v0 * (unsigned long long)__ldg(&b[j * COMPRESSION + 0]);
            acc += (unsigned long long)v1 * (unsigned long long)__ldg(&b[j * COMPRESSION + 1]);
            acc += (unsigned long long)v2 * (unsigned long long)__ldg(&b[j * COMPRESSION + 2]);
        }
        __syncthreads(); // reuse shared mem for next tile
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


// One warp per row, many rows per block.
// Reuses the b tile across all warps (rows) within a block.
// tput: 175 GB/s
__global__ void matMulVecPackedWarpSpanTile(Elem * __restrict__ out,
                                    const Elem * __restrict__ a,
                                    const Elem * __restrict__ b,
                                    size_t aRows, size_t aCols,
                                    size_t startRow, size_t numRows)
{
    // Expect blockDim.x to be a multiple of 32
    const int lane         = threadIdx.x & 31;       // 0..31
    const int warpId       = threadIdx.x >> 5;       // 0..(rowsPerBlock-1)
    const int rowsPerBlock = blockDim.x >> 5;        // warps per block

    // Global row index for this warp
    const size_t rowLocal = (size_t)blockIdx.x * (size_t)rowsPerBlock + (size_t)warpId;
    if (rowLocal >= numRows) return;

    const size_t row      = startRow + rowLocal;
    const size_t rowBase0 = row * aCols; // start of this row in A

    // Shared cache for the current tile of b (COMPRESSION = 3 streams)
    extern __shared__ Elem s_b[];
    Elem* s_b0 = s_b + 0 * TILE_COLS;
    Elem* s_b1 = s_b + 1 * TILE_COLS;
    Elem* s_b2 = s_b + 2 * TILE_COLS;

    unsigned long long acc = 0ull;

    // Process columns in tiles so b fits in shared memory
    for (size_t tileBase = 0; tileBase < aCols; tileBase += TILE_COLS) {
        const size_t tileCols = min((size_t)TILE_COLS, aCols - tileBase);

        // All threads in the block cooperatively load the b tile -> shared
        // Coalesced loads over c by threadIdx.x stride
        for (size_t c = threadIdx.x; c < tileCols; c += blockDim.x) {
            const size_t j = tileBase + c;
            s_b0[c] = __ldg(&b[j * COMPRESSION + 0]);
            s_b1[c] = __ldg(&b[j * COMPRESSION + 1]);
            s_b2[c] = __ldg(&b[j * COMPRESSION + 2]);
        }
        __syncthreads();

        // Each warp walks its own row over this shared b tile.
        // Within a warp, lanes stride by 32 to keep A loads coalesced per row.
        const size_t rowBase = rowBase0 + tileBase;
        for (size_t c = lane; c < tileCols; c += 32) {
            Elem db = __ldg(&a[rowBase + c]);

            // Unpack 3 BASIS-bit fields
            Elem v0 =  db                & MASK;
            Elem v1 = (db >> BASIS)      & MASK;
            Elem v2 = (db >> (2*BASIS))  & MASK;

            // MAC in 64-bit
            acc += (unsigned long long)v0 * (unsigned long long)s_b0[c];
            acc += (unsigned long long)v1 * (unsigned long long)s_b1[c];
            acc += (unsigned long long)v2 * (unsigned long long)s_b2[c];
        }

        __syncthreads(); // reuse shared mem for next tile
    }

    // Warp reduction to a single value for this row
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        // Truncation to Elem performs mod 2^w if Elem is u32
        out[row] = (Elem)acc;
    }
}

// K rows per warp, multiple warps per block; reuse b-tile across K rows
// tputs: 177 GB/s
template<int K>
__global__ void matMulVecPackedWarpSpanTileK(Elem * __restrict__ out,
                                             const Elem * __restrict__ a,
                                             const Elem * __restrict__ b,
                                             size_t aRows, size_t aCols,
                                             size_t startRow, size_t numRows)
{
    const int lane   = threadIdx.x & 31;      // 0..31
    const int warpId = threadIdx.x >> 5;      // warp idx within block
    const int warpsPerBlock = blockDim.x >> 5;

    // Base row for this warp's K-pack
    const size_t warpPackBaseLocal = (size_t)blockIdx.x * (size_t)(warpsPerBlock * K)
                                     + (size_t)warpId * (size_t)K;
    if (warpPackBaseLocal >= numRows) return;

    // Shared cache for current b tile (COMPRESSION=3)
    extern __shared__ Elem s_b[];
    Elem* s_b0 = s_b + 0 * TILE_COLS;
    Elem* s_b1 = s_b + 1 * TILE_COLS;
    Elem* s_b2 = s_b + 2 * TILE_COLS;

    // K per-thread accumulators
    unsigned long long acc[K];
    #pragma unroll
    for (int r=0; r<K; ++r) acc[r] = 0ull;

    // Walk columns in tiles (to fit b in shared)
    for (size_t tileBase = 0; tileBase < aCols; tileBase += TILE_COLS) {
        const size_t tileCols = min((size_t)TILE_COLS, aCols - tileBase);

        // Cooperative, coalesced load of b tile into shared
        for (size_t c = threadIdx.x; c < tileCols; c += blockDim.x) {
            const size_t j = tileBase + c;
            s_b0[c] = __ldg(&b[j * COMPRESSION + 0]);
            s_b1[c] = __ldg(&b[j * COMPRESSION + 1]);
            s_b2[c] = __ldg(&b[j * COMPRESSION + 2]);
        }
        __syncthreads();

        // Each warp processes K rows using this same b tile
        #pragma unroll
        for (int r=0; r<K; ++r) {
            const size_t rowLocal = warpPackBaseLocal + (size_t)r;
            if (rowLocal >= numRows) break; // tail warp-pack
            const size_t row       = startRow + rowLocal;
            const size_t rowBaseA  = (row * aCols) + tileBase;

            for (size_t c = lane; c < tileCols; c += 32) {
                Elem db = __ldg(&a[rowBaseA + c]);
                Elem v0 =  db               & MASK;
                Elem v1 = (db >> BASIS)     & MASK;
                Elem v2 = (db >> (2*BASIS)) & MASK;
                acc[r] += (unsigned long long)v0 * (unsigned long long)s_b0[c];
                acc[r] += (unsigned long long)v1 * (unsigned long long)s_b1[c];
                acc[r] += (unsigned long long)v2 * (unsigned long long)s_b2[c];
            }
        }
        __syncthreads(); // before next b tile
    }

    // Reduce and write K results (one per row)
    #pragma unroll
    for (int r=0; r<K; ++r) {
        const size_t rowLocal = warpPackBaseLocal + (size_t)r;
        if (rowLocal >= numRows) break;
        unsigned long long v = acc[r];
        #pragma unroll
        for (int off=16; off>0; off>>=1) v += __shfl_down_sync(0xffffffff, v, off);
        if (lane == 0) out[startRow + rowLocal] = (Elem)v;
    }
}


// Benchmark: measure raw A read bandwidth (no b, just coalesced loads)
// tput: 185 GB/s
__global__ void measureABandwidthKernel(const Elem *__restrict__ a,
                                        Elem *__restrict__ out,
                                        size_t aRows, size_t aCols)
{
    const int lane   = threadIdx.x & 31;   // lane within warp
    const int warpId = threadIdx.x >> 5;   // warp within block
    const int rowsPerBlock = blockDim.x >> 5; // warps per block

    const size_t rowLocal = (size_t)blockIdx.x * (size_t)rowsPerBlock + warpId;
    if (rowLocal >= aRows) return;

    const size_t rowBase = rowLocal * aCols;
    unsigned long long acc = 0ull;

    // Coalesced global reads
    for (size_t c = lane; c < aCols; c += 32) {
        Elem val = __ldg(&a[rowBase + c]);
        acc += val; // keep compiler from optimizing away
    }

    // Intra-warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    if (lane == 0)
        out[rowLocal] = (Elem)acc;
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

    if (false) {
        // simple
        const int threads = BLOCK_SIZE;
        const int blocks = (numRows + threads - 1) / threads;
        matMulVecPackedKernel<<<blocks, threads>>>(d_out, d_a, d_b, d_aRows, d_aCols, startRow, numRows);
    } else if (false) {
        // optimized with b cache
        dim3 block(BLOCK_SIZE);
        dim3 grid((numRows + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Compute required shared memory (for sanity check)
        constexpr size_t COMPRESSION = 3;
        constexpr size_t SHMEM_BYTES = TILE_COLS * COMPRESSION * sizeof(Elem);

        matMulVecPackedKernel_opt<<<grid, block, SHMEM_BYTES>>>(d_out, d_a, d_b, d_aRows, d_aCols, startRow, numRows);
    }
    else if (false) {
        // tiled
        const int threads = BLOCK_SIZE;
        const int blocks = numRows; // 1 row per block
        matMulVecPackedKernelTiled<<<blocks, threads>>>(d_out, d_a, d_b, d_aRows, d_aCols, startRow, numRows);
    } else if (false) { // span kernel: warp-per-row (reuses b tile across rows in a block)
        // Choose how many warps (rows) per block you want
        constexpr int rowsPerBlock = (BLOCK_SIZE / 32); // e.g., if BLOCK_SIZE=256, rowsPerBlock=8
        static_assert((BLOCK_SIZE % 32) == 0, "BLOCK_SIZE must be a multiple of 32");

        dim3 block(BLOCK_SIZE); // 32 * rowsPerBlock
        dim3 grid( (int)((numRows + rowsPerBlock - 1) / rowsPerBlock) );

        const size_t sharedBytes = TILE_COLS * COMPRESSION * sizeof(Elem);
        matMulVecPackedWarpSpanTile<<<grid, block, sharedBytes>>>(d_out, d_a, d_b,
                                                        d_aRows, d_aCols,
                                                        startRow, numRows);
    } else if (true) {
        // choose block size as before (e.g., 256, 512, or 1024)
        dim3 block(BLOCK_SIZE);
        const int warpsPerBlock = BLOCK_SIZE / 32;
        const int K = 4; // tuned rows per warp
        const size_t rowsPerBlockLogical = (size_t)warpsPerBlock * (size_t)K;
        dim3 grid( (int)((numRows + rowsPerBlockLogical - 1) / rowsPerBlockLogical) );

        size_t shmemBytes = (size_t)TILE_COLS * (size_t)COMPRESSION * sizeof(Elem);
        matMulVecPackedWarpSpanTileK<4><<<grid, block, shmemBytes>>>(
            d_out, d_a, d_b, d_aRows, d_aCols, startRow, numRows);
    }
    else { // measure raw A read bandwidth
        constexpr int rowsPerBlock = (BLOCK_SIZE / 32);

        dim3 block(BLOCK_SIZE);
        dim3 grid((int)((numRows + rowsPerBlock - 1) / rowsPerBlock));
        measureABandwidthKernel<<<grid, block>>>(d_a, d_out, d_aRows, d_aCols);
    }



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
