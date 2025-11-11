// double_pir_cuda.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "double_pir_cuda.h"

typedef uint32_t Elem;

#ifndef TILE_COLS
#define TILE_COLS 4096
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024 // must be multiple of 32
#endif

static constexpr int  COMPRESSION = 3;
static constexpr int  BASIS       = 10;
static constexpr Elem MASK        = (1u << BASIS) - 1u;

#define CUDA_ASSERT(stmt) do { \
    cudaError_t err = (stmt);  \
    if (err != cudaSuccess) {  \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        abort();               \
    }                          \
} while (0)

// ---------------- Persistent state (host mirror) ----------------
static Elem* DB = nullptr;  static size_t DB_rows = 0, DB_cols = 0;
static Elem* H1 = nullptr;  static size_t H1_rows = 0, H1_cols = 0;
static Elem* A2t = nullptr; static size_t A2t_rows = 0, A2t_cols = 0;
static unsigned int g_X = 1, g_delta = 1;

// ---------------- Persistent device buffers ----------------
static Elem *d_DB        = nullptr;
static Elem *d_H1        = nullptr;
static Elem *d_A2t       = nullptr;
static Elem *d_q1        = nullptr;
static Elem *d_q2        = nullptr;
static Elem *d_a1        = nullptr;
static Elem *d_a1_packed = nullptr;
static Elem *d_h1        = nullptr;
static Elem *d_a2        = nullptr;
static Elem *d_h2        = nullptr;

// ---------------- Utility ----------------
static inline uint32_t base_p_digit(uint64_t v, uint32_t f, uint32_t p) {
    for (uint32_t i = 0; i < f; ++i) v /= p;
    return (uint32_t)(v % p);
}

// ---------------- CPU reference helpers (for fallback) ----------------
static void matvec_cpu(Elem* out, const Elem* A, const Elem* b,
                       size_t rows, size_t cols)
{
    for (size_t i = 0; i < rows; ++i) {
        uint64_t acc = 0;
        for (size_t j = 0; j < cols; ++j) {
            Elem db = A[i * cols + j];
            Elem v0 = db & MASK;
            Elem v1 = (db >> BASIS) & MASK;
            Elem v2 = (db >> (2 * BASIS)) & MASK;
            acc += (uint64_t)v0 * b[j * 3 + 0];
            acc += (uint64_t)v1 * b[j * 3 + 1];
            acc += (uint64_t)v2 * b[j * 3 + 2];
        }
        out[i] = (Elem)acc;
    }
}

static void transform_cpu(const Elem* m, uint32_t R, uint32_t C,
                          uint32_t mod_p, uint32_t delta, uint32_t concat,
                          uint32_t basis, uint32_t d,
                          Elem* n, uint32_t nRows, uint32_t nCols)
{
    for (uint32_t outIdx = 0; outIdx < nRows * nCols; ++outIdx) {
        uint32_t r = outIdx / nCols;
        uint32_t cIdx = outIdx % nCols;

        uint32_t tmp = r / delta;
        uint32_t f = r % delta;
        uint32_t i = tmp % C;
        uint32_t jstripe = tmp / C;
        uint32_t stripe = jstripe % concat;

        uint64_t packed = 0ull;
        for (uint32_t t = 0; t < d; ++t) {
            uint32_t c = cIdx * d + t;
            if (c >= (R / concat)) break;
            uint32_t j = c * concat + stripe;
            if (j >= R) continue;

            uint64_t val = (uint64_t)m[j * C + i];
            for (uint32_t step = 0; step < f; ++step) val /= (uint64_t)mod_p;
            uint32_t digit = (uint32_t)(val % (uint64_t)mod_p);
            packed |= (uint64_t)digit << (basis * t);
        }
        n[r * nCols + cIdx] = (Elem)packed;
    }
}

static void gemm_cpu(Elem* out, const Elem* A, const Elem* B,
                     size_t A_r, size_t A_c, size_t B_r, size_t B_c)
{
    for (size_t i = 0; i < A_r; ++i) {
        for (size_t j = 0; j < B_r; ++j) {
            uint64_t acc = 0;
            for (size_t k = 0; k < A_c; ++k) {
                Elem a = A[i * A_c + k];
                Elem b0 = B[j * B_c + k * COMPRESSION + 0];
                Elem b1 = B[j * B_c + k * COMPRESSION + 1];
                Elem b2 = B[j * B_c + k * COMPRESSION + 2];
                Elem v0 = a & MASK;
                Elem v1 = (a >> BASIS) & MASK;
                Elem v2 = (a >> (2 * BASIS)) & MASK;
                acc += (uint64_t)v0 * b0 + (uint64_t)v1 * b1 + (uint64_t)v2 * b2;
            }
            out[i * B_r + j] = (Elem)acc;
        }
    }
}

// ---------------- GPU kernels ----------------

// Robust block-per-row tiled kernel for A×q (used for a1, a2, h2)
__global__ void matMulVecPackedKernelTiledD(Elem * __restrict__ out,
                                           const Elem * __restrict__ a,
                                           const Elem * __restrict__ b,
                                           size_t aRows, size_t aCols,
                                           size_t startRow, size_t numRows)
{
    const size_t rowLocal = blockIdx.x; // one block per row
    if (rowLocal >= numRows) return;

    const size_t row = startRow + rowLocal;
    const size_t rowBase = row * aCols;

    unsigned long long acc = 0ull;

    for (size_t tileBase = 0; tileBase < aCols; tileBase += TILE_COLS) {
        const int lane = threadIdx.x;
        const int blockThreads = blockDim.x;
        const size_t tileCols = min((size_t)TILE_COLS, aCols - tileBase);

        for (size_t c = lane; c < tileCols; c += blockThreads) {
            const size_t j = tileBase + c;
            Elem db = __ldg(&a[rowBase + j]);
            Elem v0 =  db               & MASK;
            Elem v1 = (db >> BASIS)     & MASK;
            Elem v2 = (db >> (2*BASIS)) & MASK;

            acc += (unsigned long long)v0 * (unsigned long long)__ldg(&b[j * COMPRESSION + 0]);
            acc += (unsigned long long)v1 * (unsigned long long)__ldg(&b[j * COMPRESSION + 1]);
            acc += (unsigned long long)v2 * (unsigned long long)__ldg(&b[j * COMPRESSION + 2]);
        }
        __syncthreads();
    }

    // reduce across threads in block
    unsigned long long val = acc;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    __shared__ unsigned long long warpSums[BLOCK_SIZE / 32];
    const int warpId = threadIdx.x >> 5;
    const int lane   = threadIdx.x & 31;
    if (lane == 0) warpSums[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        unsigned long long sum = (lane < (BLOCK_SIZE / 32)) ? warpSums[lane] : 0ull;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) out[row] = (Elem)sum;
    }
}

// Byte-for-byte equivalent of CPU transform
__global__ void transformKernel_safe(const Elem* __restrict__ m,
                                     uint32_t R, uint32_t C,
                                     uint32_t mod_p, uint32_t delta, uint32_t concat,
                                     uint32_t basis, uint32_t d,
                                     Elem* __restrict__ n,
                                     uint32_t nRows, uint32_t nCols)
{
    const uint32_t outIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total  = nRows * nCols;
    if (outIdx >= total) return;

    const uint32_t r    = outIdx / nCols;
    const uint32_t cIdx = outIdx % nCols;

    const uint32_t tmp     = r / delta;
    const uint32_t f       = r % delta;
    const uint32_t i       = tmp % C;
    const uint32_t jstripe = tmp / C;
    const uint32_t stripe  = jstripe % concat;

    uint64_t packed = 0ull;

    for (uint32_t t = 0; t < d; ++t) {
        uint32_t c = cIdx * d + t;
        if (c >= (R / concat)) break;

        uint32_t j = c * concat + stripe;
        if (j >= R) continue;

        uint64_t val = (uint64_t)m[(size_t)j * (size_t)C + (size_t)i];

        for (uint32_t step = 0; step < f; ++step) {
            val /= (uint64_t)mod_p;
        }
        uint32_t digit = (uint32_t)(val % (uint64_t)mod_p);

        packed |= (uint64_t)digit << (basis * t);
    }

    n[(size_t)r * (size_t)nCols + (size_t)cIdx] = (Elem)packed;
}

// C = A × Bᵗ; identical indexing/loop order to CPU gemm
__global__ void gemmPackedKernel_rowmajor(Elem* __restrict__ out,
                                          const Elem* __restrict__ A,
                                          const Elem* __restrict__ B,
                                          uint32_t nRows, uint32_t nCols,
                                          uint32_t B_r,   uint32_t B_c)
{
    const uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; // row in A/out
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; // row in B (col of out)
    if (i >= nRows || j >= B_r) return;

    uint64_t acc = 0ull;
    for (uint32_t k = 0; k < nCols; ++k) {
        Elem a  = __ldg(&A[(size_t)i * (size_t)nCols + (size_t)k]);

        Elem b0 = __ldg(&B[(size_t)j * (size_t)B_c + (size_t)k * COMPRESSION + 0]);
        Elem b1 = __ldg(&B[(size_t)j * (size_t)B_c + (size_t)k * COMPRESSION + 1]);
        Elem b2 = __ldg(&B[(size_t)j * (size_t)B_c + (size_t)k * COMPRESSION + 2]);

        Elem v0 =  a               & MASK;
        Elem v1 = (a >> BASIS)     & MASK;
        Elem v2 = (a >> (2*BASIS)) & MASK;

        acc += (uint64_t)v0 * b0 + (uint64_t)v1 * b1 + (uint64_t)v2 * b2;
    }
    out[(size_t)i * (size_t)B_r + (size_t)j] = (Elem)acc;
}

// ---------------- API ----------------

extern "C" void doublePIRGPUInit(const Elem* DB_in, size_t DB_r, size_t DB_c,
                                 const Elem* H1_in, size_t H1_r, size_t H1_c,
                                 const Elem* A2t_in, size_t A2t_r, size_t A2t_c,
                                 unsigned int X, unsigned int delta)
{
    DB_rows = DB_r; DB_cols = DB_c;
    H1_rows = H1_r; H1_cols = H1_c;
    A2t_rows = A2t_r; A2t_cols = A2t_c;
    g_X = X; g_delta = delta;

    // host mirrors (optional)
    DB  = (Elem*)malloc(DB_r * DB_c * sizeof(Elem));
    H1  = (Elem*)malloc(H1_r * H1_c * sizeof(Elem));
    A2t = (Elem*)malloc(A2t_r * A2t_c * sizeof(Elem));
    memcpy(DB,  DB_in,  DB_r * DB_c * sizeof(Elem));
    memcpy(H1,  H1_in,  H1_r * H1_c * sizeof(Elem));
    memcpy(A2t, A2t_in, A2t_r * A2t_c * sizeof(Elem));

    // device buffers
    CUDA_ASSERT(cudaMalloc(&d_DB,  DB_r * DB_c * sizeof(Elem)));
    CUDA_ASSERT(cudaMalloc(&d_H1,  H1_r * H1_c * sizeof(Elem)));
    CUDA_ASSERT(cudaMalloc(&d_A2t, A2t_r * A2t_c * sizeof(Elem)));
    CUDA_ASSERT(cudaMemcpy(d_DB,  DB_in,  DB_r * DB_c * sizeof(Elem),  cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_H1,  H1_in,  H1_r * H1_c * sizeof(Elem),  cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_A2t, A2t_in, A2t_r * A2t_c * sizeof(Elem), cudaMemcpyHostToDevice));

    // temps (max sizes; we’ll reuse)
    CUDA_ASSERT(cudaMalloc(&d_q1,        DB_cols * COMPRESSION * sizeof(Elem)));
    CUDA_ASSERT(cudaMalloc(&d_q2,        max((size_t)H1_cols, (size_t)DB_cols) * COMPRESSION * sizeof(Elem)));
    CUDA_ASSERT(cudaMalloc(&d_a1,        DB_rows * sizeof(Elem)));
    // a1_packed dims depend on R,C,delta,X,d
    uint32_t R = (uint32_t)DB_rows, C = 1, d = 3;
    uint32_t nRows = C * g_delta * g_X;
    uint32_t Cmax  = R / g_X;
    uint32_t nCols = (Cmax + d - 1) / d;
    CUDA_ASSERT(cudaMalloc(&d_a1_packed, (size_t)nRows * (size_t)nCols * sizeof(Elem)));
    CUDA_ASSERT(cudaMalloc(&d_h1,        (size_t)nRows * (size_t)A2t_r * sizeof(Elem)));
    CUDA_ASSERT(cudaMalloc(&d_a2,        H1_rows * sizeof(Elem)));
    CUDA_ASSERT(cudaMalloc(&d_h2,        (size_t)nRows * sizeof(Elem)));

    CUDA_ASSERT(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    fprintf(stderr, "[GPU-ALL] Initialized device buffers\n");
}

extern "C" void doublePIRGPUAnswerFull(const Elem* q1s, size_t q1_len, int numQ1,
                                       const Elem* q2s, size_t q2_len, int numQ2,
                                       unsigned int mod_p,
                                       Elem* h1_out, size_t h1_rows, size_t h1_cols,
                                       Elem* a2_out_all, size_t H1_rows_out,
                                       Elem* h2_out_all, size_t h2_vec_len)
{
    (void)h1_rows; (void)h1_cols; (void)H1_rows_out; (void)h2_vec_len;
    fprintf(stderr, "[GPU-ALL] Executing DoublePIR accelerated path\n");

    // ---- Step 1: a1 = DB * q1 (batched over rows) ----
    size_t totalRows = DB_rows;
    size_t totalCols = DB_cols;
    if (numQ1 <= 0) numQ1 = 1;
    size_t batchSz = totalRows / (size_t)numQ1;
    size_t lastRow = 0;

    for (int batch = 0; batch < numQ1; ++batch) {
        if ((size_t)batch == (size_t)(numQ1 - 1))
            batchSz = totalRows - lastRow;

        const Elem* q1 = q1s + (size_t)batch * q1_len;

        // upload q1
        CUDA_ASSERT(cudaMemcpy(d_q1, q1, totalCols * COMPRESSION * sizeof(Elem), cudaMemcpyHostToDevice));

        // tiled robust kernel (one block per output row)
        dim3 block(BLOCK_SIZE);
        dim3 grid((int)batchSz);
        matMulVecPackedKernelTiledD<<<grid, block>>>(d_a1, d_DB, d_q1, DB_rows, DB_cols, lastRow, batchSz);
        CUDA_ASSERT(cudaGetLastError());

        lastRow += batchSz;
    }

    // ---- Step 2: transform a1 → a1_packed ----
    {
        uint32_t R = (uint32_t)DB_rows;
        uint32_t C = 1;
        uint32_t d = 3;
        uint32_t basis = 10;
        uint32_t nRows = C * g_delta * g_X;
        uint32_t Cmax = R / g_X;
        uint32_t nCols = (Cmax + d - 1) / d;

        // transform on GPU
        const size_t total = (size_t)nRows * (size_t)nCols;
        const int threads = 256;
        const int blocks = (int)((total + threads - 1) / threads);
        transformKernel_safe<<<blocks, threads>>>(d_a1, R, C, mod_p, g_delta, g_X, basis, d,
                                                  d_a1_packed, nRows, nCols);
        CUDA_ASSERT(cudaGetLastError());
    }

    // ---- Step 3: h1 = a1_packed × A2ᵗ ----
    {
        uint32_t R = (uint32_t)DB_rows;
        uint32_t C = 1;
        uint32_t d = 3;
        uint32_t nRows = C * g_delta * g_X;
        uint32_t Cmax = R / g_X;
        uint32_t nCols = (Cmax + d - 1) / d;

        dim3 block(32, 8);
        dim3 grid( (unsigned)((A2t_rows + block.x - 1) / block.x),
                   (unsigned)((nRows     + block.y - 1) / block.y) );
        gemmPackedKernel_rowmajor<<<grid, block>>>(d_h1, d_a1_packed, d_A2t,
                                                   nRows, nCols,
                                                   (uint32_t)A2t_rows, (uint32_t)A2t_cols);
        CUDA_ASSERT(cudaGetLastError());

        CUDA_ASSERT(cudaMemcpy(h1_out, d_h1, (size_t)nRows * (size_t)A2t_rows * sizeof(Elem),
                               cudaMemcpyDeviceToHost));
    }

    // ---- Step 4: for each q2 → compute a2 = H1*q2, h2 = a1_packed*q2 ----
    {
        uint32_t R = (uint32_t)DB_rows;
        uint32_t C = 1;
        uint32_t d = 3;
        uint32_t nRows = C * g_delta * g_X;
        uint32_t Cmax = R / g_X;
        uint32_t nCols = (Cmax + d - 1) / d;

        for (int k = 0; k < numQ2; ++k) {
            const Elem* q2 = q2s + (size_t)k * q2_len;
            CUDA_ASSERT(cudaMemcpy(d_q2, q2,
                                   max((size_t)H1_cols, (size_t)DB_cols) * COMPRESSION * sizeof(Elem),
                                   cudaMemcpyHostToDevice));

            // a2 = H1 * q2
            {
                dim3 block(BLOCK_SIZE);
                dim3 grid((int)H1_rows);
                matMulVecPackedKernelTiledD<<<grid, block>>>(d_a2, d_H1, d_q2, H1_rows, H1_cols, 0, H1_rows);
                CUDA_ASSERT(cudaGetLastError());
                CUDA_ASSERT(cudaMemcpy(a2_out_all + (size_t)k * H1_rows, d_a2,
                                       H1_rows * sizeof(Elem), cudaMemcpyDeviceToHost));
            }

            // h2 = a1_packed * q2  (A is [nRows x nCols])
            {
                dim3 block(BLOCK_SIZE);
                dim3 grid((int)nRows);
                matMulVecPackedKernelTiledD<<<grid, block>>>(d_h2, d_a1_packed, d_q2, nRows, nCols, 0, nRows);
                CUDA_ASSERT(cudaGetLastError());
                CUDA_ASSERT(cudaMemcpy(h2_out_all + (size_t)k * nRows, d_h2,
                                       nRows * sizeof(Elem), cudaMemcpyDeviceToHost));
            }
        }
    }

    fprintf(stderr, "[GPU-ALL] Completed DoublePIR accelerated path\n");
}

extern "C" void doublePIRGPUFree(void)
{
    if (DB)  { free(DB);  DB=nullptr; }
    if (H1)  { free(H1);  H1=nullptr; }
    if (A2t) { free(A2t); A2t=nullptr; }

    if (d_DB)        { CUDA_ASSERT(cudaFree(d_DB));        d_DB=nullptr; }
    if (d_H1)        { CUDA_ASSERT(cudaFree(d_H1));        d_H1=nullptr; }
    if (d_A2t)       { CUDA_ASSERT(cudaFree(d_A2t));       d_A2t=nullptr; }
    if (d_q1)        { CUDA_ASSERT(cudaFree(d_q1));        d_q1=nullptr; }
    if (d_q2)        { CUDA_ASSERT(cudaFree(d_q2));        d_q2=nullptr; }
    if (d_a1)        { CUDA_ASSERT(cudaFree(d_a1));        d_a1=nullptr; }
    if (d_a1_packed) { CUDA_ASSERT(cudaFree(d_a1_packed)); d_a1_packed=nullptr; }
    if (d_h1)        { CUDA_ASSERT(cudaFree(d_h1));        d_h1=nullptr; }
    if (d_a2)        { CUDA_ASSERT(cudaFree(d_a2));        d_a2=nullptr; }
    if (d_h2)        { CUDA_ASSERT(cudaFree(d_h2));        d_h2=nullptr; }

    fprintf(stderr, "[GPU-ALL] Freed device buffers\n");
}
