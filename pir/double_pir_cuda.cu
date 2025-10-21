// double_pir_cuda.cu
#include <cuda_runtime.h>
#include <cstdio>
#include "double_pir_cuda.h"

#ifndef TILE_COLS
#define TILE_COLS 4096
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
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

// ------------------ Persistent device buffers ------------------
static Elem* d_DB   = nullptr;  static size_t DB_rows=0,  DB_cols=0;
static Elem* d_H1   = nullptr;  static size_t H1_rows=0,  H1_cols=0;
static Elem* d_A2t  = nullptr;  static size_t A2t_rows=0, A2t_cols=0;

static Elem* d_q1   = nullptr;  static size_t cap_q1 = 0;
static Elem* d_q2   = nullptr;  static size_t cap_q2 = 0;

static unsigned int g_X = 1;
static unsigned int g_delta = 1;

// scratch
static unsigned long long* d_acc64 = nullptr;      // max(rows) accumulator
static size_t              cap_acc64 = 0;
static Elem*               d_vec_out = nullptr;    // staging for a single vec result
static size_t              cap_vec_out = 0;
static Elem* h_tmp = nullptr;
static size_t cap_h_tmp = 0;

static inline void ensure_host(size_t need) {
    if (cap_h_tmp >= need) return;
    if (h_tmp) CUDA_ASSERT(cudaFreeHost(h_tmp));
    // pinned host buffer for fast D2H
    CUDA_ASSERT(cudaMallocHost((void**)&h_tmp, need * sizeof(Elem)));
    cap_h_tmp = need;
}

// ------------------ Kernels ------------------


// ---- Fused packed GEMM: h1 = a1_packed [nRows x nCols]  ×  A2ᵗ [h1_cols x (nCols*COMPRESSION)] ----
// We tile K (packed columns) by TILE_K and output columns by TILE_N.
// Shared memory holds TILE_N * COMPRESSION * TILE_K elements of A2ᵗ's packed limbs.

#ifndef H1_TILE_K
#define H1_TILE_K 256 // fits in smem with TILE_N=4 (4*3*1024*4B = 48KB)
#endif
#ifndef H1_TILE_N
#define H1_TILE_N 2      // number of output columns processed at once
#endif

template<int KROWS>
__global__ void packed_gemm_all_cols(
    Elem* __restrict__ H,            // out: [nRows x h1_cols] row-major
    const Elem* __restrict__ A,      // a1_packed: [nRows x nCols] (packed elems)
    const Elem* __restrict__ Bp,     // A2ᵗ in packed layout: [h1_cols x (nCols*COMPRESSION)]
    size_t nRows, size_t nColsPacked, size_t h1_cols, size_t Bp_colsPackedTimes3)
{
    // thread tiling (same warp-span KROWS as your fast matvec)
    const int lane          = threadIdx.x & 31;
    const int warpId        = threadIdx.x >> 5;
    const int warpsPerBlock = blockDim.x >> 5;

    // grid.y selects a strip of output columns
    const size_t col0 = (size_t)blockIdx.y * H1_TILE_N;

    // base output row for this warp-pack
    const size_t rowPackBase = (size_t)blockIdx.x * (size_t)(warpsPerBlock * KROWS)
                             + (size_t)warpId * (size_t)KROWS;
    if (rowPackBase >= nRows || col0 >= h1_cols) return;

    // smem for TILE_N columns, each with 3 limbs, for TILE_K K-slice
    extern __shared__ Elem sb[];
    // layout: [H1_TILE_N][COMPRESSION][H1_TILE_K]
    auto sb_col = [&](int n, int limb, int k)->Elem& {
        return sb[(n*COMPRESSION + limb)*H1_TILE_K + k];
    };

    // accumulators per output column in this TILE_N
    unsigned long long acc[KROWS][H1_TILE_N];
    #pragma unroll
    for (int r=0; r<KROWS; ++r)
        #pragma unroll
        for (int n=0; n<H1_TILE_N; ++n)
            acc[r][n] = 0ull;

    // clamp packed width to what B actually has
    const size_t packedWidthB = Bp_colsPackedTimes3 / COMPRESSION; // (# packed cols B provides)
    const size_t Klim = min(nColsPacked, packedWidthB);

    // walk K in tiles
    for (size_t kBase = 0; kBase < Klim; kBase += H1_TILE_K) {
        const size_t kTile = min((size_t)H1_TILE_K, Klim - kBase);

        // cooperative load of Bp tile (limbs) for up to H1_TILE_N columns
        for (size_t t = threadIdx.x; t < (size_t)H1_TILE_N * COMPRESSION * kTile; t += blockDim.x) {
            const size_t k   = t % kTile;
            const size_t tmp = t / kTile;
            const int limb   = tmp % COMPRESSION;         // 0..2
            const int n      = tmp / COMPRESSION;         // 0..H1_TILE_N-1
            const size_t col = col0 + n;
            if (col < h1_cols) {
                sb_col(n, limb, k) = __ldg(&Bp[col * Bp_colsPackedTimes3 + (kBase + k)*COMPRESSION + limb]);
            }
        }
        __syncthreads();

        // compute KROWS rows per warp using this B tile
        #pragma unroll
        for (int r=0; r<KROWS; ++r) {
            const size_t row = rowPackBase + (size_t)r;
            if (row >= nRows) break;
            const size_t rowBaseA = row * nColsPacked + kBase;

            for (size_t kc = lane; kc < kTile; kc += 32) {
                const Elem aPacked = __ldg(&A[rowBaseA + kc]);
                const unsigned long long v0 = (unsigned long long)( aPacked               & ((Elem(1)<<BASIS)-1) );
                const unsigned long long v1 = (unsigned long long)((aPacked >> BASIS)     & ((Elem(1)<<BASIS)-1) );
                const unsigned long long v2 = (unsigned long long)((aPacked >> (2*BASIS)) & ((Elem(1)<<BASIS)-1) );

                #pragma unroll
                for (int n=0; n<H1_TILE_N; ++n) {
                    const size_t col = col0 + n;
                    if (col >= h1_cols) break;
                    acc[r][n] += v0 * (unsigned long long)sb_col(n, 0, kc);
                    acc[r][n] += v1 * (unsigned long long)sb_col(n, 1, kc);
                    acc[r][n] += v2 * (unsigned long long)sb_col(n, 2, kc);
                }
            }
        }
        __syncthreads();
    }

    // warp reductions and stores
    #pragma unroll
    for (int r=0; r<KROWS; ++r) {
        const size_t row = rowPackBase + (size_t)r;
        if (row >= nRows) break;

        #pragma unroll
        for (int n=0; n<H1_TILE_N; ++n) {
            unsigned long long v = acc[r][n];
            #pragma unroll
            for (int off=16; off>0; off>>=1) v += __shfl_down_sync(0xffffffff, v, off);
            if (lane == 0) {
                const size_t col = col0 + n;
                if (col < h1_cols) H[row * h1_cols + col] = (Elem)v; // mod 2^w via truncation
            }
        }
    }
}


// Warp-span, shared-b tile, K rows/warp (same style as SimplePIR)
template<int K>
__global__ void matMulVecPackedWarpSpanTileK(Elem * __restrict__ out,
                                             const Elem * __restrict__ a,
                                             const Elem * __restrict__ b,
                                             size_t aRows, size_t aCols,
                                             size_t startRow, size_t numRows)
{
    const int lane   = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;
    const int warpsPerBlock = blockDim.x >> 5;

    const size_t warpPackBaseLocal =
        (size_t)blockIdx.x * (size_t)(warpsPerBlock * K) + (size_t)warpId * (size_t)K;
    if (warpPackBaseLocal >= numRows) return;

    extern __shared__ Elem s_b[];
    Elem* s_b0 = s_b + 0 * TILE_COLS;
    Elem* s_b1 = s_b + 1 * TILE_COLS;
    Elem* s_b2 = s_b + 2 * TILE_COLS;

    unsigned long long acc[K];
    #pragma unroll
    for (int r=0; r<K; ++r) acc[r] = 0ull;

    for (size_t tileBase = 0; tileBase < aCols; tileBase += TILE_COLS) {
        const size_t tileCols = min((size_t)TILE_COLS, aCols - tileBase);

        // cooperative b tile load
        for (size_t c = threadIdx.x; c < tileCols; c += blockDim.x) {
            const size_t j = tileBase + c;
            s_b0[c] = __ldg(&b[j * COMPRESSION + 0]);
            s_b1[c] = __ldg(&b[j * COMPRESSION + 1]);
            s_b2[c] = __ldg(&b[j * COMPRESSION + 2]);
        }
        __syncthreads();

        #pragma unroll
        for (int r=0; r<K; ++r) {
            const size_t rowLocal = warpPackBaseLocal + (size_t)r;
            if (rowLocal >= numRows) break;
            const size_t row      = startRow + rowLocal;
            const size_t rowBaseA = row * aCols + tileBase;

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
        __syncthreads();
    }

    // reduce & write
    #pragma unroll
    for (int r=0; r<K; ++r) {
        const size_t rowLocal = warpPackBaseLocal + (size_t)r;
        if (rowLocal >= numRows) break;
        unsigned long long v = acc[r];
        #pragma unroll
        for (int off=16; off>0; off>>=1) v += __shfl_down_sync(0xffffffff, v, off);
        if ((threadIdx.x & 31) == 0) out[startRow + rowLocal] = (Elem)v; // truncation == mod 2^w
    }
}

// Extract f-th base-p digit from v (small delta => loop is fine).
__device__ __forceinline__ uint32_t base_p_digit(uint64_t v, uint32_t f, uint32_t p) {
    for (uint32_t i=0;i<f;++i) v /= p;
    return (uint32_t)(v % p);
}

// This fn takes matrix m (rows=R, cols=C) and produces n by:
// 1) Transpose
// 2) Expand cols by delta (base-p digits)
// 3) Concat every "concat" cols together
// 4) Squish every "d" cols into one by packing base-p digits in base-(p^d)
// Input m: rows=R, cols=C
// Output n: rows=C*delta*concat, cols=(R/concat + d- 1)/d
__global__ void transpose_expand_concat_squish_kernel(
    const Elem* __restrict__ m,  // size R*C (row-major)
    uint32_t R, uint32_t C,
    uint32_t mod_p, uint32_t delta, uint32_t concat, uint32_t basis, uint32_t d,
    Elem* __restrict__ n,        // out rows = C*delta*concat, cols = (R/concat + d-1)/d
    uint32_t nRows, uint32_t nCols)
{
    const uint32_t outLinear = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t totalOut  = nRows * nCols;
    if (outLinear >= totalOut) return;

    const uint32_t r = outLinear / nCols;       // 0..(C*delta*concat-1)
    const uint32_t cIdx = outLinear % nCols;    // 0..nCols-1

    const uint32_t tmp = r / delta;
    const uint32_t f   = r % delta;
    const uint32_t i   = tmp % C;
    const uint32_t jstripe = tmp / C;
    const uint32_t stripe  = jstripe % concat;

    uint64_t packed = 0ull;
    for (uint32_t t = 0; t < d; ++t) {
        const uint32_t c = cIdx * d + t;
        if (c >= (R / concat)) break;
        const uint32_t j = c * concat + stripe;
        if (j >= R) continue;

        const uint64_t val = (uint64_t)m[j * C + i];
        const uint32_t digit = base_p_digit(val, f, mod_p);
        packed |= (uint64_t)digit << (basis * t);
    }

    n[r * nCols + cIdx] = (Elem)packed;
}

// ------------------ Helpers ------------------

static inline void ensure_buf(Elem** p, size_t* cap, size_t need) {
    if (*cap >= need) return;
    if (*p) CUDA_ASSERT(cudaFree(*p));
    CUDA_ASSERT(cudaMalloc(p, need * sizeof(Elem)));
    *cap = need;
}

static inline void ensure_acc(size_t need64) {
    if (cap_acc64 >= need64) return;
    if (d_acc64) CUDA_ASSERT(cudaFree(d_acc64));
    CUDA_ASSERT(cudaMalloc(&d_acc64, need64 * sizeof(unsigned long long)));
    cap_acc64 = need64;
}

static inline void ensure_vec(size_t need) {
    if (cap_vec_out >= need) return;
    if (d_vec_out) CUDA_ASSERT(cudaFree(d_vec_out));
    CUDA_ASSERT(cudaMalloc(&d_vec_out, need * sizeof(Elem)));
    cap_vec_out = need;
}

// ------------------ Public API ------------------

extern "C" void doublePIRGPUInit(const Elem* DB, size_t DB_r, size_t DB_c,
                                  const Elem* H1, size_t H1_r, size_t H1_c,
                                  const Elem* A2t, size_t A2t_r, size_t A2t_c,
                                  unsigned int X, unsigned int delta)
{
    DB_rows=DB_r; DB_cols=DB_c; H1_rows=H1_r; H1_cols=H1_c; A2t_rows=A2t_r; A2t_cols=A2t_c;
    g_X = X; g_delta = delta;

    CUDA_ASSERT(cudaMalloc(&d_DB,  DB_r*DB_c*sizeof(Elem)));
    CUDA_ASSERT(cudaMalloc(&d_H1,  H1_r*H1_c*sizeof(Elem)));
    CUDA_ASSERT(cudaMalloc(&d_A2t, A2t_r*A2t_c*sizeof(Elem)));

    CUDA_ASSERT(cudaMemcpy(d_DB,  DB,  DB_r*DB_c*sizeof(Elem),  cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_H1,  H1,  H1_r*H1_c*sizeof(Elem),  cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_A2t, A2t, A2t_r*A2t_c*sizeof(Elem),cudaMemcpyHostToDevice));
}


extern "C" void doublePIRGPUAnswerRange(const Elem* q1, size_t q1_len,
                                        const Elem* q2s, size_t q2_len, int numQ2,
                                        size_t startRow, size_t numRows,
                                        unsigned int mod_p,
                                        Elem* h1_out, size_t h1_rows, size_t h1_cols,
                                        Elem* a2_out_all, size_t a2_vec_len,
                                        Elem* h2_out_all, size_t h2_vec_len)
{
    // 1) upload q1
    ensure_buf(&d_q1, &cap_q1, q1_len);
    CUDA_ASSERT(cudaMemcpy(d_q1, q1, q1_len*sizeof(Elem), cudaMemcpyHostToDevice));

    // 2) a1 := (DB[start:start+numRows] · q1)
    ensure_acc(numRows);
    {
        dim3 block(BLOCK_SIZE);
        const int warpsPerBlock = BLOCK_SIZE / 32;
        const int K = 4;
        const size_t rowsPerBlockLogical = (size_t)warpsPerBlock * (size_t)K;
        dim3 grid( (int)((numRows + rowsPerBlockLogical - 1) / rowsPerBlockLogical) );
        const size_t shmemBytes = (size_t)TILE_COLS * (size_t)COMPRESSION * sizeof(Elem);

        matMulVecPackedWarpSpanTileK<4><<<grid, block, shmemBytes>>>(
            (Elem*)d_acc64,
            d_DB, d_q1, DB_rows, DB_cols, startRow, numRows);
    }
    CUDA_ASSERT(cudaGetLastError());

    // 3) a1':= TransposeAndExpandAndConcatColsAndSquish(a1)
    //    m.Rows = numRows, m.Cols = 1
    //    n.Rows = (1 * delta * X), n.Cols = (numRows/X + d - 1)/d, where d=3, basis=10
    const uint32_t R = (uint32_t)numRows;
    const uint32_t C = 1u;
    const uint32_t d  = 3u;
    const uint32_t basis = 10u;
    const uint32_t nRows = C * g_delta * g_X;
    const uint32_t Cmax = R / g_X;
    const uint32_t nCols = (Cmax + d - 1) / d;

    Elem* d_a1_packed = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_a1_packed, (size_t)nRows * (size_t)nCols * sizeof(Elem)));

    {
        const uint32_t totalOut = nRows * nCols;
        const int threads = 256;
        const int blocks = (int)((totalOut + threads - 1) / threads);
        transpose_expand_concat_squish_kernel<<<blocks, threads>>>(
            (Elem*)d_acc64, R, C,
            mod_p, g_delta, g_X, basis, d,
            d_a1_packed, nRows, nCols);
    }
    CUDA_ASSERT(cudaGetLastError());

    // --- Step 4 (fused): h1 = a1_packed × A2ᵗ in ONE kernel ---
    {
        const size_t packedWidthA = nCols;
        const size_t packedWidthB = A2t_cols / COMPRESSION;
        const size_t aCols_effective = (packedWidthB < packedWidthA) ? packedWidthB : packedWidthA;

        // device output
        Elem* d_h1 = nullptr;
        CUDA_ASSERT(cudaMalloc(&d_h1, nRows * h1_cols * sizeof(Elem)));

        dim3 block(BLOCK_SIZE);                                      // e.g., 1024 (32*32)
        const int warpsPerBlock = BLOCK_SIZE / 32;
        const int KROWS = 4;
        const size_t rowsPerBlockLogical = (size_t)warpsPerBlock * (size_t)KROWS;

        dim3 grid;
        grid.x = (unsigned)((nRows + rowsPerBlockLogical - 1) / rowsPerBlockLogical);
        grid.y = (unsigned)((h1_cols + H1_TILE_N - 1) / H1_TILE_N);

        const size_t shmemBytes = (size_t)H1_TILE_N * (size_t)COMPRESSION * (size_t)H1_TILE_K * sizeof(Elem);

        packed_gemm_all_cols<KROWS><<<grid, block, shmemBytes>>>(
            d_h1,                      // out
            d_a1_packed,               // A: [nRows x aCols_effective]
            d_A2t,                     // Bᵗ packed rows
            nRows, aCols_effective, h1_cols, A2t_cols
        );
        CUDA_ASSERT(cudaGetLastError());
        CUDA_ASSERT(cudaDeviceSynchronize());

        // D2H once for the whole matrix, exactly row-major shape expected by caller
        CUDA_ASSERT(cudaMemcpy(h1_out, d_h1, nRows * h1_cols * sizeof(Elem), cudaMemcpyDeviceToHost));
        CUDA_ASSERT(cudaFree(d_h1));
    }

    // 5) For each q2: compute a single combined matvec on [H1; a1_packed]
    //    and split the outputs into a2 (top H1_rows) and h2 (bottom nRows).
    const size_t totalRows   = H1_rows + nRows;
    const size_t commonCols  = nCols; // expected to match nCols for packed layout

    Elem* d_concat = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_concat, totalRows * commonCols * sizeof(Elem)));
    // top: H1
    CUDA_ASSERT(cudaMemcpy(d_concat,
                           d_H1,
                           H1_rows * commonCols * sizeof(Elem),
                           cudaMemcpyDeviceToDevice));
    // bottom: a1_packed
    CUDA_ASSERT(cudaMemcpy(d_concat + H1_rows * commonCols,
                           d_a1_packed,
                           nRows * commonCols * sizeof(Elem),
                           cudaMemcpyDeviceToDevice));

    for (int k=0; k<numQ2; ++k) {
        const Elem* q2_host = q2s + (size_t)k * q2_len;
        ensure_buf(&d_q2, &cap_q2, q2_len);
        CUDA_ASSERT(cudaMemcpy(d_q2, q2_host, q2_len*sizeof(Elem), cudaMemcpyHostToDevice));

        // combined matvec: [H1; a1_packed] · q2
        ensure_acc(totalRows);
        {
            dim3 block(BLOCK_SIZE);
            const int warpsPerBlock = BLOCK_SIZE / 32;
            const int K = 4;
            const size_t rowsPerBlockLogical = (size_t)warpsPerBlock * (size_t)K;
            dim3 grid( (int)((totalRows + rowsPerBlockLogical - 1) / rowsPerBlockLogical) );
            const size_t shmemBytes = (size_t)TILE_COLS * (size_t)COMPRESSION * sizeof(Elem);

            matMulVecPackedWarpSpanTileK<4><<<grid, block, shmemBytes>>>(
                (Elem*)d_acc64, d_concat, d_q2, totalRows, commonCols, 0, totalRows);
        }
        CUDA_ASSERT(cudaGetLastError());

        // split output buffer: first H1_rows -> a2, next nRows -> h2
        CUDA_ASSERT(cudaMemcpy(a2_out_all + (size_t)k * a2_vec_len,
                               d_acc64,
                               H1_rows*sizeof(Elem),
                               cudaMemcpyDeviceToHost));
        CUDA_ASSERT(cudaMemcpy(h2_out_all + (size_t)k * h2_vec_len,
                               d_acc64 + H1_rows,
                               nRows*sizeof(Elem),
                               cudaMemcpyDeviceToHost));
    }

    CUDA_ASSERT(cudaFree(d_concat));
    CUDA_ASSERT(cudaFree(d_a1_packed));
}

extern "C" void doublePIRGPUFree(void)
{
    if (d_DB)    { CUDA_ASSERT(cudaFree(d_DB));    d_DB=nullptr; }
    if (d_H1)    { CUDA_ASSERT(cudaFree(d_H1));    d_H1=nullptr; }
    if (d_A2t)   { CUDA_ASSERT(cudaFree(d_A2t));   d_A2t=nullptr; }
    if (d_q1)    { CUDA_ASSERT(cudaFree(d_q1));    d_q1=nullptr; cap_q1=0; }
    if (d_q2)    { CUDA_ASSERT(cudaFree(d_q2));    d_q2=nullptr; cap_q2=0; }
    if (d_acc64) { CUDA_ASSERT(cudaFree(d_acc64)); d_acc64=nullptr; cap_acc64=0; }
    if (d_vec_out){ CUDA_ASSERT(cudaFree(d_vec_out)); d_vec_out=nullptr; cap_vec_out=0; }
}
