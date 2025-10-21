// double_pir_cuda.h
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t Elem;

// Initialize persistent GPU state (upload DB, H1, A2t).  X=info.X, delta=p.delta().
void doublePIRGPUInit(const Elem* DB, size_t DB_rows, size_t DB_cols,
                      const Elem* H1, size_t H1_rows, size_t H1_cols,
                      const Elem* A2t, size_t A2t_rows, size_t A2t_cols,
                      unsigned int X, unsigned int delta);

// Core Answer computation for a DB batch [startRow, startRow+numRows):
//  - q1: packed (DB_cols * 3)
//  - q2s: concatenated packed queries (numQ2 of length q2_len = cols*3)
//  - Produces: h1, a2_all, h2_all (host buffers provided by caller).
//
// Shapes you pass here must match your host code:
//   * h1:  (h1_rows x h1_cols)  (same as out of MatrixMulTransposedPacked)
//   * a2:  H1_rows entries per q2
//   * h2:  (a1_packed_rows) entries per q2  (rows = delta*X)
void doublePIRGPUAnswerRange(const Elem* q1, size_t q1_len,
                             const Elem* q2s, size_t q2_len, int numQ2,
                             size_t startRow, size_t numRows,
                             unsigned int mod_p,
                             Elem* h1_out, size_t h1_rows, size_t h1_cols,
                             Elem* a2_out_all, size_t a2_vec_len,
                             Elem* h2_out_all, size_t h2_vec_len);

// Free all device buffers.
void doublePIRGPUFree(void);

#ifdef __cplusplus
}
#endif
