#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t Elem;

void matMulVecPackedGPUInit(const Elem *a, size_t aRows, size_t aCols);

void matMulVecPackedGPUCompute(Elem *out, const Elem *b);

void matMulVecPackedGPUComputeRange(Elem *out, const Elem *b,
                                    size_t startRow, size_t numRows);

void matMulVecPackedGPUFree(void);

#ifdef __cplusplus
}
#endif