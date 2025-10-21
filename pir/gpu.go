package pir

// #cgo CFLAGS: -O3 -march=native -I.
// #cgo LDFLAGS: -L. -lpir_cuda
// #include "pir.h"
// #include "simple_pir_cuda.h"
// #include "double_pir_cuda.h"
import "C"

import (
	"os"
)

var useGPU = os.Getenv("USE_GPU") == "1"

func GPUInitDB(a *Matrix) {
	C.matMulVecPackedGPUInit((*C.Elem)(&a.Data[0]), C.size_t(a.Rows), C.size_t(a.Cols))
}

func GPUCompute(b *Matrix, start, rows uint64) *Matrix {
	out := MatrixNew(rows, 1)
	C.matMulVecPackedGPUComputeRange(
		(*C.Elem)(&out.Data[0]),
		(*C.Elem)(&b.Data[0]),
		C.size_t(start), C.size_t(rows),
	)
	return out
}

func GPUFree() {
	C.matMulVecPackedGPUFree()
}

func DoubleGPUInit(DB, H1, A2t *Matrix, X, delta uint64) {
	C.doublePIRGPUInit(
		(*C.Elem)(&DB.Data[0]), C.size_t(DB.Rows), C.size_t(DB.Cols),
		(*C.Elem)(&H1.Data[0]), C.size_t(H1.Rows), C.size_t(H1.Cols),
		(*C.Elem)(&A2t.Data[0]), C.size_t(A2t.Rows), C.size_t(A2t.Cols),
		C.uint(X), C.uint(delta),
	)
}

func DoubleGPUAnswerRange(
	q1 *Matrix,
	q2s []*Matrix, // length = info.Ne/info.X
	start, rows uint64,
	modP uint32,
	h1_out *Matrix,
	a2_all, h2_all *Matrix,
) {
	// flatten q2s
	var flat []C.Elem
	for _, q2 := range q2s {
		flat = append(flat, q2.Data...)
	}
	C.doublePIRGPUAnswerRange(
		(*C.Elem)(&q1.Data[0]), C.size_t(q1.Rows),
		(*C.Elem)(&flat[0]), C.size_t(q2s[0].Rows), C.int(len(q2s)),
		C.size_t(start), C.size_t(rows),
		C.uint(modP),
		(*C.Elem)(&h1_out.Data[0]), C.size_t(h1_out.Rows), C.size_t(h1_out.Cols),
		(*C.Elem)(&a2_all.Data[0]), C.size_t(a2_all.Rows), // vec-len per q2
		(*C.Elem)(&h2_all.Data[0]), C.size_t(h2_all.Rows),
	)
}

func DoubleGPUFree() { C.doublePIRGPUFree() }
