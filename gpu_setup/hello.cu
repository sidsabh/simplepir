#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU(void) {
    printf("Hello World from GPU on thread %d!\n", threadIdx.x);
}
int main(void) {
    // hello from cpu
    printf("Hello World from CPU!\n");

    helloFromGPU<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}