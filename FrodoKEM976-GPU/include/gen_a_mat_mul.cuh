

// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
__global__ void aes128_ecb_gpu(unsigned char *out, const unsigned char *in, const uint64_t *rkeys);
__global__ void init_A(int16_t *A);