// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
__global__ void key_encode_gpu(uint16_t *out, const uint16_t *in);
__global__ void key_decode_gpu(uint16_t *out, const uint16_t *in);