// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sample_n_gpu(uint16_t *s);
__global__ void sample_n_gpu_small(uint16_t *s) ;
__global__ void pack_gpu(uint8_t *out, size_t outlen, const uint16_t *in, size_t inlen, uint8_t lsb, uint32_t offset);
__global__ void pack_gpu_encap(uint8_t *out, size_t outlen, const uint16_t *in, size_t inlen, uint8_t lsb, uint32_t offset);
__global__ void unpack_gpu_encap(uint16_t *out, size_t outlen, const uint8_t *in, size_t inlen, uint8_t lsb, uint32_t offset) ;
__global__ void unpack_gpu_decap(uint16_t *out, size_t outlen, const uint8_t *in, size_t inlen, uint8_t lsb, uint32_t offset, uint32_t out_stride, uint32_t in_stride);