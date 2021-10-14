// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void as_plus_e_gpu(uint16_t *out, int16_t *A, uint16_t *s);
__global__ void sa_plus_e_gpu(uint16_t *out, int16_t *A, uint16_t *s);
__global__ void mul_add_sb_plus_e_gpu(uint16_t *out, const uint16_t *b, const uint16_t *s, const uint16_t *e) ;
__global__ void mul_bs_gpu(uint16_t *out, const uint16_t *b, const uint16_t *s) ;
__global__ void DoDP4Av5s(int16_t *out, short2 *a, char4* b);
__global__ void packdp2bv2(char4 *outb, uint16_t* b) ;
