// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// Include associated header file.
#include "../include/cuda_kernel.cuh"
#include "../include/api.h"
#include "../include/params.h"

__global__ void copy_matrix_gpu(uint16_t *out, uint16_t *in);
__global__ void copy_matrix_gpu_x2(uint16_t *out, uint16_t *in);
__global__ void copy_matrix_gpu_encap(uint16_t *out, uint16_t *in);
__global__ void copy_matrix_gpu_encap_small(uint16_t *out, uint16_t *in);
__global__ void copy_sk_s(uint8_t *out, uint8_t *in);
__global__ void copy_sk_pk(uint8_t *out, uint8_t *in);
__global__ void copy_sk_S(uint8_t *out, uint8_t *in);
__global__ void copy_sk_pkh(uint8_t *out, uint8_t *in);
__global__ void copy_ct_encap(uint8_t *out, uint8_t *in, uint32_t in_offset, uint32_t out_offset, uint32_t in_stride,  uint32_t out_stride,  uint32_t len);
__global__ void copy_seedSE(uint8_t *out, uint8_t *in);
__global__ void add_gpu(uint16_t *out, const uint16_t *a, const uint16_t *b);
__global__ void sub_gpu(uint16_t *out, const uint16_t *a, const uint16_t *b);
__global__ void copy_u8_2_u16(uint16_t * out, uint8_t * in, uint32_t out_offset, uint32_t in_offset, uint32_t out_stride,  uint32_t in_stride, uint32_t len);
__global__ void copy_vector(uint8_t * out, uint8_t * in, uint32_t out_offset, uint32_t in_offset, uint32_t out_stride,  uint32_t in_stride, uint32_t len);
__global__ void reduce_q_gpu(uint16_t *data);
__global__ void ct_verify_gpu(int8_t *selector, const uint16_t *a, const uint16_t *b, size_t len);
__global__ void ct_select_gpu(uint8_t *r, const uint8_t *a, const uint8_t *b, int8_t *selector) ;
__global__ void or_gpu(int8_t *r, int8_t *a, int8_t *b, uint32_t out_stride,  uint32_t in_stride, uint32_t len);
__global__ void dummy(uint8_t *in);