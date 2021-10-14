// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// Include associated header file.
#include "../include/api.h"
#include "../include/params.h"

// TODO: Generalize for different PARAMS_N
__constant__ static const uint16_t CDF_TABLE[CDF_TABLE_LEN] = CDF_TABLE_DATA;
__global__ void sample_n_gpu(uint16_t *s) {
    // Fills vector s with n samples from the noise distribution which requires 16 bits to sample.
    // The distribution is specified by its CDF.
    // Input: pseudo-random values (2*n bytes) passed in s. The input is overwritten by the output.
    uint32_t i, j, tid= threadIdx.x, bid= blockIdx.x;

    for (i = 0; i < PARAMS_NBAR; ++i) {
        uint16_t sample = 0;
        uint16_t prnd = s[bid*PARAMS_N*PARAMS_NBAR + i*PARAMS_N + tid] >> 1;    // Drop the least significant bit
        uint16_t sign = s[bid*PARAMS_N*PARAMS_NBAR +i*PARAMS_N + tid] & 0x1;    // Pick the least significant bit

        // No need to compare with the last value.
        for (j = 0; j < (unsigned int)(CDF_TABLE_LEN - 1); j++) {
            // Constant time comparison: 1 if CDF_TABLE[j] < s, 0 otherwise. Uses the fact that CDF_TABLE[j] and s fit in 15 bits.
            sample += (uint16_t)(CDF_TABLE[j] - prnd) >> 15;
        }
        // Assuming that sign is either 0 or 1, flips sample iff sign = 1
        s[bid*PARAMS_N*PARAMS_NBAR +i*PARAMS_N + tid] = ((-sign) ^ sample) + sign;
    }
}

// For sample_n with PARAMS_NBAR x PARAMS_NBAR only
__global__ void sample_n_gpu_small(uint16_t *s) {
    uint32_t i, j, tid= threadIdx.x, bid= blockIdx.x;

    uint16_t sample = 0;
    uint16_t prnd = s[bid*PARAMS_NBAR*PARAMS_NBAR +tid] >> 1;    // 
    uint16_t sign = s[bid*PARAMS_NBAR*PARAMS_NBAR +tid] & 0x1;    

    for (j = 0; j < (unsigned int)(CDF_TABLE_LEN - 1); j++) {
        sample += (uint16_t)(CDF_TABLE[j] - prnd) >> 15;
    }
    s[bid*PARAMS_NBAR*PARAMS_NBAR +tid] = ((-sign) ^ sample) + sign;  
}

__device__ static inline uint8_t min_gpu(uint8_t x, uint8_t y) {
    if (x < y) {
        return x;
    }
    return y;
}

__global__ void pack_gpu(uint8_t *out, size_t outlen, const uint16_t *in, size_t inlen, uint8_t lsb, uint32_t offset) {
    // Pack the input uint16 vector into a char output vector, copying lsb bits from each input element. If inlen * lsb / 8 > outlen, only outlen * 8 bits are copied.
    size_t i = 0;            // whole bytes already filled in
    size_t j = 0;            // whole uint16_t already copied
    uint16_t w = 0;          // the leftover, not yet copied
    uint8_t bits = 0;        // the number of lsb in w
    uint32_t tid = threadIdx.x, bid = blockIdx.x;

    while (i < outlen && (j < inlen || ((j == inlen) && (bits > 0)))) {
        uint8_t b = 0;  // bits in out[i] already filled in
        while (b < 8) {
            int nbits = min_gpu(8 - b, bits);
            uint16_t mask = (1 << nbits) - 1;
            uint8_t t = (uint8_t) ((w >> (bits - nbits)) & mask);  // the bits to copy from w to out
            out[offset + bid*CRYPTO_PUBLICKEYBYTES + tid*outlen + i] = out[offset + bid*CRYPTO_PUBLICKEYBYTES + tid*outlen + i] + (t << (8 - b - nbits));
            b += (uint8_t) nbits;
            bits -= (uint8_t) nbits;
            w &= ~(mask << bits);  // not strictly necessary; mostly for debugging

            if (bits == 0) {
                if (j < inlen) {
                    w = in[bid*blockDim.x*inlen + tid*inlen + j];
                    bits = lsb;
                    j++;
                } else {
                    break;  // the input vector is exhausted
                }
            }
        }
        if (b == 8) {  // out[i] is filled in
            i++;
        }
    }
}

__global__ void pack_gpu_encap(uint8_t *out, size_t outlen, const uint16_t *in, size_t inlen, uint8_t lsb, uint32_t offset) {
    // Pack the input uint16 vector into a char output vector, copying lsb bits from each input element. If inlen * lsb / 8 > outlen, only outlen * 8 bits are copied.
    size_t i = 0;            // whole bytes already filled in
    size_t j = 0;            // whole uint16_t already copied
    uint16_t w = 0;          // the leftover, not yet copied
    uint8_t bits = 0;        // the number of lsb in w
    uint32_t tid = threadIdx.x, bid = blockIdx.x;

    while (i < outlen && (j < inlen || ((j == inlen) && (bits > 0)))) {
        uint8_t b = 0;  // bits in out[i] already filled in
        while (b < 8) {
            int nbits = min_gpu(8 - b, bits);
            uint16_t mask = (1 << nbits) - 1;
            uint8_t t = (uint8_t) ((w >> (bits - nbits)) & mask);  // the bits to copy from w to out
            out[offset + bid*CRYPTO_CIPHERTEXTBYTES + tid*outlen + i] = out[offset + bid*CRYPTO_CIPHERTEXTBYTES + tid*outlen + i] + (t << (8 - b - nbits));
            b += (uint8_t) nbits;
            bits -= (uint8_t) nbits;
            w &= ~(mask << bits);  // not strictly necessary; mostly for debugging

            if (bits == 0) {
                if (j < inlen) {
                    w = in[bid*blockDim.x*inlen + tid*inlen + j];
                    bits = lsb;
                    j++;
                } else {
                    break;  // the input vector is exhausted
                }
            }
        }
        if (b == 8) {  // out[i] is filled in
            i++;
        }
    }
}


__global__ void unpack_gpu_encap(uint16_t *out, size_t outlen, const uint8_t *in, size_t inlen, uint8_t lsb, uint32_t offset) {
    // Unpack the input char vector into a uint16_t output vector, copying lsb bits
    // for each output element from input. outlen must be at least ceil(inlen * 8 / lsb).

    size_t i = 0;            // whole uint16_t already filled in
    size_t j = 0;            // whole bytes already copied
    uint8_t w = 0;           // the leftover, not yet copied
    uint8_t bits = 0;        // the number of lsb bits of w
    uint32_t tid = threadIdx.x, bid = blockIdx.x;

    while (i < outlen && (j < inlen || ((j == inlen) && (bits > 0)))) {
        uint8_t b = 0;  // bits in out[i] already filled in
        while (b < lsb) {
            int nbits = min(lsb - b, bits);
            uint16_t mask = (1 << nbits) - 1;
            uint8_t t = (w >> (bits - nbits)) & mask;  // the bits to copy from w to out
            out[bid*PARAMS_N * PARAMS_NBAR + tid*outlen + i] = out[bid*PARAMS_N * PARAMS_NBAR + tid*outlen + i] + (t << (lsb - b - nbits));
            b += (uint8_t) nbits;
            bits -= (uint8_t) nbits;
            w &= ~(mask << bits);  // not strictly necessary; mostly for debugging

            if (bits == 0) {
                if (j < inlen) {
                    w = in[offset + bid*CRYPTO_PUBLICKEYBYTES + tid*inlen + j];
                    bits = 8;
                    j++;
                } else {
                    break;  // the input vector is exhausted
                }
            }
        }
        if (b == lsb) {  // out[i] is filled in
            i++;
        }
    }
}


__global__ void unpack_gpu_decap(uint16_t *out, size_t outlen, const uint8_t *in, size_t inlen, uint8_t lsb, uint32_t offset, uint32_t out_stride, uint32_t in_stride) {
    // Unpack the input char vector into a uint16_t output vector, copying lsb bits
    // for each output element from input. outlen must be at least ceil(inlen * 8 / lsb).

    size_t i = 0;            // whole uint16_t already filled in
    size_t j = 0;            // whole bytes already copied
    uint8_t w = 0;           // the leftover, not yet copied
    uint8_t bits = 0;        // the number of lsb bits of w
    uint32_t tid = threadIdx.x, bid = blockIdx.x;

    while (i < outlen && (j < inlen || ((j == inlen) && (bits > 0)))) {
        uint8_t b = 0;  // bits in out[i] already filled in
        while (b < lsb) {
            int nbits = min(lsb - b, bits);
            uint16_t mask = (1 << nbits) - 1;
            uint8_t t = (w >> (bits - nbits)) & mask;  // the bits to copy from w to out
            out[bid*out_stride + tid*outlen + i] = out[bid*out_stride + tid*outlen + i] + (t << (lsb - b - nbits));
            b += (uint8_t) nbits;
            bits -= (uint8_t) nbits;
            w &= ~(mask << bits);  // not strictly necessary; mostly for debugging

            if (bits == 0) {
                if (j < inlen) {
                    w = in[offset + bid*in_stride + tid*inlen + j];
                    bits = 8;
                    j++;
                } else {
                    break;  // the input vector is exhausted
                }
            }
        }
        if (b == lsb) {  // out[i] is filled in
            i++;
        }
    }
}