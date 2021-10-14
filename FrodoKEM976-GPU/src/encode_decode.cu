
// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// Include associated header file.
#include "../include/api.h"
#include "../include/params.h"


__global__ void key_encode_gpu(uint16_t *out, const uint16_t *in) {
    // Encoding
    unsigned int j, npieces_word = 8;
    // unsigned int nwords = (PARAMS_NBAR * PARAMS_NBAR) / 8;
    uint64_t temp, mask = ((uint64_t)1 << PARAMS_EXTRACTED_BITS) - 1;
    uint32_t tid = threadIdx.x, bid = blockIdx.x;

    // for (i = 0; i < nwords; i++) {
        temp = 0;
        for (j = 0; j < PARAMS_EXTRACTED_BITS; j++) {
            temp |= ((uint64_t)((uint8_t *)in)[bid*(BYTES_PKHASH + BYTES_MU) + BYTES_PKHASH + tid * PARAMS_EXTRACTED_BITS + j]) << (8 * j);
        }
        for (j = 0; j < npieces_word; j++) {
            // *pos = (uint16_t)((temp & mask) << (PARAMS_LOGQ - PARAMS_EXTRACTED_BITS));
            out[bid*PARAMS_NBAR*PARAMS_NBAR + tid*npieces_word + j] = (uint16_t)((temp & mask) << (PARAMS_LOGQ - PARAMS_EXTRACTED_BITS));
            temp >>= PARAMS_EXTRACTED_BITS;                        
        }
       // out[tid] = 99;
    // }
}

__global__ void key_decode_gpu(uint16_t *out, const uint16_t *in) {
    // Decoding
    unsigned int j, index = 0, npieces_word = 8;
    unsigned int nwords = (PARAMS_NBAR * PARAMS_NBAR) / 8;
    uint16_t temp, maskex = ((uint16_t)1 << PARAMS_EXTRACTED_BITS) - 1, maskq = ((uint16_t)1 << PARAMS_LOGQ) - 1;
    uint8_t  *pos = (uint8_t *)out;
    uint64_t templong;
    uint32_t tid = threadIdx.x, bid = blockIdx.x;

    // for (i = 0; i < nwords; i++) {
        templong = 0;
        for (j = 0; j < npieces_word; j++) {  // temp = floor(in*2^{-11}+0.5)
            temp = ((in[bid*PARAMS_NBAR*PARAMS_NBAR + tid*npieces_word + j] & maskq) + (1 << (PARAMS_LOGQ - PARAMS_EXTRACTED_BITS - 1))) >> (PARAMS_LOGQ - PARAMS_EXTRACTED_BITS);
            templong |= ((uint64_t)(temp & maskex)) << (PARAMS_EXTRACTED_BITS * j);
            // index++;
        }
        for (j = 0; j < PARAMS_EXTRACTED_BITS; j++) {
            pos[bid*(BYTES_PKHASH + BYTES_MU) + tid*PARAMS_EXTRACTED_BITS + j] = (templong >> (8 * j)) & 0xFF;
        }
    // }
}