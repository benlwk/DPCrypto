// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
// Include local CUDA header files.
#include "include/cuda_kernel.cuh"
#include "include/api.h"
#include "include/params.h"
// #include "include/testvector.cuh"


int main() {
	uint8_t *h_pk, *h_sk, *h_sendb, *h_key_b, *h_key_a;
    uint16_t j,i;
    cudaMallocHost((void**) &h_pk,  BLOCK*CRYPTO_PUBLICKEYBYTES * sizeof(int8_t));  
    cudaMallocHost((void**) &h_sk, BLOCK*CRYPTO_SECRETKEYBYTES *sizeof(uint8_t));
    cudaMallocHost((void**) &h_sendb, BLOCK*CRYPTO_CIPHERTEXTBYTES *sizeof(uint8_t));
    cudaMallocHost((void**) &h_key_a, BLOCK*CRYPTO_BYTES *sizeof(uint8_t));
    cudaMallocHost((void**) &h_key_b, BLOCK*CRYPTO_BYTES *sizeof(uint8_t));

    frodoKEMKeypair(h_pk, h_sk);
    frodoKEMEncap(h_sendb, h_key_b, h_pk);
    frodoKEMDecap(h_key_a, h_sendb, h_sk);

    for (j = 0; j < CRYPTO_BYTES; j++) {    
        if (h_key_a[j] != h_key_b[j]) {
            printf("ERROR\n");
           return -1;
        }
    }
    printf("\nCORRECT!\n");
    return 0;
}