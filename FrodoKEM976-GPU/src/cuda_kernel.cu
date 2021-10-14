// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// Include associated header file.
#include "../include/cuda_kernel.cuh"
#include "../include/aes.h"
// #include "../include/testvector.cuh"
#include "../include/api.h"
#include "../include/params.h"
#include "../include/gen_a_mat_mul.cuh"
#include "../include/matrix_kernel.cuh"
#include "../include/sample_pack.cuh"
#include "../include/fips202.h"
#include "../include/shake.cuh"
#include "../include/util.cuh"
#include "../include/encode_decode.cuh"

void frodoKEMKeypair(uint8_t *h_pk, uint8_t *h_sk) {
    uint32_t i, j, k;
    uint8_t *d_pk, *h_randomness, *d_randomness, *h_sk_pkh, *d_sk_pkh, *d_seedSE, *h_seedSE, *d_sk;
    int16_t *d_A, *h_A;
    uint16_t *d_B, *h_B, *d_s, *h_s, *d_S, *h_S;
    cudaEvent_t start, stop;
    float elapsed = 0.0f;

    // Allocate memory
    cudaMallocHost((void**) &h_A, BLOCK*PARAMS_N * PARAMS_N * sizeof(int16_t));
    cudaMallocHost((void**) &h_s, BLOCK*2*PARAMS_N * PARAMS_NBAR * sizeof(uint16_t));
    cudaMallocHost((void**) &h_S, BLOCK*PARAMS_N * PARAMS_NBAR * sizeof(uint16_t));
    cudaMallocHost((void**) &h_randomness, BLOCK*(2 * CRYPTO_BYTES + BYTES_SEED_A)*sizeof(uint8_t));
    cudaMallocHost((void**) &h_sk_pkh, BLOCK*24 *sizeof(uint8_t));
    cudaMallocHost((void**) &h_seedSE, BLOCK*(1 + CRYPTO_BYTES)*sizeof(uint8_t));

    cudaMalloc((void**) &d_A,  BLOCK*PARAMS_N * PARAMS_N * sizeof(int16_t));    
    cudaMalloc((void**) &d_B, BLOCK*PARAMS_N * PARAMS_NBAR * sizeof(uint16_t)); 
    cudaMalloc((void**) &d_s, BLOCK*2*PARAMS_N * PARAMS_NBAR * sizeof(uint16_t));
    cudaMalloc((void**) &d_S, BLOCK*PARAMS_N * PARAMS_NBAR * sizeof(uint16_t));
    cudaMalloc((void**) &d_pk,  BLOCK*CRYPTO_PUBLICKEYBYTES * sizeof(int8_t));  
    cudaMalloc((void**) &d_randomness, BLOCK*(2 * CRYPTO_BYTES + BYTES_SEED_A)*sizeof(uint8_t));
    cudaMalloc((void**) &d_sk_pkh, BLOCK*24 *sizeof(uint8_t));
    cudaMalloc((void**) &d_seedSE, BLOCK*(1 + CRYPTO_BYTES)*sizeof(uint8_t));
    cudaMalloc((void**) &d_sk, BLOCK*CRYPTO_SECRETKEYBYTES *sizeof(uint8_t));
    cudaMemset(d_s, 0 , BLOCK*2*PARAMS_N * PARAMS_NBAR * sizeof(uint16_t));
    cudaEventCreate(&start);    cudaEventCreate(&stop) ;

    // Initialize arrays    
    // Generate the secret value s, the seed for S and E, and the seed for the seed for A. Add seed_A to the public key          
    for(i=0; i<CRYPTO_BYTES + BYTES_SEED_A; i++) h_randomness[i] = i;    
    h_seedSE[0] = 0x5F;
    for(i=0; i<CRYPTO_BYTES; i++) h_seedSE[i+1] = h_randomness[CRYPTO_BYTES+i];
        
    cudaEventRecord(start, 0); 
    // Transfer arrays to GPU.
    cudaMemcpy(d_randomness, h_randomness, (2 * CRYPTO_BYTES + BYTES_SEED_A)* sizeof(uint8_t), cudaMemcpyHostToDevice);   
    cudaMemcpy(d_seedSE, h_seedSE, BLOCK*(CRYPTO_BYTES+1)* sizeof(uint8_t), cudaMemcpyHostToDevice);    
#ifdef AES    
    cudaMemcpy(d_aes_key_exp, h_aes_key_exp, PQC_AES128_STATESIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);      
#endif  

    //shake256    
    shake256_gpu<<<BLOCK,32>>>(d_pk, d_randomness + (2 * CRYPTO_BYTES)/*randomness_z*/, BYTES_SEED_A, BYTES_SEED_A, CRYPTO_PUBLICKEYBYTES); 
    shake256_gpu<<<BLOCK,32>>>((uint8_t*) d_s, d_seedSE, 1 + CRYPTO_BYTES, (2 * PARAMS_N * PARAMS_NBAR * sizeof(uint16_t)), 4 * PARAMS_N * PARAMS_NBAR); 
    //frodo_sample
    copy_matrix_gpu_x2<<<BLOCK,PARAMS_N>>>(d_B, d_s + PARAMS_N * PARAMS_NBAR); // Copy E to B.
    copy_matrix_gpu<<<BLOCK,PARAMS_N>>>(d_S, d_s); // Copy s to S.
    sample_n_gpu<<<BLOCK,PARAMS_N>>>(d_S); // S.    
    sample_n_gpu<<<BLOCK,PARAMS_N>>>(d_B); // E. Store in d_B directly.
    // gen_a
    // shake128_gpu<<<BLOCK, 32>>>((uint8_t *)d_A, (uint8_t *)d_pk, BYTES_SEED_A, 2*PARAMS_N*PARAMS_N, PARAMS_N * PARAMS_N);
    shake128_ser<<<BLOCK, PARAMS_N/2>>>((uint8_t *)d_A, 2*PARAMS_N, d_pk, BYTES_SEED_A);
    shake128_ser<<<BLOCK, PARAMS_N/2>>>((uint8_t *)d_A+PARAMS_N* PARAMS_N, 2*PARAMS_N, d_pk, BYTES_SEED_A);

    // // // as_plus_e
    as_plus_e_gpu<<<BLOCK, PARAMS_N>>>(d_B, d_A, d_S);
    // // // Each thread pack 16 words to 32 bytes. 
    pack_gpu<<<BLOCK,488>>>(d_pk, 32, d_B, 16, PARAMS_LOGQ, BYTES_SEED_A);    
    shake256_gpu<<<BLOCK,32>>>(d_sk_pkh, d_pk, CRYPTO_PUBLICKEYBYTES, BYTES_PKHASH, BYTES_PKHASH);
    copy_sk_s<<<BLOCK, CRYPTO_BYTES>>>(d_sk, d_randomness);
    copy_sk_pk<<<BLOCK, 1024>>>(d_sk, d_pk);
    copy_sk_S<<<BLOCK, 1024>>>(d_sk, (uint8_t*) d_S);
    copy_sk_pkh<<<BLOCK, 24>>>(d_sk, d_sk_pkh);  

    cudaMemcpy(h_pk, d_pk, BLOCK*CRYPTO_PUBLICKEYBYTES* sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sk, d_sk, BLOCK*CRYPTO_SECRETKEYBYTES* sizeof(int8_t), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop) ;
    printf("\nTotal Keypair time: %.4f ms, TP: %.0f\n", elapsed, BLOCK*1000/elapsed);
    // printf("\n pk\n"); for(k=0; k<BLOCK; k++) for(i=0; i<CRYPTO_PUBLICKEYBYTES; i++) printf("%u ", h_pk[k* CRYPTO_PUBLICKEYBYTES + i]);       
    // printf("\n sk\n"); for(k=0; k<BLOCK; k++) {printf("\nbatch %u\n", k);        for(i=CRYPTO_SECRETKEYBYTES-16; i<CRYPTO_SECRETKEYBYTES; i++) printf("%u %x \n", i, h_sk[k* CRYPTO_SECRETKEYBYTES + i]);   }
  
    cudaFreeHost(h_randomness);    cudaFreeHost(h_sk_pkh);
    cudaFreeHost(h_seedSE);        cudaFreeHost(h_s);    cudaFreeHost(h_S);
    cudaFree(d_randomness);    cudaFree(d_sk_pkh);
    cudaFree(d_seedSE);    cudaFree(d_A);  
    cudaFree(d_B);    cudaFree(d_s);    cudaFree(d_S);
    cudaFree(d_pk);
    cudaFree(d_sk);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


void frodoKEMEncap(uint8_t *ct, uint8_t *ss, const uint8_t *pk) 
{
    uint8_t *h_G2in, *d_G2in, *d_pk, *h_G2out, *d_G2out, *h_seedSE, *d_seedSE, *d_ct, *h_Fin, *d_Fin, *d_ss;
    uint16_t *d_sp, *h_Sp, *d_Sp, *d_Bp, *d_Spp, *h_Epp, *d_Epp, *h_Bp, *d_B, *h_C, *d_V, *d_C;
    int32_t i, j, k;
    int16_t *d_A, *h_A;        
    char4 *d_packed_dSp;
    cudaEvent_t start, stop;
    float elapsed = 0.0f;

    cudaMallocHost((void**) &h_G2in, BLOCK* (BYTES_PKHASH + BYTES_MU) *sizeof(uint8_t));
    cudaMallocHost((void**) &h_G2out, BLOCK* 2 * CRYPTO_BYTES *sizeof(uint8_t));
    cudaMallocHost((void**) &h_seedSE, BLOCK*(1 + CRYPTO_BYTES)*sizeof(uint8_t));
    cudaMallocHost((void**) &h_Sp, BLOCK*PARAMS_N*PARAMS_NBAR*sizeof(uint16_t));
    
    cudaMallocHost((void**) &h_Bp, BLOCK*PARAMS_N*PARAMS_NBAR*sizeof(uint16_t));
    cudaMallocHost((void**) &h_Epp, BLOCK*PARAMS_NBAR*PARAMS_NBAR*sizeof(uint16_t));   
    cudaMallocHost((void**) &h_C, BLOCK*PARAMS_NBAR*PARAMS_NBAR*sizeof(uint16_t));
    cudaMallocHost((void**) &h_Fin, BLOCK*(CRYPTO_CIPHERTEXTBYTES + CRYPTO_BYTES)*sizeof(uint8_t));
    cudaMallocHost((void**) &h_A,  BLOCK*PARAMS_N * PARAMS_N * sizeof(int16_t)); 
    
    cudaMalloc((void**)&d_packed_dSp, BLOCK * PARAMS_N * PARAMS_NBAR/2 * sizeof(char4));
    cudaMalloc((void**) &d_G2in, BLOCK* (BYTES_PKHASH + BYTES_MU) *sizeof(uint8_t));
    cudaMalloc((void**) &d_pk,  BLOCK*CRYPTO_PUBLICKEYBYTES * sizeof(int8_t));  
    cudaMalloc((void**) &d_G2out, BLOCK* 2 * CRYPTO_BYTES *sizeof(uint8_t));
    cudaMalloc((void**) &d_seedSE, BLOCK*(1 + CRYPTO_BYTES)*sizeof(uint8_t));
    cudaMalloc((void**) &d_sp, BLOCK*(2 * PARAMS_N + PARAMS_NBAR)*PARAMS_NBAR*sizeof(uint16_t));
    cudaMalloc((void**) &d_Sp, BLOCK*PARAMS_N*PARAMS_NBAR*sizeof(uint16_t));
    cudaMalloc((void**) &d_Bp, BLOCK*PARAMS_N*PARAMS_NBAR*sizeof(uint16_t));
    cudaMalloc((void**) &d_B, BLOCK*PARAMS_N*PARAMS_NBAR*sizeof(uint16_t));
    cudaMalloc((void**) &d_Epp, BLOCK*PARAMS_NBAR*PARAMS_NBAR*sizeof(uint16_t));
    cudaMalloc((void**) &d_A,  BLOCK*PARAMS_N * PARAMS_N * sizeof(int16_t)); 
    cudaMalloc((void**) &d_ct, BLOCK*CRYPTO_CIPHERTEXTBYTES *sizeof(uint8_t));
    cudaMalloc((void**) &d_V, BLOCK*PARAMS_NBAR*PARAMS_NBAR*sizeof(uint16_t));
    cudaMalloc((void**) &d_C, BLOCK*PARAMS_NBAR*PARAMS_NBAR*sizeof(uint16_t));
    cudaMalloc((void**) &d_Fin, BLOCK*(CRYPTO_CIPHERTEXTBYTES + CRYPTO_BYTES)*sizeof(uint8_t));
    cudaMalloc((void**) &d_ss, BLOCK*CRYPTO_BYTES*sizeof(uint8_t));

    cudaEventCreate(&start);    cudaEventCreate(&stop) ;
    // TODO: Use different randomness
    for(k=0; k<BLOCK; k++) for(i=0; i<BYTES_MU+BYTES_PKHASH; i++)h_G2in[i]=0;
    for(i=0; i<BYTES_MU; i++) h_G2in[BYTES_PKHASH + i] = i;
    for(k=0; k<BLOCK; k++) for(i=0; i<BYTES_MU; i++) h_G2in[k*(BYTES_PKHASH + BYTES_MU) + BYTES_PKHASH + i] = i;        
    
    cudaEventRecord(start, 0); 
for(k=0; k<REPEAT; k++)    
{
    // Transfer arrays to GPU.
     cudaMemcpy(d_pk, pk, CRYPTO_PUBLICKEYBYTES* sizeof(uint8_t), cudaMemcpyHostToDevice);  
    cudaMemcpy(d_G2in, h_G2in, BLOCK*(BYTES_PKHASH + BYTES_MU)* sizeof(uint8_t), cudaMemcpyHostToDevice); 

    shake256_gpu<<<BLOCK,32>>>(d_G2in, d_pk, CRYPTO_PUBLICKEYBYTES, BYTES_PKHASH, (BYTES_PKHASH + BYTES_MU)); 
    shake256_gpu<<<BLOCK,32>>>(d_G2out, d_G2in, BYTES_PKHASH + BYTES_MU, CRYPTO_BYTES + CRYPTO_BYTES, 2 * CRYPTO_BYTES ); 
    copy_seedSE<<<BLOCK,CRYPTO_BYTES>>>(d_seedSE, d_G2out);
    shake256_gpu<<<BLOCK,32>>>((uint8_t *)d_sp, d_seedSE, 1 + CRYPTO_BYTES, (2 * PARAMS_N + PARAMS_NBAR) * PARAMS_NBAR * sizeof(uint16_t), (2 * PARAMS_N + PARAMS_NBAR) * PARAMS_NBAR * sizeof(uint16_t)); 

    copy_matrix_gpu_encap<<<BLOCK,PARAMS_N>>>(d_Bp, d_sp + PARAMS_N * PARAMS_NBAR);
    copy_matrix_gpu_encap_small<<<BLOCK,64>>>(d_Epp, d_sp + 2*PARAMS_N * PARAMS_NBAR);
    copy_matrix_gpu_encap<<<BLOCK,PARAMS_N>>>(d_Sp, d_sp); 
    sample_n_gpu<<<BLOCK,PARAMS_N>>>(d_Sp); // S.    
    sample_n_gpu<<<BLOCK,PARAMS_N>>>(d_Bp); // Ep. // E. Store in d_B directly.
    sample_n_gpu_small<<<BLOCK,PARAMS_NBAR*PARAMS_NBAR>>>(d_Epp); // Epp
    // Generate random samples   
    // shake128_gpu<<<BLOCK, 32>>>((uint8_t *)d_A, (uint8_t *)d_pk, BYTES_SEED_A, 2*PARAMS_N*PARAMS_N, PARAMS_N * PARAMS_N);   
    shake128_ser<<<BLOCK, PARAMS_N/2>>>((uint8_t *)d_A, 2*PARAMS_N, d_pk, BYTES_SEED_A);
    shake128_ser<<<BLOCK, PARAMS_N/2>>>((uint8_t *)d_A+PARAMS_N* PARAMS_N, 2*PARAMS_N, d_pk, BYTES_SEED_A);   
#ifdef DPFRO
    packdp2bv2<<<BLOCK,PARAMS_N/2>>>(d_packed_dSp, d_Sp);
    DoDP4Av5s<<<BLOCK, PARAMS_N>>>((int16_t*) d_Bp, (short2*)d_A,d_packed_dSp);
#else    
    sa_plus_e_gpu<<<BLOCK, PARAMS_N>>>(d_Bp, d_A, d_Sp);  
#endif    
 
    pack_gpu_encap<<<BLOCK,488>>>(d_ct, 32, d_Bp, 16, PARAMS_LOGQ, 0); 
    unpack_gpu_encap<<<BLOCK,488>>>(d_B, 16, d_pk, 32, PARAMS_LOGQ, BYTES_SEED_A); 
    mul_add_sb_plus_e_gpu<<<BLOCK,PARAMS_NBAR>>>(d_V, d_B, d_Sp, d_Epp);
    key_encode_gpu<<<BLOCK, PARAMS_NBAR>>>(d_C, (uint16_t*) d_G2in) ; // encode mu
    add_gpu<<<BLOCK, PARAMS_NBAR*PARAMS_NBAR>>>(d_C, d_V, d_C);
    pack_gpu_encap<<<BLOCK,4>>>(d_ct + (PARAMS_LOGQ * PARAMS_N * PARAMS_NBAR) / 8, 32, d_C, 16, PARAMS_LOGQ, 0); 
    copy_ct_encap<<<BLOCK,PARAMS_N>>>(d_Fin, d_ct, 0, 0, CRYPTO_CIPHERTEXTBYTES, CRYPTO_CIPHERTEXTBYTES+CRYPTO_BYTES, CRYPTO_CIPHERTEXTBYTES);
    copy_ct_encap<<<BLOCK,PARAMS_N>>>(d_Fin, d_G2out, CRYPTO_BYTES, CRYPTO_CIPHERTEXTBYTES,  2 * CRYPTO_BYTES, CRYPTO_CIPHERTEXTBYTES+CRYPTO_BYTES, CRYPTO_BYTES);
    shake256_gpu<<<BLOCK,32>>>(d_ss, d_Fin, CRYPTO_CIPHERTEXTBYTES + CRYPTO_BYTES , CRYPTO_BYTES, CRYPTO_BYTES);   
    cudaMemcpy(ss, d_ss, BLOCK*CRYPTO_BYTES* sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(ct, d_ct, BLOCK*CRYPTO_CIPHERTEXTBYTES* sizeof(uint8_t), cudaMemcpyDeviceToHost);
}
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop) ;
    printf("\nTotal encapsulation time: %.4f ms, TP: %.0f \n", elapsed, BLOCK*REPEAT* 1000/elapsed);

    // printf("\n ct\n"); for(k=0; k<BLOCK; k++) {printf("\n"); for(i=0; i<CRYPTO_CIPHERTEXTBYTES; i++) printf("%x ", ct[k*CRYPTO_CIPHERTEXTBYTES + i]);}    
    // printf("\n ss\n"); for(k=0; k<8; k++) {printf("\n"); for(i=0; i<CRYPTO_BYTES; i++) printf("%x ", ss[k*CRYPTO_BYTES + i]);}
    
    cudaFreeHost(h_Sp); cudaFreeHost(h_Bp); cudaFreeHost(h_Epp);
    cudaFreeHost(h_C); cudaFreeHost(h_Fin);

    cudaFree(d_pk);    cudaFree(d_G2in);    cudaFree(d_G2out);
    cudaFree(d_seedSE);    cudaFree(d_sp);    cudaFree(d_Sp);
    cudaFree(d_Bp);    cudaFree(d_B);    cudaFree(d_Epp);
    cudaFree(d_A);        cudaFree(d_ct);
    cudaFree(d_V);    cudaFree(d_C);    cudaFree(d_Fin);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void frodoKEMDecap(uint8_t *ss, const uint8_t *ct, const uint8_t *sk) 
{
    uint16_t *d_B, *d_Bp, *d_W, *d_C, *d_CC, *h_C, *d_S, *h_S, *d_sp, *h_sp, *d_Sp, *h_Sp, *h_BBp, *d_BBp, *d_Epp;
    uint32_t i, j, k;
    uint8_t *d_sk, *d_ct, *h_G2in, *d_G2in, *h_G2out, *d_G2out, *d_seedSE, *h_Fin, *d_Fin, *d_ss;    
    int16_t *d_A, *h_A;
    int8_t *h_selector, *h_selector1, *h_selector2, *d_selector, *d_selector1, *d_selector2;
    char4 *d_packed_dSp;

    cudaEvent_t start, stop;
    float elapsed = 0.0f;

    cudaMallocHost((void**) &h_C, BLOCK*PARAMS_NBAR*PARAMS_NBAR*sizeof(uint16_t));
    cudaMallocHost((void**) &h_S, BLOCK*PARAMS_N*PARAMS_NBAR*sizeof(uint16_t));
    cudaMallocHost((void**) &h_G2in, BLOCK*(BYTES_PKHASH + BYTES_MU)*sizeof(uint8_t));
    cudaMallocHost((void**) &h_G2out, BLOCK*2*CRYPTO_BYTES*sizeof(uint8_t)); 
    cudaMallocHost((void**) &h_sp, BLOCK*(2 * PARAMS_N + PARAMS_NBAR)*PARAMS_NBAR* sizeof(uint16_t));
    cudaMallocHost((void**) &h_Sp, BLOCK*PARAMS_N * PARAMS_NBAR * sizeof(uint16_t));
    cudaMallocHost((void**) &h_BBp, BLOCK*PARAMS_N * PARAMS_NBAR * sizeof(uint16_t));
    cudaMallocHost((void**) &h_Fin, BLOCK*(CRYPTO_CIPHERTEXTBYTES + CRYPTO_BYTES) * sizeof(uint8_t));
    cudaMallocHost((void**) &h_A,  BLOCK*PARAMS_N * PARAMS_N * sizeof(int16_t)); 
    cudaMallocHost((void**) &h_selector, BLOCK* sizeof(int8_t));
    cudaMallocHost((void**) &h_selector1, BLOCK* sizeof(int8_t));
    cudaMallocHost((void**) &h_selector2, BLOCK* sizeof(int8_t));

    cudaMalloc((void**)&d_packed_dSp, BLOCK * PARAMS_N * PARAMS_NBAR/2 * sizeof(char4));
    cudaMalloc((void**) &d_B, BLOCK*PARAMS_N*PARAMS_NBAR*sizeof(uint16_t));
    cudaMalloc((void**) &d_Bp, BLOCK*PARAMS_N*PARAMS_NBAR*sizeof(uint16_t));
    cudaMalloc((void**) &d_W, BLOCK*PARAMS_NBAR*PARAMS_NBAR*sizeof(uint16_t));
    cudaMalloc((void**) &d_C, BLOCK*PARAMS_NBAR*PARAMS_NBAR*sizeof(uint16_t));
    cudaMalloc((void**) &d_CC, BLOCK*PARAMS_NBAR*PARAMS_NBAR*sizeof(uint16_t));
    cudaMalloc((void**) &d_S, BLOCK*PARAMS_N*PARAMS_NBAR*sizeof(uint16_t));
    cudaMalloc((void**) &d_sk, BLOCK*CRYPTO_SECRETKEYBYTES *sizeof(uint8_t));
    cudaMalloc((void**) &d_ct, BLOCK*CRYPTO_CIPHERTEXTBYTES *sizeof(uint8_t));
    cudaMalloc((void**) &d_G2in, BLOCK*(BYTES_PKHASH + BYTES_MU)*sizeof(uint8_t));
    cudaMalloc((void**) &d_G2out, BLOCK*2*CRYPTO_BYTES*sizeof(uint8_t));
    cudaMalloc((void**) &d_seedSE, BLOCK*(1 + CRYPTO_BYTES)*sizeof(uint8_t));
    cudaMalloc((void**) &d_sp, BLOCK*(2 * PARAMS_N + PARAMS_NBAR)*PARAMS_NBAR* sizeof(uint16_t));
    cudaMalloc((void**) &d_Sp, BLOCK*PARAMS_N * PARAMS_NBAR * sizeof(uint16_t));
    cudaMalloc((void**) &d_Epp, BLOCK*PARAMS_NBAR * PARAMS_NBAR * sizeof(uint16_t));
    cudaMalloc((void**) &d_BBp, BLOCK*PARAMS_N * PARAMS_NBAR * sizeof(uint16_t));
    cudaMalloc((void**) &d_A,  BLOCK*PARAMS_N * PARAMS_N * sizeof(int16_t)); 
    cudaMalloc((void**) &d_Fin, BLOCK*(CRYPTO_CIPHERTEXTBYTES + CRYPTO_BYTES) * sizeof(uint8_t));
    cudaMalloc((void**) &d_selector, BLOCK* sizeof(int8_t));
    cudaMalloc((void**) &d_selector1, BLOCK* sizeof(int8_t));
    cudaMalloc((void**) &d_selector2, BLOCK* sizeof(int8_t));
    cudaMalloc((void**) &d_ss, BLOCK*CRYPTO_BYTES*sizeof(uint8_t));
    
    // Transfer arrays to GPU.
    cudaEventCreate(&start);    cudaEventCreate(&stop) ;
    cudaMemset(d_B, 0, BLOCK*PARAMS_N * PARAMS_NBAR*sizeof(uint16_t));
    cudaMemset(d_Bp, 0, BLOCK*PARAMS_N * PARAMS_NBAR*sizeof(uint16_t));
    cudaMemset(d_BBp, 0, BLOCK*PARAMS_N * PARAMS_NBAR*sizeof(uint16_t));
    cudaMemset(d_Sp, 0, BLOCK*PARAMS_N * PARAMS_NBAR*sizeof(uint16_t));
    cudaMemset(d_C, 0, BLOCK*PARAMS_NBAR* PARAMS_NBAR*sizeof(uint16_t));
    cudaMemset(d_CC, 0, BLOCK*PARAMS_NBAR* PARAMS_NBAR*sizeof(uint16_t));
    cudaMemset(d_W, 0, BLOCK*PARAMS_NBAR* PARAMS_NBAR*sizeof(uint16_t));
    cudaMemset(d_A, 0, BLOCK*PARAMS_N* PARAMS_N*sizeof(uint16_t));

    cudaEventRecord(start, 0); 
for(k=0; k<REPEAT; k++)    // To compute the average throughput
{
    // Transfer arrays to GPU.
#ifdef AES    
    cudaMemcpy(d_aes_key_exp, h_aes_key_exp, PQC_AES128_STATESIZE * sizeof(uint64_t), cudaMemcpyHostToDevice); // expanded key
#endif    
    cudaMemcpy(d_sk, sk, BLOCK*CRYPTO_SECRETKEYBYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);  
    cudaMemcpy(d_ct, ct, BLOCK*CRYPTO_CIPHERTEXTBYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);  

    copy_u8_2_u16<<<BLOCK, PARAMS_N>>>(d_S, d_sk, 0, CRYPTO_BYTES + CRYPTO_PUBLICKEYBYTES, PARAMS_N * PARAMS_NBAR, CRYPTO_SECRETKEYBYTES, PARAMS_N * PARAMS_NBAR);    
    // Compute W = C - Bp*S (mod q), and decode the randomness mu
    unpack_gpu_decap<<<BLOCK,488>>>(d_Bp, 16, d_ct, 32, PARAMS_LOGQ, 0, PARAMS_N*PARAMS_NBAR, CRYPTO_CIPHERTEXTBYTES); 
    unpack_gpu_decap<<<BLOCK,4>>>(d_C, 16, d_ct, 32, PARAMS_LOGQ, (PARAMS_LOGQ * PARAMS_N * PARAMS_NBAR) / 8, PARAMS_NBAR*PARAMS_NBAR, CRYPTO_CIPHERTEXTBYTES); 
    mul_bs_gpu<<<BLOCK, PARAMS_NBAR>>>(d_W, d_Bp, d_S);
    sub_gpu<<<BLOCK, PARAMS_NBAR*PARAMS_NBAR>>>(d_W, d_C, d_W);
    key_decode_gpu<<<BLOCK, PARAMS_NBAR>>>((uint16_t *)d_G2in + BYTES_PKHASH/2, d_W);
    copy_vector<<<BLOCK, BYTES_PKHASH>>>(d_G2in, d_sk, 0, CRYPTO_BYTES + CRYPTO_PUBLICKEYBYTES + 2 * PARAMS_N * PARAMS_NBAR, (BYTES_PKHASH + BYTES_MU), CRYPTO_SECRETKEYBYTES, BYTES_PKHASH);
    shake256_gpu<<<BLOCK,32>>>(d_G2out, d_G2in, BYTES_PKHASH + BYTES_MU, CRYPTO_BYTES + CRYPTO_BYTES, 2*CRYPTO_BYTES);   
    copy_seedSE<<<BLOCK, CRYPTO_BYTES>>>(d_seedSE, d_G2out);
    shake256_gpu<<<BLOCK,32>>>((uint8_t *)d_sp, d_seedSE, 1+CRYPTO_BYTES, (2 * PARAMS_N + PARAMS_NBAR) * PARAMS_NBAR *sizeof(uint16_t), (2 * PARAMS_N + PARAMS_NBAR) * PARAMS_NBAR *sizeof(uint16_t));   
    copy_matrix_gpu_encap<<<BLOCK,PARAMS_N>>>(d_BBp, d_sp + PARAMS_N * PARAMS_NBAR); 
    copy_matrix_gpu_encap<<<BLOCK,PARAMS_N>>>(d_Sp, d_sp); // Copy s to S.
    copy_matrix_gpu_encap_small<<<BLOCK,64>>>(d_Epp, d_sp + 2*PARAMS_N * PARAMS_NBAR);
    sample_n_gpu<<<BLOCK,PARAMS_N>>>(d_Sp); // S.    
    sample_n_gpu<<<BLOCK,PARAMS_N>>>(d_BBp); // BBp. Store in d_BBp directly.
    sample_n_gpu_small<<<BLOCK,PARAMS_NBAR*PARAMS_NBAR>>>(d_Epp); 

    // Generate random samples
    // shake128_gpu<<<BLOCK, 32>>>((uint8_t *)d_A, (uint8_t *)d_sk+24, BYTES_SEED_A, 2*PARAMS_N*PARAMS_N, PARAMS_N * PARAMS_N);
    shake128_ser<<<BLOCK, PARAMS_N/2>>>((uint8_t *)d_A, 2*PARAMS_N, d_sk+24, BYTES_SEED_A);    
    shake128_ser<<<BLOCK, PARAMS_N/2>>>((uint8_t *)d_A+PARAMS_N*PARAMS_N, 2*PARAMS_N, d_sk+24, BYTES_SEED_A); 
          
#ifdef DPFRO
    packdp2bv2<<<BLOCK,PARAMS_N/2>>>(d_packed_dSp, d_Sp);
    DoDP4Av5s<<<BLOCK, PARAMS_N>>>((int16_t*) d_BBp, (short2*)d_A,d_packed_dSp);
#else        
    sa_plus_e_gpu<<<BLOCK, PARAMS_N>>>(d_BBp, d_A, d_Sp);  
#endif    
    unpack_gpu_decap<<<BLOCK,488>>>(d_B, 16, sk, 32, PARAMS_LOGQ, CRYPTO_BYTES + BYTES_SEED_A, PARAMS_N*PARAMS_NBAR, CRYPTO_SECRETKEYBYTES); 
    mul_add_sb_plus_e_gpu<<<BLOCK,PARAMS_NBAR>>>(d_W, d_B, d_Sp, d_Epp);
    key_encode_gpu<<<BLOCK, PARAMS_NBAR>>>(d_CC, (uint16_t*) d_G2in) ; // encode 
    add_gpu<<<BLOCK, PARAMS_NBAR*PARAMS_NBAR>>>(d_CC, d_W, d_CC);

    copy_vector<<<BLOCK, 1024>>>(d_Fin, d_ct, 0, 0, CRYPTO_CIPHERTEXTBYTES + CRYPTO_BYTES, CRYPTO_CIPHERTEXTBYTES, CRYPTO_CIPHERTEXTBYTES);
    // Reducing BBp modulo q
    reduce_q_gpu<<<BLOCK, PARAMS_N>>>(d_BBp);
    ct_verify_gpu<<<BLOCK, 1>>> (d_selector1, d_Bp, d_BBp, PARAMS_N*PARAMS_NBAR);
    ct_verify_gpu<<<BLOCK, 1>>> (d_selector2, d_C, d_CC, PARAMS_NBAR*PARAMS_NBAR);    
    or_gpu<<<BLOCK, 1>>>(d_selector, d_selector1, d_selector2, 1, 1, BLOCK);
    ct_select_gpu<<<BLOCK, CRYPTO_BYTES>>>((uint8_t *)d_Fin+CRYPTO_CIPHERTEXTBYTES, d_G2out+CRYPTO_BYTES, d_sk, d_selector);
    shake256_gpu<<<BLOCK,32>>>(d_ss, d_Fin, CRYPTO_CIPHERTEXTBYTES+CRYPTO_BYTES, CRYPTO_BYTES, CRYPTO_BYTES);   

    // cudaMemcpy(h_Fin, d_Fin, BLOCK* (CRYPTO_CIPHERTEXTBYTES + CRYPTO_BYTES) *sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(ss, d_ss, BLOCK* CRYPTO_BYTES*sizeof(uint8_t), cudaMemcpyDeviceToHost);
}
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop) ;
    printf("\nTotal decapsulation time: %.4f ms, TP: %.0f \n", elapsed, BLOCK*REPEAT*1000/elapsed);

    // printf("\n Fin\n"); for(k=0; k<BLOCK; k++) {printf("\n"); for(i=0; i<(CRYPTO_CIPHERTEXTBYTES + CRYPTO_BYTES); i++) printf("%x ", h_Fin[k*(CRYPTO_CIPHERTEXTBYTES + CRYPTO_BYTES) + i]);}
    // printf("\n ss\n"); for(k=0; k<10; k++) {printf("\n"); for(i=0; i<CRYPTO_BYTES; i++) printf("%x ", ss[k*CRYPTO_BYTES + i]);}
}
