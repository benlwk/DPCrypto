// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Include associated header file.
#include "../include/cuda_kernel.cuh"
#include "../include/params.h"
#include "../include/poly.cuh"
#include "../include/pack.cuh"
#include "../include/tv.h"

#include <stdio.h>
#include <stdint.h>
#define MOD256(X) ((X) & (256-1))
#define MOD65536(X) ((X) & (65536-1))


void saber_enc(uint8_t mode, uint8_t *h_k, uint8_t *h_pk, uint8_t *h_c)  {
    uint8_t *d_pk, *d_c, *d_m, *d_kr, *d_buf, *d_k, *d_A8;
    uint8_t *h_buf, *h_m, *h_A8;
    int i, j, k;
    uint16_t *h_sp, *h_bp, *h_b, *h_vp, *h_A; 
    uint16_t *d_A, *d_sp, *d_bp, *d_b, *d_vp; 
    uint16_t *d_mp; 

    short2 *d_packed_A1, *d_packed_A2;
    char4 *d_packed_b, *h_packed_b;
    cudaEvent_t start, stop, startIP, startMV, stopIP, stopMV;
    float elapsed, elapsedIP, elapsedMV;

    cudaEventCreate(&start);    cudaEventCreate(&stop);
    cudaEventCreate(&startIP);    cudaEventCreate(&stopIP);
    cudaEventCreate(&startMV);    cudaEventCreate(&stopMV);
    cudaMallocHost((void**) &h_buf, BATCH*64* sizeof(uint8_t));    
    cudaMallocHost((void**) &h_sp, BATCH*SABER_L*SABER_N* sizeof(uint16_t));    
    cudaMallocHost((void**) &h_bp, BATCH*SABER_L*SABER_N* sizeof(uint16_t));   
    cudaMallocHost((void**) &h_b, BATCH*SABER_L*SABER_N* sizeof(uint16_t));        
    cudaMallocHost((void**) &h_vp, BATCH*SABER_N* sizeof(uint16_t));  
    cudaMallocHost((void**) &h_m, BATCH*SABER_KEYBYTES* sizeof(uint8_t));

    cudaMalloc((void**) &d_buf, BATCH*64* sizeof(uint8_t));
    cudaMalloc((void**) &d_kr, BATCH*64* sizeof(uint8_t));
    cudaMalloc((void**) &d_k, BATCH*SABER_KEYBYTES* sizeof(uint8_t));    
    cudaMalloc((void**) &d_A, BATCH*SABER_L*SABER_L*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_sp, BATCH*SABER_L*SABER_N* sizeof(uint16_t));    
    cudaMalloc((void**) &d_bp, BATCH*SABER_L*SABER_N* sizeof(uint16_t));       
    cudaMalloc((void**) &d_b, BATCH*SABER_L*SABER_N* sizeof(uint16_t));       
    cudaMalloc((void**) &d_pk, BATCH*SABER_INDCPA_PUBLICKEYBYTES* sizeof(uint8_t));   
    cudaMalloc((void**) &d_packed_A1, BATCH*SABER_L*SABER_L*SABER_N* sizeof(short2));   
    cudaMalloc((void**) &d_packed_A2, BATCH*SABER_L*SABER_L*SABER_N* sizeof(short2));    
    cudaMalloc((void**) &d_packed_b, BATCH*SABER_L*SABER_N* sizeof(char4));    
    cudaMalloc((void**) &d_c, BATCH*SABER_BYTES_CCA_DEC* sizeof(uint8_t));     
    cudaMalloc((void**) &d_vp, BATCH*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_m, BATCH*SABER_KEYBYTES* sizeof(uint8_t));
    cudaMalloc((void**) &d_mp, BATCH*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_A8, BATCH*SABER_L * SABER_POLYVECBYTES* sizeof(uint8_t));       

    // Public and private key are from test vectors
    for(j=0; j<BATCH; j++) for(i=0; i<SABER_INDCPA_PUBLICKEYBYTES; i++) h_pk[j*SABER_INDCPA_PUBLICKEYBYTES + i] = pk[i];
    for(j=0; j<BATCH; j++) for(i=0; i<SABER_KEYBYTES; i++) h_m[j*SABER_KEYBYTES + i] = m_tv[i];      
    for(j=0; j<BATCH; j++) for(i=0; i<64; i++) h_buf[j*64 + i] = buf_tv[i];  // from randombytes()
    cudaEventRecord(start);         
    cudaMemcpy(d_buf, h_buf, BATCH*64 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pk, h_pk, BATCH*SABER_INDCPA_PUBLICKEYBYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, BATCH*SABER_KEYBYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);
 
    sha3_256_gpu<<<1,BATCH>>>(d_buf, d_buf, 32, 64, 64); 
    sha3_256_gpu<<<1,BATCH>>>(d_buf + 32, d_pk, SABER_INDCPA_PUBLICKEYBYTES, SABER_INDCPA_PUBLICKEYBYTES, 64); 
    sha3_512_gpu<<<1,BATCH>>>(d_kr, d_buf, 64);

    // start of indcpa_kem_enc
    GenSecret_gpu<<<1,BATCH>>>(d_sp, d_kr + 32);    
    // GenMatrix_gpu2<<<1,BATCH>>>(d_A, d_pk + SABER_POLYVECCOMPRESSEDBYTES);    // wklee, this is slow
    shake128_gpu<<<BATCH, 32>>>(d_A8, d_pk + SABER_POLYVECCOMPRESSEDBYTES, SABER_SEEDBYTES, SABER_L * SABER_POLYVECBYTES, SABER_L * SABER_POLYVECBYTES);
    BS2POLVECq_gpu2<<<BATCH, SABER_N/8>>>(d_A8, d_A);    
#ifdef MEAS_MV    
    cudaEventRecord(startMV);         
#endif      
if(mode==0) 
{   
    packdp2av2<<<BATCH,SABER_N/2>>>(d_packed_A1, d_packed_A2, d_A);
    packdp2b<<<BATCH,SABER_N/2>>>(d_packed_b, d_sp);
    DoDP4Av7<<<BATCH, SABER_N/2>>>(d_bp, d_packed_A1, d_packed_A2, d_packed_b);
}
else{
    MatVecMul_gpu_shared<<<BATCH, SABER_N>>>(d_bp, d_A, d_sp);     
}    
    post_process<<<BATCH, SABER_N>>>(d_bp);
    // // POLVECp2BS(ciphertext, bp);
    POLVECp2BS_gpu<<<BATCH, SABER_N / 4>>>(d_c, d_bp);
    BS2POLVECp_gpu<<<BATCH, SABER_N / 4>>>(d_pk, d_b, SABER_INDCPA_PUBLICKEYBYTES);
#ifdef MEAS_MV    
    cudaEventRecord(stopMV);      
    cudaEventSynchronize(stopMV);    
    cudaEventElapsedTime(&elapsedMV, startMV, stopMV);    
    if(mode==0)
        printf("Matrix-Vec: DPSaber %.4f ms TP %.0f /s\n", elapsedMV, BATCH*1000/elapsedMV);
    else
        printf("Matrix-Vec: INT32 %.4f ms TP %.0f /s\n", elapsedMV, BATCH*1000/elapsedMV);       
#endif 

#ifdef MEAS_IP    
    cudaEventRecord(startIP);         
#endif 
if(mode==0)
{
    //InnerProd(b, sp, vp);   
    packdp2av2Inner<<<BATCH,SABER_N/2>>>(d_packed_A1, d_packed_A2, d_b);
    packdp2b<<<BATCH,SABER_N/2>>>(d_packed_b, d_sp);
    DoDP4Av7_inner<<<BATCH, SABER_N/2>>>(d_vp, d_packed_A1, d_packed_A2, d_packed_b);     
}
else
{
    VecVecMul_gpu_shared<<<BATCH, SABER_N>>>(d_vp, d_b, d_sp);
}
#ifdef MEAS_IP    
    cudaEventRecord(stopIP);      
    cudaEventSynchronize(stopIP);    
    cudaEventElapsedTime(&elapsedIP, startIP, stopIP);    
    if(mode==0)
        printf("Inner Product: DPSaber %.4f ms TP %.0f /s\n", elapsedIP, BATCH*1000/elapsedIP);
    else
        printf("Inner Product: INT32 %.4f ms TP %.0f /s\n", elapsedIP, BATCH*1000/elapsedIP);       
#endif   

    BS2POLmsg_gpu<<<BATCH, SABER_KEYBYTES>>>(d_m, d_mp);
    post_process2<<<BATCH, SABER_N>>>(d_vp, d_mp);
    POLT2BS_gpu<<<BATCH, SABER_N / 2>>>(d_c+SABER_POLYVECCOMPRESSEDBYTES, d_vp);
    // end of indcpa_kem_enc

    sha3_256_gpu<<<1,BATCH>>>(d_kr + 32, d_c, SABER_BYTES_CCA_DEC, SABER_BYTES_CCA_DEC, 64); 
    sha3_256_gpu<<<1,BATCH>>>(d_k, d_kr, 64, 64, 32); 
       
    cudaMemcpy(h_c, d_c, BATCH*SABER_BYTES_CCA_DEC * sizeof(uint8_t), cudaMemcpyDeviceToHost);           
    cudaMemcpy(h_k, d_k, BATCH*SABER_KEYBYTES * sizeof(uint8_t), cudaMemcpyDeviceToHost);    
    cudaEventRecord(stop);      
    cudaEventSynchronize(stop);    
    cudaEventElapsedTime(&elapsed, start, stop);    
    if(mode==0)
        printf("Encap: DPSaber %.4f ms TP %.0f Encap/s\n", elapsed, BATCH*1000/elapsed);
    else
        printf("Encap: INT32 %.4f ms TP %.0f Encap/s\n", elapsed, BATCH*1000/elapsed);
#ifdef DEBUG
    printf("\n h_k:\n"); for(j=0; j<2; j++) {printf("\nbatch: %u\n", j); for(i=0; i<SABER_KEYBYTES; i++) printf("%u ", h_k[j*SABER_KEYBYTES + i]);}
    printf("\n h_c:\n"); for(k=0; k<2; k++) {printf("\nbatch %u\n", k); for(i=0; i<SABER_BYTES_CCA_DEC; i++) printf("%u ", h_c[k*SABER_BYTES_CCA_DEC + i]);}        
#endif
    
    cudaDeviceSynchronize();
    cudaFree(d_pk); cudaFree(d_c);  cudaFree(d_m);
    cudaFreeHost(h_sp);  cudaFreeHost(h_bp); 
    cudaFreeHost(h_b); cudaFreeHost(h_vp); 
}

void saber_dec(uint8_t mode, uint8_t *h_c, uint8_t *h_pk, uint8_t *h_sk, uint8_t *h_k) 
{
    uint8_t *h_m, *h_buf, *h_cCompare;
    uint8_t *d_sk, *d_c, *d_m, *d_kr, *d_buf, *d_pk, *d_cCompare, *d_k, *d_A8;
    uint16_t *h_s, *h_b, *h_v, *d_A;
    uint16_t *d_s, *d_b, *d_v, *d_cm, *d_sp, *d_bp, *d_vp, *d_mp;
    uint64_t *d_r;
    int i, j, k;

    short2 *d_packed_A1, *d_packed_A2;
    char4 *d_packed_b, *h_packed_b;
    cudaEvent_t start, stop;
    float elapsed;

    cudaEventCreate(&start);    cudaEventCreate(&stop);
    cudaMallocHost((void**) &h_s, BATCH*SABER_L*SABER_N* sizeof(uint16_t));
    cudaMallocHost((void**) &h_b, BATCH*SABER_L*SABER_N* sizeof(uint16_t));       
    cudaMallocHost((void**) &h_v, BATCH*SABER_N* sizeof(uint16_t));
    cudaMallocHost((void**) &h_m, BATCH*SABER_KEYBYTES* sizeof(uint8_t));
    cudaMallocHost((void**) &h_buf, BATCH*64* sizeof(uint8_t));  
    cudaMallocHost((void**) &h_cCompare, BATCH*SABER_BYTES_CCA_DEC* sizeof(uint8_t));

    cudaMalloc((void**) &d_A8, BATCH*SABER_L * SABER_POLYVECBYTES* sizeof(uint8_t));       
    cudaMalloc((void**) &d_r, BATCH*SABER_BYTES_CCA_DEC* sizeof(uint64_t));  
    cudaMalloc((void**) &d_vp, BATCH*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_mp, BATCH*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_pk, BATCH*SABER_INDCPA_PUBLICKEYBYTES* sizeof(uint8_t));   
    cudaMalloc((void**) &d_bp, BATCH*SABER_L*SABER_N* sizeof(uint16_t));  
    cudaMalloc((void**) &d_sp, BATCH*SABER_L*SABER_N* sizeof(uint16_t));     
    cudaMalloc((void**) &d_A, BATCH*SABER_L*SABER_L*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_sk, BATCH*SABER_SECRETKEYBYTES* sizeof(uint8_t));
    cudaMalloc((void**) &d_c, BATCH*SABER_BYTES_CCA_DEC* sizeof(uint8_t));
    cudaMalloc((void**) &d_cCompare, BATCH*SABER_BYTES_CCA_DEC* sizeof(uint8_t));    
    cudaMalloc((void**) &d_s, BATCH*SABER_L*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_b, BATCH*SABER_L*SABER_N* sizeof(uint16_t)); 
    cudaMalloc((void**) &d_v, BATCH*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_cm, BATCH*SABER_N* sizeof(uint16_t));
    cudaMalloc((void**) &d_packed_A1, BATCH*SABER_L*SABER_L*SABER_N* sizeof(short2));   
    cudaMalloc((void**) &d_packed_A2, BATCH*SABER_L*SABER_L*SABER_N* sizeof(short2));    
    cudaMalloc((void**) &d_packed_b, BATCH*SABER_L*SABER_N/2* sizeof(char4));     
    cudaMalloc((void**) &d_m, BATCH*SABER_KEYBYTES* sizeof(uint8_t));     
    cudaMalloc((void**) &d_buf, BATCH*64* sizeof(uint8_t));        
    cudaMalloc((void**) &d_kr, BATCH*64* sizeof(uint8_t));
    cudaMalloc((void**) &d_k, BATCH*SABER_KEYBYTES* sizeof(uint8_t));
    cudaMemset(d_m, 0, BATCH*SABER_KEYBYTES* sizeof(uint8_t)); 

        // Public and private key are from test vectors
    for(j=0; j<BATCH; j++) for(i=0; i<SABER_SECRETKEYBYTES; i++) h_sk[j*SABER_SECRETKEYBYTES + i] = sk_tv[i];
    for(j=0; j<BATCH; j++) for(i=0; i<SABER_INDCPA_PUBLICKEYBYTES; i++) h_pk[j*SABER_INDCPA_PUBLICKEYBYTES + i] = pk[i];
    
    cudaEventRecord(start);     
    cudaMemcpy(d_pk, h_pk, BATCH*SABER_INDCPA_PUBLICKEYBYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);  
    cudaMemcpy(d_s, h_s, BATCH*SABER_L*SABER_N * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, BATCH*SABER_L*SABER_N * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sk, h_sk, BATCH*SABER_SECRETKEYBYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, BATCH*SABER_BYTES_CCA_DEC * sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    // start of indcpa_kem_dec
    BS2POLVECq_gpu<<<BATCH, SABER_N/8>>>(d_sk, d_s);
    BS2POLVECp_gpu<<<BATCH, SABER_N / 4>>>(d_c, d_b, SABER_BYTES_CCA_DEC);

if(mode==0)
{
    //InnerProd(b, sp, vp);
    packdp2av2Inner<<<BATCH,SABER_N/2>>>( d_packed_A1,  d_packed_A2, d_b);
    packdp2b<<<BATCH,SABER_N/2>>>(d_packed_b, d_s);
    DoDP4Av7_inner<<<BATCH, SABER_N/2>>>(d_v,  d_packed_A1,  d_packed_A2, d_packed_b);
}
else{
    VecVecMul_gpu_shared<<<BATCH, SABER_N>>>(d_v, d_b, d_s);
}
    BS2POLT_gpu<<<BATCH, SABER_N / 2>>>(d_c + SABER_POLYVECCOMPRESSEDBYTES, d_cm);
    post_process3<<<BATCH, SABER_N >>>(d_v, d_cm);
    POLmsg2BS_gpu<<<BATCH, SABER_KEYBYTES >>>(d_m, d_v);
    // end of indcpa_kem_dec
  // Multitarget countermeasure for coins + contributory KEM
    copysk<<<BATCH, 32>>>(d_buf, d_m, d_sk);
    sha3_512_gpu<<<1,BATCH>>>(d_kr, d_buf, 64);

    // ************* start of indcpa_kem_enc *************
    // GenMatrix_gpu2<<<1,BATCH>>>(d_A, d_pk + SABER_POLYVECCOMPRESSEDBYTES);
    shake128_gpu<<<BATCH, 32>>>(d_A8, d_pk + SABER_POLYVECCOMPRESSEDBYTES, SABER_SEEDBYTES, SABER_L * SABER_POLYVECBYTES, SABER_L * SABER_POLYVECBYTES);
    BS2POLVECq_gpu2<<<BATCH, SABER_N/8>>>(d_A8, d_A);    
    GenSecret_gpu<<<1,BATCH>>>(d_sp, d_kr + 32);    
    
if(mode==0)    
{
    packdp2av2<<<BATCH,SABER_N/2>>>(d_packed_A1, d_packed_A2, d_A);
    packdp2b<<<BATCH,SABER_N/2>>>(d_packed_b, d_sp);
    DoDP4Av7<<<BATCH, SABER_N/2>>>(d_bp, d_packed_A1, d_packed_A2, d_packed_b);
}
else{
    MatVecMul_gpu_shared<<<BATCH, SABER_N>>>(d_bp, d_A, d_sp);     
}
    post_process<<<BATCH, SABER_N>>>(d_bp);    
    POLVECp2BS_gpu<<<BATCH, SABER_N / 4>>>(d_cCompare, d_bp);
    BS2POLVECp_gpu<<<BATCH, SABER_N / 4>>>(d_pk, d_b, SABER_INDCPA_PUBLICKEYBYTES);

if(mode==0)
{
    //InnerProd(b, sp, vp);
    packdp2av2Inner<<<BATCH,SABER_N/2>>>(d_packed_A1, d_packed_A2, d_b);
    packdp2b<<<BATCH,SABER_N/2>>>(d_packed_b, d_sp);
    DoDP4Av7_inner<<<BATCH, SABER_N/2>>>(d_vp, d_packed_A1, d_packed_A2, d_packed_b);
}
else{
    VecVecMul_gpu_shared<<<BATCH, SABER_N>>>(d_vp, d_b, d_sp);
}
    BS2POLmsg_gpu<<<BATCH, SABER_KEYBYTES>>>(d_m, d_mp);
    post_process2<<<BATCH, SABER_N>>>(d_vp, d_mp);
    POLT2BS_gpu<<<BATCH, SABER_N / 2>>>(d_cCompare+SABER_POLYVECCOMPRESSEDBYTES, d_vp);
    // ************* end of indcpa_kem_enc *************
    // printf("SABER_BYTES_CCA_DEC: %u\n", SABER_BYTES_CCA_DEC);
    verify_gpu<<<BATCH, SABER_N>>>(d_r, d_c, d_cCompare, SABER_BYTES_CCA_DEC);
    // overwrite coins in kr with h(c)
    sha3_256_gpu<<<1,BATCH>>>(d_kr + 32, d_c, SABER_BYTES_CCA_DEC, SABER_BYTES_CCA_DEC, 64); 
    // hash concatenation of pre-k and h(c) to k
    sha3_256_gpu<<<1,BATCH>>>(d_k, d_kr, 64, 64, SABER_KEYBYTES); 
    cmov_gpu<<<BATCH,SABER_N>>>(d_kr, d_sk+ SABER_SECRETKEYBYTES - SABER_KEYBYTES, SABER_KEYBYTES, d_r);

    cudaMemcpy(h_m, d_k, BATCH*SABER_KEYBYTES * sizeof(uint8_t), cudaMemcpyDeviceToHost);       

    cudaEventRecord(stop);      
    cudaEventSynchronize(stop);    
    cudaEventElapsedTime(&elapsed, start, stop);    
    if(mode==0)
        printf("Decap: DPSaber %.4f ms TP %.0f Decap/s\n", elapsed, BATCH*1000/elapsed);
    else
        printf("Decap: INT32 %.4f ms TP %.0f Decap/s\n", elapsed, BATCH*1000/elapsed);    
#ifdef DEBUG
    printf("\n h_m:\n"); for(k=0; k<2; k++) {printf("\nbatch %u\n", k); for(i=0; i<SABER_KEYBYTES; i++) printf("%u ", h_m[k*SABER_KEYBYTES + i]);}    
    // Functional verification: check if k_a == k_b?
    for(j=0; j<BATCH; j++)
    {
        for(i=0; i<SABER_KEYBYTES; i++)
        {
            if(h_m[j*SABER_KEYBYTES + i]!=h_k[j*SABER_KEYBYTES + i]){
                printf("wrong at batch %u element %u: %u %u\n", j, i, h_m[j*SABER_BYTES_CCA_DEC + i], h_k[j*SABER_KEYBYTES + i]);
                break;
            }
        }
    }
#endif
    cudaFreeHost(h_s); cudaFreeHost(h_b);  cudaFreeHost(h_v);     
}