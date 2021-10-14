// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// Include associated header file.
#include "../include/cuda_kernel.cuh"
#include "../include/api.h"
#include "../include/params.h"

// //TODO: optimized with tiling?
__global__ void as_plus_e_gpu(uint16_t *out, int16_t *A, uint16_t *s)
{
    uint16_t sum = 0;
    uint32_t tid = threadIdx.x, j, k, bid = blockIdx.x;
    __shared__ uint16_t s_A[PARAMS_N];    
    __shared__ uint16_t s_out[PARAMS_N* PARAMS_NBAR];    

    for (k = 0; k < PARAMS_NBAR; k++)
        s_out[k * PARAMS_N + tid] = out[bid * PARAMS_N* PARAMS_NBAR + k * PARAMS_N + tid];
    __syncthreads();
    for (j = 0; j < PARAMS_N; j++) 
    {
        s_A[tid] =  A[bid * PARAMS_N* PARAMS_N + tid * PARAMS_N + j];        
        // __syncthreads();        
        for (k = 0; k < PARAMS_NBAR; k++)
        {         
            // out[tid * PARAMS_NBAR + k] += A[tid * PARAMS_N + j] * s[k * PARAMS_N + j];            
            // out[tid * PARAMS_NBAR + k] += s_A[tid] * s_s[k * PARAMS_N + j];
            s_out[tid * PARAMS_NBAR + k] += s_A[tid] * s[k * PARAMS_N + j];     
        }   
        __syncthreads();
    }    
    for (k = 0; k < PARAMS_NBAR; k++)
        out[bid * PARAMS_N* PARAMS_NBAR + k * PARAMS_N + tid] = s_out[k * PARAMS_N + tid];
}

// __global__ void as_plus_e_gpu(uint16_t *out, int16_t *A, uint16_t *s)
// {
//     uint16_t sum = 0;
//     uint32_t tid = threadIdx.x, j, k;
//     // __shared__ uint16_t s_s[PARAMS_N];

//     for (k = 0; k < PARAMS_NBAR; k++) {
//         sum = 0;       
        
//         for (j = 0; j < PARAMS_N; j++) 
//             sum += A[tid * PARAMS_N + j] * s[k * PARAMS_N + j];
//             // sum += A[tid * PARAMS_N + j] * s_s[j];
//         out[tid * PARAMS_NBAR + k] += sum;  // Adding e. No need to reduce modulo 2^15, extra bits are taken care of during packing later on.
//     }    
// }

// // working, faster
// __global__ void sa_plus_e_gpu(uint16_t *out, int16_t *A, uint16_t *s)
// {
//     uint16_t sum = 0;
//     uint32_t tid = threadIdx.x, j, k, bid = blockIdx.x;
//     __shared__ int16_t s_A[PARAMS_N];

//     // Matrix multiplication-addition s*A + e
//     for (j = 0; j < PARAMS_N; j++) {
//         s_A[tid] = A[bid * PARAMS_N* PARAMS_N + j * PARAMS_N + tid];
//         for (k = 0; k < PARAMS_NBAR; k++) {        
//             sum = out[bid * PARAMS_N* PARAMS_NBAR + k * PARAMS_N + tid];        
//             sum += s_A[tid] * s[bid*PARAMS_N*PARAMS_NBAR + k * PARAMS_N + j];
//             out[bid*PARAMS_N*PARAMS_NBAR +k * PARAMS_N + tid] = sum;
//             // __syncthreads();
//         }
//     }
// }

// working, faster, unrolled 
__global__ void sa_plus_e_gpu(uint16_t *out, int16_t *A, uint16_t *s)
{
    uint32_t sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;
    uint32_t tid = threadIdx.x, j, k, bid = blockIdx.x;    
    int16_t load_a;

    sum0 = out[bid * PARAMS_N* PARAMS_NBAR + tid];  
    sum1 = out[bid * PARAMS_N* PARAMS_NBAR + PARAMS_N + tid];  
    sum2 = out[bid * PARAMS_N* PARAMS_NBAR + 2*PARAMS_N + tid];  
    sum3 = out[bid * PARAMS_N* PARAMS_NBAR + 3*PARAMS_N + tid];  
    sum4 = out[bid * PARAMS_N* PARAMS_NBAR + 4*PARAMS_N + tid];  
    sum5 = out[bid * PARAMS_N* PARAMS_NBAR + 5*PARAMS_N + tid];  
    sum6 = out[bid * PARAMS_N* PARAMS_NBAR + 6*PARAMS_N + tid];  
    sum7 = out[bid * PARAMS_N* PARAMS_NBAR + 7*PARAMS_N + tid];      
// #pragma unroll
    // Matrix multiplication-addition s*A + e
    for (j = 0; j < PARAMS_N; j++) {
        load_a = A[bid * PARAMS_N* PARAMS_N + j * PARAMS_N + tid];
        sum0 += load_a * s[bid*PARAMS_N*PARAMS_NBAR + j];
        sum1 += load_a * s[bid*PARAMS_N*PARAMS_NBAR + PARAMS_N + j];        
        sum2 += load_a * s[bid*PARAMS_N*PARAMS_NBAR + 2*PARAMS_N + j];
        sum3 += load_a * s[bid*PARAMS_N*PARAMS_NBAR + 3*PARAMS_N + j];  
        sum4 += load_a * s[bid*PARAMS_N*PARAMS_NBAR + 4*PARAMS_N + j];
        sum5 += load_a * s[bid*PARAMS_N*PARAMS_NBAR + 5*PARAMS_N + j];        
        sum6 += load_a * s[bid*PARAMS_N*PARAMS_NBAR + 6*PARAMS_N + j];
        sum7 += load_a * s[bid*PARAMS_N*PARAMS_NBAR + 7*PARAMS_N + j];  
    }

    out[bid*PARAMS_N*PARAMS_NBAR + tid] = sum0;  
    out[bid*PARAMS_N*PARAMS_NBAR + PARAMS_N + tid] = sum1;  
    out[bid*PARAMS_N*PARAMS_NBAR + 2*PARAMS_N + tid] = sum2;  
    out[bid*PARAMS_N*PARAMS_NBAR + 3*PARAMS_N + tid] = sum3;      
    out[bid*PARAMS_N*PARAMS_NBAR + 4*PARAMS_N + tid] = sum4;  
    out[bid*PARAMS_N*PARAMS_NBAR + 5*PARAMS_N + tid] = sum5;  
    out[bid*PARAMS_N*PARAMS_NBAR + 6*PARAMS_N + tid] = sum6;  
    out[bid*PARAMS_N*PARAMS_NBAR + 7*PARAMS_N + tid] = sum7;   
}

#ifdef DPFRO

__global__ void packdp2bv2(char4 *outb, uint16_t* b) 
{
    int i;
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    char temp1;

    for(i=0; i<PARAMS_NBAR; i++)
    {
        temp1 = b[bid * PARAMS_N * PARAMS_NBAR + i*PARAMS_N+2*tid];
        if(temp1==65535)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].x = -1;
        else if(temp1==65534)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].x = -2;
        else if(temp1==65533)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].x = -3;
        else if(temp1==65532)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].x = -4;
        else if(temp1==65531)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].x = -5;
        else if(temp1==65530)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].x = -6;
        else if(temp1==65529)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].x = -7;
        else if(temp1==65528)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].x = -8;
        else if(temp1==65527)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].x = -9;
        else if(temp1==65526)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].x = -10;
        else if(temp1==65525)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].x = -11;
        else if(temp1==65524)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].x = -12;
        else 
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].x = temp1;

        temp1 = b[bid * PARAMS_N * PARAMS_NBAR + i*PARAMS_N+2*tid + 1];
        if(temp1==65535)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].y = -1;
        else if(temp1==65534)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].y = -2;
        else if(temp1==65533)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].y = -3;
        else if(temp1==65532)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].y = -4;
        else if(temp1==65531)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].y = -5;
        else if(temp1==65530)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].y = -6;
        else if(temp1==65529)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].y = -7;
        else if(temp1==65528)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].y = -8;
        else if(temp1==65527)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].y = -9;
        else if(temp1==65526)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].y = -10;
        else if(temp1==65525)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].y = -11;
        else if(temp1==65524)
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].y = -12;
        else 
            outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].y = temp1;        
        outb[bid * PARAMS_N * PARAMS_NBAR/2 + i*PARAMS_N/2+tid].y = temp1;
    }    
}

// signed version of DoDP4Av5
__global__ void DoDP4Av5s(int16_t *out, short2 *a, char4* b) {
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    int i, j, k;
    int16_t sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0,sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;
    short2 load_a;    
    __shared__ char4 s_b[PARAMS_N*PARAMS_NBAR];
    for (k = 0; k < PARAMS_NBAR/2; k++)     
        s_b[k*PARAMS_N + tid] = b[bid*PARAMS_N*PARAMS_NBAR/2 + k*PARAMS_N + tid];
    __syncthreads();

    sum0 = out[bid*PARAMS_N*PARAMS_NBAR + tid];
    sum1 = out[bid*PARAMS_N*PARAMS_NBAR + PARAMS_N + tid];
    sum2 = out[bid*PARAMS_N*PARAMS_NBAR + 2*PARAMS_N + tid];
    sum3 = out[bid*PARAMS_N*PARAMS_NBAR + 3*PARAMS_N + tid];
    sum4 = out[bid*PARAMS_N*PARAMS_NBAR + 4*PARAMS_N + tid];
    sum5 = out[bid*PARAMS_N*PARAMS_NBAR + 5*PARAMS_N + tid];
    sum6 = out[bid*PARAMS_N*PARAMS_NBAR + 6*PARAMS_N + tid];
    sum7 = out[bid*PARAMS_N*PARAMS_NBAR + 7*PARAMS_N + tid];    

        // Matrix multiplication-addition s*A + e
    for (j = 0; j < PARAMS_N/2; j++) {                
        if(tid%2==0)// warp divergence, can we solve this???
        {
            load_a.x = a[bid*PARAMS_N*PARAMS_N/2 + j*PARAMS_N + tid/2].x;    
            load_a.y = a[bid*PARAMS_N*PARAMS_N/2 + j*PARAMS_N + PARAMS_N/2 +tid/2].x;
        }
        else
        {
            load_a.x = a[bid*PARAMS_N*PARAMS_N/2 + j*PARAMS_N + tid/2].y;    
            load_a.y = a[bid*PARAMS_N*PARAMS_N/2 + j*PARAMS_N + PARAMS_N/2 +tid/2].y;
        }
        // sum0 = __dp2a_lo(load_a, s_b[j], sum0);           
        // sum1 = __dp2a_lo(load_a, s_b[PARAMS_N/2 + j], sum1);           
        // sum2 = __dp2a_lo(load_a, s_b[2*PARAMS_N/2 + j], sum2);           
        // sum3 = __dp2a_lo(load_a, s_b[3*PARAMS_N/2 + j], sum3);      
        // sum4 = __dp2a_lo(load_a, s_b[4*PARAMS_N/2 + j], sum4);           
        // sum5 = __dp2a_lo(load_a, s_b[5*PARAMS_N/2 + j], sum5);           
        // sum6 = __dp2a_lo(load_a, s_b[6PARAMS_N/2 + j], sum6);           
        // sum7 = __dp2a_lo(load_a, s_b[7*PARAMS_N/2 + j], sum7);  

        sum0 += load_a.x*s_b[j].x + load_a.y * s_b[j].y;
        sum1 += load_a.x*s_b[PARAMS_N/2 + j].x + load_a.y * s_b[PARAMS_N/2 + j].y;
        sum2 += load_a.x*s_b[2*PARAMS_N/2 + j].x + load_a.y * s_b[2*PARAMS_N/2 + j].y;
        sum3 += load_a.x*s_b[3*PARAMS_N/2 + j].x + load_a.y * s_b[3*PARAMS_N/2 + j].y;
        sum4 += load_a.x*s_b[4*PARAMS_N/2 + j].x + load_a.y * s_b[4*PARAMS_N/2 + j].y;
        sum5 += load_a.x*s_b[5*PARAMS_N/2 + j].x + load_a.y * s_b[5*PARAMS_N/2 + j].y;
        sum6 += load_a.x*s_b[6*PARAMS_N/2 + j].x + load_a.y * s_b[6*PARAMS_N/2 + j].y;
        sum7 += load_a.x*s_b[7*PARAMS_N/2 + j].x + load_a.y * s_b[7*PARAMS_N/2 + j].y; 
        // if(tid==0 ) printf("tid: %u j: %u a: %d %d b: %d %d sum: %d\n", tid, j, load_a.x, load_a.y, s_b[j].x, s_b[j].y, sum0);
    }
    out[bid*PARAMS_N*PARAMS_NBAR + tid] = sum0;
    out[bid*PARAMS_N*PARAMS_NBAR + PARAMS_N +tid] = sum1;
    out[bid*PARAMS_N*PARAMS_NBAR + 2*PARAMS_N + tid] = sum2;
    out[bid*PARAMS_N*PARAMS_NBAR + 3*PARAMS_N + tid] = sum3;
    out[bid*PARAMS_N*PARAMS_NBAR + 4*PARAMS_N + tid] = sum4;
    out[bid*PARAMS_N*PARAMS_NBAR + 5*PARAMS_N + tid] = sum5;
    out[bid*PARAMS_N*PARAMS_NBAR + 6*PARAMS_N + tid] = sum6;
    out[bid*PARAMS_N*PARAMS_NBAR + 7*PARAMS_N + tid] = sum7;  
}

#endif

__global__ void mul_add_sb_plus_e_gpu(uint16_t *out, const uint16_t *b, const uint16_t *s, const uint16_t *e) {
    // Multiply by s on the left
    // Inputs: b (N x N_BAR), s (N_BAR x N), e (N_BAR x N_BAR)
    // Output: out = s*b + e (N_BAR x N_BAR)
    uint32_t tid = threadIdx.x, j, k, bid = blockIdx.x;

    for (k = 0; k < PARAMS_NBAR; k++) {
        // for (i = 0; i < PARAMS_NBAR; i++) {
            out[bid*PARAMS_NBAR*PARAMS_NBAR + k * PARAMS_NBAR + tid] = e[bid*PARAMS_NBAR * PARAMS_NBAR + k * PARAMS_NBAR + tid];
            for (j = 0; j < PARAMS_N; j++) {
                out[bid*PARAMS_NBAR*PARAMS_NBAR + k * PARAMS_NBAR + tid] += s[bid*PARAMS_N * PARAMS_NBAR + k * PARAMS_N + j] * b[bid*PARAMS_N * PARAMS_NBAR + j * PARAMS_NBAR + tid];
            }
            out[bid*PARAMS_NBAR*PARAMS_NBAR + k * PARAMS_NBAR + tid] = (uint32_t)(out[bid*PARAMS_NBAR*PARAMS_NBAR + k * PARAMS_NBAR + tid]) & ((1 << PARAMS_LOGQ) - 1);
        // }
    }
}

__global__ void mul_bs_gpu(uint16_t *out, const uint16_t *b, const uint16_t *s) {
    // Multiply by s on the right
    // Inputs: b (N_BAR x N), s (N x N_BAR)
    // Output: out = b*s (N_BAR x N_BAR)
    int i, j, k;
    uint32_t tid = threadIdx.x, bid = blockIdx.x;

    for (i = 0; i < PARAMS_NBAR; i++) {
        // for (j = 0; j < PARAMS_NBAR; j++) {
            out[i * PARAMS_NBAR + tid] = 0;
            for (k = 0; k < PARAMS_N; k++) {
                out[bid*PARAMS_NBAR*PARAMS_NBAR + i * PARAMS_NBAR + tid] += b[bid*PARAMS_N*PARAMS_NBAR + i * PARAMS_N + k] * s[bid*PARAMS_N*PARAMS_NBAR +tid * PARAMS_N + k];
            }
            out[bid*PARAMS_NBAR*PARAMS_NBAR + i * PARAMS_NBAR + tid] = (uint32_t)(out[bid*PARAMS_NBAR*PARAMS_NBAR + i * PARAMS_NBAR + tid]) & ((1 << PARAMS_LOGQ) - 1);
        // }
    }
}
