#include "../include/util.cuh"


    // h_seedSE[0] = 0x96;
    // for(i=0; i<CRYPTO_BYTES; i++) h_seedSE[i+1] = h_G2out[CRYPTO_BYTES+i];

__global__ void copy_seedSE(uint8_t *out, uint8_t *in)
{
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;

    out[bid*(CRYPTO_BYTES+1) + 0] = 0x96;
    out[bid*(CRYPTO_BYTES+1) +tid+1] = in[bid*(2*CRYPTO_BYTES) + tid];
}
__global__ void copy_matrix_gpu(uint16_t *out, uint16_t *in)
{
    uint32_t tid = threadIdx.x, i;
    uint32_t bid = blockIdx.x;

    for(i=0; i<PARAMS_NBAR; i++)
        out[bid*PARAMS_N*PARAMS_NBAR + tid + i*PARAMS_N] = in[bid*2*PARAMS_N*PARAMS_NBAR + tid + i*PARAMS_N];
}

__global__ void copy_matrix_gpu_encap(uint16_t *out, uint16_t *in)
{
    uint32_t tid = threadIdx.x, i;
    uint32_t bid = blockIdx.x;

    for(i=0; i<PARAMS_NBAR; i++)
        out[bid*PARAMS_N*PARAMS_NBAR + tid + i*PARAMS_N] = in[bid*(2 * PARAMS_N + PARAMS_NBAR)*PARAMS_NBAR + tid + i*PARAMS_N];
}

__global__ void copy_matrix_gpu_encap_small(uint16_t *out, uint16_t *in)
{
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;

    out[bid*PARAMS_NBAR*PARAMS_NBAR + tid] = in[bid*((2 * PARAMS_N + PARAMS_NBAR)*PARAMS_NBAR) +tid];
}

__global__ void copy_matrix_gpu_x2(uint16_t *out, uint16_t *in)
{
    uint32_t tid = threadIdx.x, i;
    uint32_t bid = blockIdx.x;

    for(i=0; i<PARAMS_NBAR; i++)
        out[bid*PARAMS_N*PARAMS_NBAR + tid + i*PARAMS_N] = in[bid*2*PARAMS_N*PARAMS_NBAR +tid + i*PARAMS_N];
}
//TODO: Need to use different randonmess
__global__ void copy_sk_s(uint8_t *out, uint8_t *in)
{
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;

    out[bid*CRYPTO_SECRETKEYBYTES + tid] = in[tid];
}

__global__ void copy_sk_pk(uint8_t *out, uint8_t *in)
{
    uint32_t tid = threadIdx.x, i;
    uint32_t bid = blockIdx.x;
    uint32_t idxOut = bid*CRYPTO_SECRETKEYBYTES + CRYPTO_BYTES + tid;
    uint32_t idxIn = bid*CRYPTO_PUBLICKEYBYTES + tid;

    for(i=0; i<CRYPTO_PUBLICKEYBYTES/blockDim.x; i++)
        out[i*blockDim.x + idxOut] = in[i*blockDim.x +idxIn];

    // Remaining last chunk
    if(tid < (CRYPTO_PUBLICKEYBYTES - i*blockDim.x))
        out[i*blockDim.x + idxOut] = in[i*blockDim.x +idxIn];
}

__global__ void copy_sk_S(uint8_t *out, uint8_t *in)
{
    uint32_t tid = threadIdx.x, i;
    uint32_t bid = blockIdx.x;
    uint32_t idxOut = bid*CRYPTO_SECRETKEYBYTES + CRYPTO_BYTES + CRYPTO_PUBLICKEYBYTES + tid;
    uint32_t idxIn = bid*2*PARAMS_N*PARAMS_NBAR + tid;    

    for(i=0; i<2*PARAMS_N*PARAMS_NBAR/blockDim.x; i++)
        out[i*blockDim.x + idxOut] = in[i*blockDim.x + idxIn];
        // Remaining last chunk
    if(tid < (2*PARAMS_N*PARAMS_NBAR - i*blockDim.x))
        out[i*blockDim.x + idxOut] = in[i*blockDim.x +idxIn];
}

__global__ void copy_sk_pkh(uint8_t *out, uint8_t *in)
{
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t idxOut = bid*CRYPTO_SECRETKEYBYTES + CRYPTO_SECRETKEYBYTES - BYTES_PKHASH + tid;
    uint32_t idxIn = bid*BYTES_PKHASH + tid;    

    out[idxOut] = in[idxIn];
}

__global__ void copy_sk_pkh2(uint8_t *out, uint8_t *in)
{
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t idxOut = bid*CRYPTO_SECRETKEYBYTES + CRYPTO_SECRETKEYBYTES - BYTES_PKHASH + tid;
    uint32_t idxIn = bid*BYTES_PKHASH + tid;    

    out[idxOut] = in[idxIn];
}


__global__ void add_gpu(uint16_t *out, const uint16_t *a, const uint16_t *b) {
    // Add a and b
    // Inputs: a, b (N_BAR x N_BAR)
    // Output: c = a + b
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;

    // out[tid] = (a[tid] + b[tid]);// & ((1 << PARAMS_LOGQ) - 1);
    // if(tid<4) printf("%u %u = %u + %u\n", tid, out[tid], a[tid], b[tid]);
    out[bid*PARAMS_NBAR * PARAMS_NBAR + tid] = (a[bid*PARAMS_NBAR * PARAMS_NBAR + tid] + b[bid*PARAMS_NBAR * PARAMS_NBAR + tid]) & ((1 << PARAMS_LOGQ) - 1);
}

__global__ void sub_gpu(uint16_t *out, const uint16_t *a, const uint16_t *b) {
    // Add a and b
    // Inputs: a, b (N_BAR x N_BAR)
    // Output: c = a + b
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;

    out[bid*PARAMS_NBAR * PARAMS_NBAR + tid] = (a[bid*PARAMS_NBAR * PARAMS_NBAR + tid] - b[bid*PARAMS_NBAR * PARAMS_NBAR + tid]) & ((1 << PARAMS_LOGQ) - 1);
}

__global__ void copy_ct_encap(uint8_t *out, uint8_t *in, uint32_t in_offset, uint32_t out_offset, uint32_t in_stride,  uint32_t out_stride,  uint32_t len)
{
    uint32_t tid = threadIdx.x, i = 0;
    uint32_t bid = blockIdx.x;
    uint32_t idxOut = bid*out_stride + out_offset + tid;
    uint32_t idxIn = bid*in_stride + in_offset + tid;    
    
    for(i=0; i<len/blockDim.x; i++)
        out[i*blockDim.x +idxOut] = in[i*blockDim.x +idxIn];

        // Remaining last chunk
    if(tid < (len - i*blockDim.x))
        out[i*blockDim.x + idxOut] = in[i*blockDim.x +idxIn];
}

__global__ void copy_u8_2_u16(uint16_t * out, uint8_t * in, uint32_t out_offset, uint32_t in_offset, uint32_t out_stride,  uint32_t in_stride, uint32_t len)
{
    uint32_t tid = threadIdx.x, i=0;    
    uint32_t bid = blockIdx.x;
    uint32_t idxOut = bid*out_stride + out_offset + tid;
    uint32_t idxIn = bid*in_stride + in_offset + 2*tid;    


    for(i=0; i<len/blockDim.x; i++)
    {
        out[i*blockDim.x +idxOut] = in[i*2*blockDim.x +idxIn] | (in[i*2*blockDim.x +idxIn + 1]<<8);
    }
    //     // Remaining last chunk
    // if(tid < (len - i*blockDim.x))
    //     out[i*blockDim.x +idxOut] = in[i*blockDim.x +idxIn] | (in[i*blockDim.x +idxIn + 1]<<8);

    // for (size_t i = 0; i < PARAMS_N * PARAMS_NBAR; i++) {
    //         S[i] = sk_S[2 * i] | (sk_S[2 * i + 1] << 8);
    //     }    
}

__global__ void copy_vector(uint8_t * out, uint8_t * in, uint32_t out_offset, uint32_t in_offset, uint32_t out_stride,  uint32_t in_stride, uint32_t len)
{
    uint32_t tid = threadIdx.x, i=0;    
    uint32_t bid = blockIdx.x;
    uint32_t idxOut = bid*out_stride + out_offset + tid;
    uint32_t idxIn = bid*in_stride + in_offset + tid;    

    for(i=0; i<len/blockDim.x; i++)
        out[i*blockDim.x +idxOut] = in[i*blockDim.x +idxIn];
    
        // Remaining last chunk
    if(tid < (len - i*blockDim.x))
        out[i*blockDim.x +idxOut] = in[i*blockDim.x +idxIn];

}

__global__ void reduce_q_gpu(uint16_t *data)
{
    uint32_t tid = threadIdx.x;    
    uint32_t bid = blockIdx.x;   

    for (size_t i = 0; i < PARAMS_NBAR; i++) {
        data[bid*PARAMS_N*PARAMS_NBAR + i*PARAMS_N + tid] = data[bid*PARAMS_N*PARAMS_NBAR + i*PARAMS_N + tid] & ((1 << PARAMS_LOGQ) - 1);
    }
}

// Only use 1 thread to process
__global__ void ct_verify_gpu(int8_t *selector, const uint16_t *a, const uint16_t *b, size_t len) {
    // Compare two arrays in constant time.
    // selector = 0 if the byte arrays are equal, -1 otherwise.
    uint16_t r = 0;    
    uint32_t bid = blockIdx.x;    ;
    uint32_t idxIn = bid*len;    

    for (size_t i = 0; i < len; i++) {
        r |= a[idxIn + i] ^ b[idxIn + i];
    }

    r = (-(int16_t)r) >> (8 * sizeof(uint16_t) -1);
    selector[bid] = (int8_t)r;
    // printf("selector[%u]: %d\n", bid, selector[bid]);
}


__global__ void ct_select_gpu(uint8_t *r, const uint8_t *a, const uint8_t *b, int8_t *selector) {
    // Select one of the two input arrays to be moved to r
    // If (selector == 0) then load r with a, else if (selector == -1) load r with b
    uint32_t tid = threadIdx.x;    
    uint32_t bid = blockIdx.x;    

    r[bid*(CRYPTO_CIPHERTEXTBYTES + CRYPTO_BYTES) + tid] = (~selector[bid] & a[bid*2*CRYPTO_BYTES + tid]) | (selector[bid] & b[bid*CRYPTO_SECRETKEYBYTES + tid]);
}

__global__ void or_gpu(int8_t *r, int8_t *a, int8_t *b, uint32_t out_stride,  uint32_t in_stride, uint32_t len)
{
    uint32_t i=0;    
    uint32_t bid = blockIdx.x;

    for(i=0; i<len; i++)
        r[bid] = a[bid] | b[bid];

    // printf("r[%u]: %d\n", tid, r[tid]);
}
