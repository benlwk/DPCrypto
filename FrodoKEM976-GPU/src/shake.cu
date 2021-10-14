#include "../include/shake.cuh"

/*************************************************
 * Name:        load64
 *
 * Description: Load 8 bytes into uint64_t in little-endian order
 *
 * Arguments:   - const uint8_t *x: pointer to input byte array
 *
 * Returns the loaded 64-bit unsigned integer
 **************************************************/
__host__ __device__ static uint64_t load64(const uint8_t *x) {
    uint64_t r = 0;
    for (size_t i = 0; i < 8; ++i) {
        r |= (uint64_t)x[i] << 8 * i;
    }

    return r;
}

/*************************************************
 * Name:        store64
 *
 * Description: Store a 64-bit integer to a byte array in little-endian order
 *
 * Arguments:   - uint8_t *x: pointer to the output byte array
 *              - uint64_t u: input 64-bit unsigned integer
 **************************************************/
__host__ __device__ static void store64(uint8_t *x, uint64_t u) {
    for (size_t i = 0; i < 8; ++i) {
        x[i] = (uint8_t) (u >> 8 * i);
    }
}



__constant__ uint64_t KeccakF_RoundConstants[NROUNDS] = 
{
    (uint64_t)0x0000000000000001ULL,
    (uint64_t)0x0000000000008082ULL,
    (uint64_t)0x800000000000808aULL,
    (uint64_t)0x8000000080008000ULL,
    (uint64_t)0x000000000000808bULL,
    (uint64_t)0x0000000080000001ULL,
    (uint64_t)0x8000000080008081ULL,
    (uint64_t)0x8000000000008009ULL,
    (uint64_t)0x000000000000008aULL,
    (uint64_t)0x0000000000000088ULL,
    (uint64_t)0x0000000080008009ULL,
    (uint64_t)0x000000008000000aULL,
    (uint64_t)0x000000008000808bULL,
    (uint64_t)0x800000000000008bULL,
    (uint64_t)0x8000000000008089ULL,
    (uint64_t)0x8000000000008003ULL,
    (uint64_t)0x8000000000008002ULL,
    (uint64_t)0x8000000000000080ULL,
    (uint64_t)0x000000000000800aULL,
    (uint64_t)0x800000008000000aULL,
    (uint64_t)0x8000000080008081ULL,
    (uint64_t)0x8000000000008080ULL,
    (uint64_t)0x0000000080000001ULL,
    (uint64_t)0x8000000080008008ULL
};


__device__ void KeccakF1600_StatePermute(uint64_t * state)
{
  int round;

        uint64_t Aba, Abe, Abi, Abo, Abu;
        uint64_t Aga, Age, Agi, Ago, Agu;
        uint64_t Aka, Ake, Aki, Ako, Aku;
        uint64_t Ama, Ame, Ami, Amo, Amu;
        uint64_t Asa, Ase, Asi, Aso, Asu;
        uint64_t BCa, BCe, BCi, BCo, BCu;
        uint64_t Da, De, Di, Do, Du;
        uint64_t Eba, Ebe, Ebi, Ebo, Ebu;
        uint64_t Ega, Ege, Egi, Ego, Egu;
        uint64_t Eka, Eke, Eki, Eko, Eku;
        uint64_t Ema, Eme, Emi, Emo, Emu;
        uint64_t Esa, Ese, Esi, Eso, Esu;

        //copyFromState(A, state)
        Aba = state[ 0];
        Abe = state[ 1];
        Abi = state[ 2];
        Abo = state[ 3];
        Abu = state[ 4];
        Aga = state[ 5];
        Age = state[ 6];
        Agi = state[ 7];
        Ago = state[ 8];
        Agu = state[ 9];
        Aka = state[10];
        Ake = state[11];
        Aki = state[12];
        Ako = state[13];
        Aku = state[14];
        Ama = state[15];
        Ame = state[16];
        Ami = state[17];
        Amo = state[18];
        Amu = state[19];
        Asa = state[20];
        Ase = state[21];
        Asi = state[22];
        Aso = state[23];
        Asu = state[24];

        for( round = 0; round < NROUNDS; round += 2 )
        {
            //    prepareTheta
            BCa = Aba^Aga^Aka^Ama^Asa;
            BCe = Abe^Age^Ake^Ame^Ase;
            BCi = Abi^Agi^Aki^Ami^Asi;
            BCo = Abo^Ago^Ako^Amo^Aso;
            BCu = Abu^Agu^Aku^Amu^Asu;

            //thetaRhoPiChiIotaPrepareTheta(round  , A, E)
            Da = BCu^ROL(BCe, 1);
            De = BCa^ROL(BCi, 1);
            Di = BCe^ROL(BCo, 1);
            Do = BCi^ROL(BCu, 1);
            Du = BCo^ROL(BCa, 1);

            Aba ^= Da;
            BCa = Aba;
            Age ^= De;
            BCe = ROL(Age, 44);
            Aki ^= Di;
            BCi = ROL(Aki, 43);
            Amo ^= Do;
            BCo = ROL(Amo, 21);
            Asu ^= Du;
            BCu = ROL(Asu, 14);
            Eba =   BCa ^((~BCe)&  BCi );
            Eba ^= (uint64_t)KeccakF_RoundConstants[round];
            Ebe =   BCe ^((~BCi)&  BCo );
            Ebi =   BCi ^((~BCo)&  BCu );
            Ebo =   BCo ^((~BCu)&  BCa );
            Ebu =   BCu ^((~BCa)&  BCe );

            Abo ^= Do;
            BCa = ROL(Abo, 28);
            Agu ^= Du;
            BCe = ROL(Agu, 20);
            Aka ^= Da;
            BCi = ROL(Aka,  3);
            Ame ^= De;
            BCo = ROL(Ame, 45);
            Asi ^= Di;
            BCu = ROL(Asi, 61);
            Ega =   BCa ^((~BCe)&  BCi );
            Ege =   BCe ^((~BCi)&  BCo );
            Egi =   BCi ^((~BCo)&  BCu );
            Ego =   BCo ^((~BCu)&  BCa );
            Egu =   BCu ^((~BCa)&  BCe );

            Abe ^= De;
            BCa = ROL(Abe,  1);
            Agi ^= Di;
            BCe = ROL(Agi,  6);
            Ako ^= Do;
            BCi = ROL(Ako, 25);
            Amu ^= Du;
            BCo = ROL(Amu,  8);
            Asa ^= Da;
            BCu = ROL(Asa, 18);
            Eka =   BCa ^((~BCe)&  BCi );
            Eke =   BCe ^((~BCi)&  BCo );
            Eki =   BCi ^((~BCo)&  BCu );
            Eko =   BCo ^((~BCu)&  BCa );
            Eku =   BCu ^((~BCa)&  BCe );

            Abu ^= Du;
            BCa = ROL(Abu, 27);
            Aga ^= Da;
            BCe = ROL(Aga, 36);
            Ake ^= De;
            BCi = ROL(Ake, 10);
            Ami ^= Di;
            BCo = ROL(Ami, 15);
            Aso ^= Do;
            BCu = ROL(Aso, 56);
            Ema =   BCa ^((~BCe)&  BCi );
            Eme =   BCe ^((~BCi)&  BCo );
            Emi =   BCi ^((~BCo)&  BCu );
            Emo =   BCo ^((~BCu)&  BCa );
            Emu =   BCu ^((~BCa)&  BCe );

            Abi ^= Di;
            BCa = ROL(Abi, 62);
            Ago ^= Do;
            BCe = ROL(Ago, 55);
            Aku ^= Du;
            BCi = ROL(Aku, 39);
            Ama ^= Da;
            BCo = ROL(Ama, 41);
            Ase ^= De;
            BCu = ROL(Ase,  2);
            Esa =   BCa ^((~BCe)&  BCi );
            Ese =   BCe ^((~BCi)&  BCo );
            Esi =   BCi ^((~BCo)&  BCu );
            Eso =   BCo ^((~BCu)&  BCa );
            Esu =   BCu ^((~BCa)&  BCe );

            //    prepareTheta
            BCa = Eba^Ega^Eka^Ema^Esa;
            BCe = Ebe^Ege^Eke^Eme^Ese;
            BCi = Ebi^Egi^Eki^Emi^Esi;
            BCo = Ebo^Ego^Eko^Emo^Eso;
            BCu = Ebu^Egu^Eku^Emu^Esu;

            //thetaRhoPiChiIotaPrepareTheta(round+1, E, A)
            Da = BCu^ROL(BCe, 1);
            De = BCa^ROL(BCi, 1);
            Di = BCe^ROL(BCo, 1);
            Do = BCi^ROL(BCu, 1);
            Du = BCo^ROL(BCa, 1);

            Eba ^= Da;
            BCa = Eba;
            Ege ^= De;
            BCe = ROL(Ege, 44);
            Eki ^= Di;
            BCi = ROL(Eki, 43);
            Emo ^= Do;
            BCo = ROL(Emo, 21);
            Esu ^= Du;
            BCu = ROL(Esu, 14);
            Aba =   BCa ^((~BCe)&  BCi );
            Aba ^= (uint64_t)KeccakF_RoundConstants[round+1];
            Abe =   BCe ^((~BCi)&  BCo );
            Abi =   BCi ^((~BCo)&  BCu );
            Abo =   BCo ^((~BCu)&  BCa );
            Abu =   BCu ^((~BCa)&  BCe );

            Ebo ^= Do;
            BCa = ROL(Ebo, 28);
            Egu ^= Du;
            BCe = ROL(Egu, 20);
            Eka ^= Da;
            BCi = ROL(Eka, 3);
            Eme ^= De;
            BCo = ROL(Eme, 45);
            Esi ^= Di;
            BCu = ROL(Esi, 61);
            Aga =   BCa ^((~BCe)&  BCi );
            Age =   BCe ^((~BCi)&  BCo );
            Agi =   BCi ^((~BCo)&  BCu );
            Ago =   BCo ^((~BCu)&  BCa );
            Agu =   BCu ^((~BCa)&  BCe );

            Ebe ^= De;
            BCa = ROL(Ebe, 1);
            Egi ^= Di;
            BCe = ROL(Egi, 6);
            Eko ^= Do;
            BCi = ROL(Eko, 25);
            Emu ^= Du;
            BCo = ROL(Emu, 8);
            Esa ^= Da;
            BCu = ROL(Esa, 18);
            Aka =   BCa ^((~BCe)&  BCi );
            Ake =   BCe ^((~BCi)&  BCo );
            Aki =   BCi ^((~BCo)&  BCu );
            Ako =   BCo ^((~BCu)&  BCa );
            Aku =   BCu ^((~BCa)&  BCe );

            Ebu ^= Du;
            BCa = ROL(Ebu, 27);
            Ega ^= Da;
            BCe = ROL(Ega, 36);
            Eke ^= De;
            BCi = ROL(Eke, 10);
            Emi ^= Di;
            BCo = ROL(Emi, 15);
            Eso ^= Do;
            BCu = ROL(Eso, 56);
            Ama =   BCa ^((~BCe)&  BCi );
            Ame =   BCe ^((~BCi)&  BCo );
            Ami =   BCi ^((~BCo)&  BCu );
            Amo =   BCo ^((~BCu)&  BCa );
            Amu =   BCu ^((~BCa)&  BCe );

            Ebi ^= Di;
            BCa = ROL(Ebi, 62);
            Ego ^= Do;
            BCe = ROL(Ego, 55);
            Eku ^= Du;
            BCi = ROL(Eku, 39);
            Ema ^= Da;
            BCo = ROL(Ema, 41);
            Ese ^= De;
            BCu = ROL(Ese, 2);
            Asa =   BCa ^((~BCe)&  BCi );
            Ase =   BCe ^((~BCi)&  BCo );
            Asi =   BCi ^((~BCo)&  BCu );
            Aso =   BCo ^((~BCu)&  BCa );
            Asu =   BCu ^((~BCa)&  BCe );
        }

        //copyToState(state, A)
        state[ 0] = Aba;
        state[ 1] = Abe;
        state[ 2] = Abi;
        state[ 3] = Abo;
        state[ 4] = Abu;
        state[ 5] = Aga;
        state[ 6] = Age;
        state[ 7] = Agi;
        state[ 8] = Ago;
        state[ 9] = Agu;
        state[10] = Aka;
        state[11] = Ake;
        state[12] = Aki;
        state[13] = Ako;
        state[14] = Aku;
        state[15] = Ama;
        state[16] = Ame;
        state[17] = Ami;
        state[18] = Amo;
        state[19] = Amu;
        state[20] = Asa;
        state[21] = Ase;
        state[22] = Asi;
        state[23] = Aso;
        state[24] = Asu;

        #undef    round
}

#include <string.h>
#define MIN(a, b) ((a) < (b) ? (a) : (b))


__device__ static void keccak_absorb_ser(uint64_t *s, unsigned int r, const unsigned char *m, unsigned long long int mlen, unsigned char p)
{
  unsigned long long i;
  unsigned char t[200];
 
  while (mlen >= r) 
  {
    for (i = 0; i < r / 8; ++i)
      s[i] ^= load64(m + 8 * i);
    
    KeccakF1600_StatePermute(s);
    mlen -= r;
    m += r;
  }

  for (i = 0; i < r; ++i)
    t[i] = 0;
  for (i = 0; i < mlen; ++i)
    t[i] = m[i];
  t[i] = p;
  t[r - 1] |= 128;
  for (i = 0; i < r / 8; ++i)
    s[i] ^= load64(t + 8 * i);
}


__device__ static void keccak_squeezeblocks_ser(unsigned char *h, unsigned long long int nblocks, uint64_t *s, unsigned int r)
{
  unsigned int i;

  while(nblocks > 0) 
  {
    KeccakF1600_StatePermute(s);
    for (i = 0; i < (r>>3); i++)
    {
      store64(h+8*i, s[i]);
    }
    h += r;
    nblocks--;
  }
}

// wklee, fine grain version, slower
__global__ void shake128_gpu(uint8_t *out, const uint8_t *in, size_t inlen, uint32_t outlen, uint32_t out_stride) 
{
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint8_t p = 0x1F;   // For absorb
    uint32_t r = 168;   // For shake128
    int const s = threadIdx.x % 5;
    __shared__ uint64_t A[25];
    __shared__ uint64_t C[25];
    __shared__ uint64_t D[25];
    __shared__ uint64_t d_data[25];
    uint32_t i, count=0;
    __shared__ uint8_t t[200];
    uint32_t nblocks = outlen / SHAKE128_RATE;
    outlen -= nblocks * SHAKE128_RATE;  // Remain one block?

    // Initialize arrays to zeros
    for (i = 0; i < 8; ++i) {
        if (tid < 25) 
        {
            t[i*25 + tid] = 0;
        }
    }
    if (tid < 25) 
    {
        A[tid] = 0; C[tid] = 0; D[tid] = 0; d_data[tid] = 0; 
    }

    // Absorb phase
    while (inlen >= r) 
    {         
        if(tid<17) d_data[tid] ^= load64(in + bid*CRYPTO_PUBLICKEYBYTES + 8 * tid + count*r); //136 / 8
        if (tid < 25) {
              
            A[tid] = d_data[tid];
            for (int i = 0; i<NROUNDS; ++i) {
                C[tid] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
                D[tid] = C[b[20 + s]] ^ R64(C[b[5 + s]], 1, 63);
                C[tid] = R64(A[a[tid]] ^ D[b[tid]], ro[tid][0], ro[tid][1]);
                A[d[tid]] = C[c[tid][0]] ^ ((~C[c[tid][1]]) & C[c[tid][2]]);
                
                A[tid] ^= rc[(tid == 0) ? 0 : 1][i];
            }          
            d_data[tid] = A[tid];
        }        
        inlen -= r;
        count++;
    }

    if(tid==0) 
    {
        t[inlen] = p;
        t[r - 1] |= 128;
    }
    __syncthreads();
    // printf("%u %x\n", tid, d_data[tid]);
    uint32_t repeat = (inlen/blockDim.x)+1;
    if(repeat==0) repeat = 1;
    for (i = 0; i < repeat; i++)
    {
        if(tid < inlen) t[i*blockDim.x + tid] = in[i*blockDim.x + tid+ count*r];
        inlen-=blockDim.x;
        __syncthreads();
    }   
    
    if(tid < 21) d_data[tid] ^= load64(t + 8*tid);    
    // if(bid==0) if(tid < 25) printf("%u %lx %u\n", tid, d_data[tid], count);
    // if(tid < 25) printf("%u %d %u\n", tid, in[tid], inlen);
   // if(threadIdx.x==0) 
   // {
   //    printf("\n seed\n"); for (int i = 0; i < SABER_SEEDBYTES; i++) printf("%u ", in[i]);  
   // }

    // Squeeze phase
    if (tid < 25) 
    {
        A[tid] = 0; C[tid] = 0; D[tid] = 0; 
    }    
    count=0;
    while (nblocks > 0) {
        if (tid < 25) {              
            A[tid] = d_data[tid];
            for (int i = 0; i<NROUNDS; ++i) {
                C[tid] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
                D[tid] = C[b[20 + s]] ^ R64(C[b[5 + s]], 1, 63);
                C[tid] = R64(A[a[tid]] ^ D[b[tid]], ro[tid][0], ro[tid][1]);
                A[d[tid]] = C[c[tid][0]] ^ ((~C[c[tid][1]]) & C[c[tid][2]]);
                
                A[tid] ^= rc[(tid == 0) ? 0 : 1][i];
            }          
            d_data[tid] = A[tid];
            store64(out + bid*out_stride+ count*r + 8*tid, d_data[ tid]);       
        }                
        count++;
        nblocks--;
    }

    if (outlen) {
        if (tid < 25) {              
            A[tid] = d_data[tid];
            for (int i = 0; i<NROUNDS; ++i) {
                C[tid] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
                D[tid] = C[b[20 + s]] ^ R64(C[b[5 + s]], 1, 63);
                C[tid] = R64(A[a[tid]] ^ D[b[tid]], ro[tid][0], ro[tid][1]);
                A[d[tid]] = C[c[tid][0]] ^ ((~C[c[tid][1]]) & C[c[tid][2]]);
                
                A[tid] ^= rc[(tid == 0) ? 0 : 1][i];
            }          
            d_data[tid] = A[tid];         
        }             
        
        if(tid<outlen/8) store64(out + bid*out_stride+ count*r + 8*tid, d_data[tid]);
    }
}

__global__ void shake256_gpu(uint8_t *out, const uint8_t *in, size_t inlen, uint32_t outlen, uint32_t out_stride) 
{
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint8_t p = 0x1F;   // For absorb
    uint32_t r = 136;   // For shake256
    int const s = threadIdx.x % 5;
    __shared__ uint64_t A[25];
    __shared__ uint64_t C[25];
    __shared__ uint64_t D[25];
    __shared__ uint64_t d_data[25];
    uint32_t i, count=0;
    __shared__ uint8_t t[200];
    uint32_t nblocks = outlen / SHAKE256_RATE;
    outlen -= nblocks * SHAKE256_RATE;  // Remain one block?

    // Initialize arrays to zeros
    for (i = 0; i < 8; ++i) {
        if (tid < 25) 
        {
            t[i*25 + tid] = 0;
        }
    }
    if (tid < 25) 
    {
        A[tid] = 0; C[tid] = 0; D[tid] = 0; d_data[tid] = 0; 
    }

    // Absorb phase
    while (inlen >= r) 
    {         
        if(tid<17) d_data[tid] ^= load64(in + 8 * tid + count*r); //136 / 8
        if (tid < 25) {
              
            A[tid] = d_data[tid];
            for (int i = 0; i<NROUNDS; ++i) {
                C[tid] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
                D[tid] = C[b[20 + s]] ^ R64(C[b[5 + s]], 1, 63);
                C[tid] = R64(A[a[tid]] ^ D[b[tid]], ro[tid][0], ro[tid][1]);
                A[d[tid]] = C[c[tid][0]] ^ ((~C[c[tid][1]]) & C[c[tid][2]]);
                
                A[tid] ^= rc[(tid == 0) ? 0 : 1][i];
            }          
            d_data[tid] = A[tid];
        }        
        inlen -= r;
        count++;
    }

    if(tid==0) 
    {
        t[inlen] = p;
        t[r - 1] |= 128;
    }
    // printf("%u %x\n", tid, t[tid]);
    uint32_t repeat = (inlen/blockDim.x)+1;
    if(repeat==0) repeat = 1;
    for (i = 0; i < repeat; i++)
    {
        if(tid < inlen) t[i*blockDim.x + tid] = in[i*blockDim.x + tid+ count*r];
        inlen-=blockDim.x;
        __syncthreads();
    }   
    
    if(tid < 21) d_data[tid] ^= load64(t + 8*tid);    
    // if(bid==0) if(tid < 25) printf("%u %lx %u\n", tid, d_data[tid], count);
    // if(tid < 25) printf("%u %d %u\n", tid, in[tid], inlen);
    // Squeeze phase
    if (tid < 25) 
    {
        A[tid] = 0; C[tid] = 0; D[tid] = 0; 
    }    
    count=0;
    while (nblocks > 0) {
        if (tid < 25) {              
            A[tid] = d_data[tid];
            for (int i = 0; i<NROUNDS; ++i) {
                C[tid] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
                D[tid] = C[b[20 + s]] ^ R64(C[b[5 + s]], 1, 63);
                C[tid] = R64(A[a[tid]] ^ D[b[tid]], ro[tid][0], ro[tid][1]);
                A[d[tid]] = C[c[tid][0]] ^ ((~C[c[tid][1]]) & C[c[tid][2]]);
                
                A[tid] ^= rc[(tid == 0) ? 0 : 1][i];
            }          
            d_data[tid] = A[tid];
            store64(out + bid*out_stride+ count*r + 8*tid, d_data[ tid]);       
        }                
        count++;
        nblocks--;
    }

    if (outlen) {
        if (tid < 25) {              
            A[tid] = d_data[tid];
            for (int i = 0; i<NROUNDS; ++i) {
                C[tid] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
                D[tid] = C[b[20 + s]] ^ R64(C[b[5 + s]], 1, 63);
                C[tid] = R64(A[a[tid]] ^ D[b[tid]], ro[tid][0], ro[tid][1]);
                A[d[tid]] = C[c[tid][0]] ^ ((~C[c[tid][1]]) & C[c[tid][2]]);
                
                A[tid] ^= rc[(tid == 0) ? 0 : 1][i];
            }          
            d_data[tid] = A[tid];         
        }             
        
        if(tid<outlen/8) store64(out + bid*out_stride+ count*r + 8*tid, d_data[tid]);
            
        
    }
}


__device__ void shake128_absorb_ser(uint64_t *s, const unsigned char *input, unsigned int inputByteLen)
{
    keccak_absorb_ser(s, SHAKE128_RATE, input, inputByteLen, 0x1F);
}


__device__ void shake128_squeezeblocks_ser(unsigned char *output, unsigned long long nblocks, uint64_t *s)
{
    keccak_squeezeblocks_ser(output, nblocks, s, SHAKE128_RATE);
}

// wklee, coarse grain version, good for generating many samples
__global__ void shake128_ser(uint8_t *output, unsigned long long outlen, uint8_t *input,  unsigned long long inlen)
{
  uint64_t s[25] = {0};
  unsigned char t[SHAKE128_RATE];
  unsigned long long nblocks = outlen/SHAKE128_RATE;
  size_t i;
  uint32_t tid = threadIdx.x;
  uint8_t seed_A[BYTES_SEED_A+2];

  // Repeat twice, save some local memories
  seed_A[0] = tid;
  for(i=0; i<BYTES_SEED_A; i++) seed_A[i+2] = input[i];
  /* Absorb input */
    keccak_absorb_ser(s, SHAKE128_RATE, seed_A, BYTES_SEED_A+2, 0x1F);
  /* Squeeze output */
  keccak_squeezeblocks_ser(output + tid*2*PARAMS_N, nblocks, s, SHAKE128_RATE);

  output += nblocks*SHAKE128_RATE;
  outlen -= nblocks*SHAKE128_RATE;

  if (outlen) 
  {
    keccak_squeezeblocks_ser(t, 1, s, SHAKE128_RATE);
    for (i = 0; i < outlen; i++)
      output[i + tid*2*PARAMS_N] = t[i];
  }
}