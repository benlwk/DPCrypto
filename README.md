# DPCrypto
DPCrypto: Acceleration of Post-quantum Cryptographic Algorithms using Dot Product2Instruction on GPUs

#Introduction

The dot-product instructions in NVIDIA GPU are exploited to perform polynomial convolution/matrix multiplication found in several lattice-based cryptosystems. In this paper, we demonsrate two successful cases: FrodoKEM (DPFrodo) and Saber (DPSaber). We beleive that this can benefit other similar lattice-based schemes that cannot be accelerated by NTT. This repository also contain source codes for implementing FrodoKEM976-SHAKE and Saber parameter sets.

#How to use

There is a Makefile accompanied with the source codes in each separate folder. You can build the executable by typing "make".

Note that you need to change the sm version in GPU to suit your device in the Makefile. The default is -arch=sm_86, which is suitable for RTX3080.



