
#define BLOCK 768	// Parallel blocks
#define DPFRO		// Dot-product or INT32

#define REPEAT 10	// For performance measurement

void frodoKEMKeypair(uint8_t *h_pk, uint8_t *h_sk);
void frodoKEMEncap(uint8_t *ct, uint8_t *ss, const uint8_t *pk) ;
void frodoKEMDecap(uint8_t *ss, const uint8_t *ct, const uint8_t *sk) ;




