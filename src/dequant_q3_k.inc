// Q3_K dequantization
// Block: 256 elements, 110 bytes/block
// Structure: d(fp16,2) + qs(64) + hmask(32) + scales(12)
// From llama.cpp ggml-common.h:
//   ggml_half  d;         // 2 - super-block scale
//   uint8_t    hmask[32];  // 32 - high bit mask
//   uint8_t    qs[64];     // 64 - low 2 bits
//   uint8_t    scales[12]; // 12 - packed sub-block scales
#define Q3_K_BLOCK_SIZE 110

void dequantize_q3_K_row(const uint8_t *data, float *output, int64_t n_elems) {
    // Reference: llama.cpp ggml-quants.c dequantize_row_q3_K()
    static const uint32_t kmask1 = 0x03030303;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    
    int64_t n_blocks = (n_elems + QK_K - 1) / QK_K;
    
    for (int64_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = data + b * Q3_K_BLOCK_SIZE;
        
        // d: fp16 at offset 0
        uint16_t d_bits;
        memcpy(&d_bits, block, 2);
        float d_all = f16_to_f32(d_bits);
        
        const uint8_t *qs = block + 2;        // 64 bytes at offset 2
        const uint8_t *hm = block + 66;       // 32 bytes at offset 66
        const uint8_t *scales_raw = block + 98; // 12 bytes at offset 98
        
        // Unpack scales: 12 bytes → 16 × int8 scales
        uint32_t aux[4];
        memcpy(aux, scales_raw, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2)           | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2)           | (((tmp >> 2) & kmask1) << 4);
        const int8_t *scales = (const int8_t *)aux;
        
        int is = 0;
        float *y = output + b * QK_K;
        uint8_t m = 1;
        
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d_all * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    int low2 = (qs[l] >> shift) & 3;
                    int sign = (hm[l] & m) ? 0 : 4;  // if hm bit set: non-negative, else sub 4
                    y[128*n/128 + j*32 + l] = dl * (float)((int8_t)(low2) - sign);
                }
                
                dl = d_all * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    int low2 = (qs[l+16] >> shift) & 3;
                    int sign = (hm[l+16] & m) ? 0 : 4;
                    y[128*n/128 + j*32 + 16 + l] = dl * (float)((int8_t)(low2) - sign);
                }
                
                shift += 2;
                m <<= 1;
            }
            qs += 32;
        }
    }
}
