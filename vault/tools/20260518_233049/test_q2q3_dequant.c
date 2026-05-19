// Minimal Q2_K and Q3_K dequantization functions for bytropix
// Based on llama.cpp's ggml-quants.c (format documented in ggml-common.h)
// Block size: QK_K = 256 for all K-quants
#include "gguf_reader.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifndef QK_K
#define QK_K 256
#endif

// Block format for Q2_K (84 bytes per 256 elements)
// Layout: scales[16] | qs[64] | d[2](half) | dmin[2](half)
// scales: 16 bytes, each byte = 4-bit scale (lo) + 4-bit min (hi)
// qs: 64 bytes = 256 × 2-bit values
// d: super-block scale (float16)
// dmin: super-block min (float16)

// Block format for Q3_K (110 bytes per 256 elements)
// Layout: hmask[32] | qs[64] | scales[12] | d[2](half)
// hmask: 256 bits = 32 bytes (high bit for each quantized value)
// qs: 64 bytes = 256 × 2-bit values (low 2 bits)
// scales: 12 bytes packed into 16 × 6-bit signed values
// d: super-block scale (float16)

static float ggml_half_to_float(uint16_t h) {
    // IEEE 754 half-precision (FP16)
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        // Subnormal
        f = (sign << 31) | (0x7f - 15 + 1) << 23 | (mant << 13);
    } else if (exp == 31) {
        // Inf/NaN
        f = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(result));
    return result;
}

void dequantize_q2_K_row(const uint8_t *data, float *output, int64_t n_elems) {
    int nb = (int)((n_elems + QK_K - 1) / QK_K);
    
    for (int i = 0; i < nb; i++) {
        const uint8_t *block = data + i * 84;  // 84 bytes per Q2_K block
        
        // Read d and dmin (float16)
        float d = ggml_half_to_float(*(const uint16_t*)(block + 80));  // d at offset 80
        float min = ggml_half_to_float(*(const uint16_t*)(block + 82));  // dmin at offset 82
        
        const uint8_t *scales = block;       // scales[16] at offset 0
        const uint8_t *q = block + 16;        // qs[64] at offset 16
        
        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF);
                float ml = min * (sc >> 4);
                for (int l = 0; l < 16; l++) {
                    *output++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
                }
                
                sc = scales[is++];
                dl = d * (sc & 0xF);
                ml = min * (sc >> 4);
                for (int l = 0; l < 16; l++) {
                    *output++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;
                }
                
                shift += 2;
            }
            q += 32;
        }
    }
}

void dequantize_q3_K_row(const uint8_t *data, float *output, int64_t n_elems) {
    int nb = (int)((n_elems + QK_K - 1) / QK_K);
    
    for (int i = 0; i < nb; i++) {
        const uint8_t *block = data + i * 110;  // 110 bytes per Q3_K block
        
        float d_all = ggml_half_to_float(*(const uint16_t*)(block + 108));  // d at offset 108
        
        const uint8_t *hm = block;              // hmask[32] at offset 0
        const uint8_t *q = block + 32;           // qs[64] at offset 32
        const uint8_t *sc = block + 96;          // scales[12] at offset 96
        
        // Unpack 12 bytes of scales into 16 signed 6-bit values
        uint8_t aux[16];
        aux[0] = sc[0] & 0x3F;  aux[1] = sc[0] >> 6;
        aux[2] = sc[1] & 0x3F;  aux[3] = sc[1] >> 6;
        aux[4] = sc[2] & 0x3F;  aux[5] = sc[2] >> 6;
        aux[6] = sc[3] & 0x3F;  aux[7] = sc[3] >> 6;
        
        aux[8]  = (sc[4] & 0x0F) | ((sc[8] & 0x03) << 4);
        aux[9]  = (sc[4] >> 4) | ((sc[8] & 0x0C) << 2);
        aux[10] = (sc[5] & 0x0F) | ((sc[8] & 0x30) >> 0);  // bits from sc[8] are at pos 4,5
        aux[11] = (sc[5] >> 4) | ((sc[8] & 0xC0) >> 2);     // bits from sc[8] are at pos 6,7
        aux[12] = (sc[6] & 0x0F) | ((sc[9] & 0x03) << 4);
        aux[13] = (sc[6] >> 4) | ((sc[9] & 0x0C) << 2);
        aux[14] = (sc[7] & 0x0F) | ((sc[9] & 0x30) >> 0);
        aux[15] = (sc[7] >> 4) | ((sc[9] & 0xC0) >> 2);
        // Actually, the llama.cpp version uses a different unpack scheme
        // Let me use the bitshift approach from the reference
        // sc[8], sc[9], sc[10], sc[11] store high bits of scales 8-15
        
        // Better approach: read all 12 bytes, reconstruct 16 × 6-bit values
        // Using the bit layout from the reference:
        // The 12 scales bytes contain: 6 bits per scale, 16 scales = 96 bits = 12 bytes
        // But the bit layout interleaves the high bits differently
        // For simplicity, use the approach from the reference implementation
        
        // Actually, let me just use the raw byte layout from the reference:
        // scales are stored as 12 bytes where bytes 0-7 are the low 4 bits of scales 0-15
        // and bytes 8-11 are the high 2 bits of scales 0-15
        int8_t scales[16];
        for (int j = 0; j < 8; j++) {
            scales[j]   = (sc[j] & 0x0F) | ((sc[8+j/2] >> ((j%2)*2)) & 0x03) << 4;
            scales[j+8] = (sc[j] >> 4)   | ((sc[8+j/2] >> ((j%2)*2+1)) & 0x01) << 4;
        }
        // Wait, this doesn't work either. Let me use the reference implementation.
        
        // The reference packs 12 bytes into 16 × 6-bit scales using a specific bit layout.
        // Following llama.cpp's ggml-quants.c dequantize_row_q3_K:
        // Let me just use the simpler approach from the reference:
        
        uint32_t aux32[4];
        memcpy(aux32, sc, 12);
        uint32_t tmp = aux32[2];
        const uint32_t kmask1 = 0x03030303;
        const uint32_t kmask2 = 0x0f0f0f0f;
        
        aux32[2] = ((aux32[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux32[3] = ((aux32[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux32[0] = (aux32[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux32[1] = (aux32[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        
        int is = 0;
        uint8_t m = 1;
        
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d_all * ((int8_t)(((uint8_t*)aux32)[is++]) - 32);
                for (int l = 0; l < 16; l++) {
                    *output++ = dl * ((int8_t)((q[l+0] >> shift) & 3) - ((hm[l+0] & m) ? 0 : 4));
                }
                
                dl = d_all * ((int8_t)(((uint8_t*)aux32)[is++]) - 32);
                for (int l = 0; l < 16; l++) {
                    *output++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                }
                
                shift += 2;
                m <<= 1;
            }
            q += 32;
            hm += 32;
        }
    }
}

// Test: verify dequant against known values
int main() {
    printf("Q2_K block size: %d bytes\n", 84);
    printf("Q3_K block size: %d bytes\n", 110);
    
    // Create a simple test: single block, known input
    uint8_t q2_block[84];
    memset(q2_block, 0, sizeof(q2_block));
    
    // Set d=1.0 and dmin=0
    // FP16 1.0 = 0x3C00
    uint16_t one_f16 = 0x3C00;
    memcpy(q2_block + 80, &one_f16, 2);
    uint16_t zero_f16 = 0;
    memcpy(q2_block + 82, &zero_f16, 2);
    
    // Set all scales to 1 (value 1 in both nibbles)
    for (int i = 0; i < 16; i++) q2_block[i] = 0x11;
    
    // Set all quants to 1 (2-bit value 1 = bits 01)
    // Each byte stores 4 values: byte[0] = val0|val1<<2|val2<<4|val3<<6
    for (int i = 0; i < 64; i++) q2_block[16 + i] = 0x55;  // 01 01 01 01
    
    float q2_out[256];
    dequantize_q2_K_row(q2_block, q2_out, 256);
    
    // Check: all values should be 1*1 - 0 = 1.0
    float max_err = 0;
    for (int i = 0; i < 256; i++) {
        float err = fabsf(q2_out[i] - 1.0f);
        if (err > max_err) max_err = err;
    }
    printf("Q2_K test: max_err=%f (expect ~0)\n", max_err);
    
    // Test Q3_K
    uint8_t q3_block[110];
    memset(q3_block, 0, sizeof(q3_block));
    
    // Set d=1.0
    memcpy(q3_block + 108, &one_f16, 2);
    
    // Set all hmask=1 (all values are "minus 0" not "minus 4")
    for (int i = 0; i < 32; i++) q3_block[i] = 0xFF;
    
    // Set all quants to 1 (2-bit value 1)
    for (int i = 0; i < 64; i++) q3_block[32 + i] = 0x55;
    
    // Set all 16 scales to 33 (stored = 33, scale = 33-32 = 1.0)
    // Layout: low 4 bits in bytes 0-7, high 2 bits in bytes 8-11
    // scales[i] low4 = byte[i/2] >> ((i%2)*4) & 0xF
    // scales[i] high2 = byte[8+i/4] >> ((i%4)*2) & 0x3
    for (int i = 0; i < 12; i++) q3_block[96 + i] = 0;
    // For scales[i] = 33 = 0x21:
    //   low4 = 0x01, high2 = 0x02
    // byte[0] = (scales[1] & 0x0F) << 4 | (scales[0] & 0x0F)
    // but ALL scales are 33, so:
    q3_block[96] = 0x11;   // scales[0]=0x01, scales[1]=0x01
    q3_block[97] = 0x11;   // scales[2]=0x01, scales[3]=0x01
    q3_block[98] = 0x11;   // scales[4]=0x01, scales[5]=0x01
    q3_block[99] = 0x11;   // scales[6]=0x01, scales[7]=0x01
    // High 2 bits for scales[0..3] in byte[8]:
    //   bits 0-1: scales[0] high2 = 0x02
    //   bits 2-3: scales[1] high2 = 0x02
    //   bits 4-5: scales[2] high2 = 0x02
    //   bits 6-7: scales[3] high2 = 0x02
    q3_block[104] = (2<<0) | (2<<2) | (2<<4) | (2<<6);  // 0xAA
    // High 2 bits for scales[4..7] in byte[9]:
    q3_block[105] = 0xAA;
    // High 2 bits for scales[8..11] in byte[10]:
    q3_block[106] = 0xAA;
    // High 2 bits for scales[12..15] in byte[11]:
    q3_block[107] = 0xAA;
    
    float q3_out[256];
    dequantize_q3_K_row(q3_block, q3_out, 256);
    
    max_err = 0;
    for (int i = 0; i < 256; i++) {
        // Q3_K: 1 * (1 - (hm_bit ? 0 : 4)) = 1 * (1 - 0) = 1.0
        float expected = 1.0f;
        float err = fabsf(q3_out[i] - expected);
        if (err > max_err) max_err = err;
    }
    printf("Q3_K test: max_err=%f (expect ~0)\n", max_err);
    
    return 0;
}
