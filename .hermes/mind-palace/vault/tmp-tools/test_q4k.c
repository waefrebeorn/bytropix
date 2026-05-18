#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define QK_K 256
#define Q4K_BLOCK_SIZE 144
#define Q8K_BLOCK_SIZE 292

// Same as in quantized_matmul.c
extern void ggml_vec_dot_q4_K_q8_K(int n, float *s, size_t bs,
    const void *vx, size_t bx, const void *vy, size_t by, int nrc);
extern void quantize_row_q8_K(const float *x, void *y, int64_t k);

// Minimal fp16 decode
static float fp16_to_f32(uint16_t v) {
    int sign = (v >> 15) & 1;
    int exp = (v >> 10) & 0x1F;
    int mant = v & 0x3FF;
    if (exp == 0) return 0.0f;
    if (exp == 31) return sign ? -INFINITY : INFINITY;
    float f = (1 << (exp - 15)) * (1.0f + mant / 1024.0f);
    return sign ? -f : f;
}

int main() {
    // Create a simple test: x = all 1.0, weight = all 1.0
    // After Q4_K quantization of ones: q = max value, d ≈ 1/15 (since max q is 15 for 4-bit)
    // After Q8_K quantization of ones: q = 127, d = 1/127
    // dot(1, 1) over 256 elements = 256
    
    // But we need Q4_K data. Let me construct a Q4_K block of all-ones value:
    // d1 = 1/15 = 0.0666667... -> fp16
    // All q values = 15 (max of 4-bit = 0xF)
    // Scales: sub-blocks of 32 elements, each scale = 1.0 (quantized to 6 bits)
    
    int n_rows = 2048;
    int n_cols = 248320;
    int verbose = 0;
    
    // Allocate 1 Q4_K block
    uint8_t q4_block[144];
    memset(q4_block, 0, 144);
    
    // Set d = 1.0f in fp16
    // fp16 1.0 = 0x3C00
    uint16_t d_fp16 = 0x3C00;  // 1.0
    memcpy(q4_block, &d_fp16, 2);
    
    // Set dmin = 0
    uint16_t dmin_fp16 = 0;
    memcpy(q4_block + 2, &dmin_fp16, 2);
    
    // Set scales = all 0x3F (max for 6-bit = 63, meaning scale = 1.0 for no-min format)
    // Actually for Q4_K, sub-blocks of 32: scales[0:3] for first half of block, scales[4:7] for second half
    // Each scale is 6 bits. For 8 sub-blocks: scales[12 bytes]
    // 12 bytes = 24 6-bit values. We only need 8.
    // scale = 63 (max 6-bit) means scale factor = 1.0 (well, it's the max encoding)
    // Actually, Q4_K scale encoding: scale_val = (scale_code == 0x3F) ? no-min : some formula
    // Let me just set all scales to 0 which means scale = 0 and the result will be zero.
    // That won't help.
    
    // Actually this is getting too complex. Let me just test with REAL data from the model.
    
    printf("Test: Q4_K vec_dot sanity check\n");
    printf("  Q4K_BLOCK_SIZE = %d\n", Q4K_BLOCK_SIZE);
    printf("  Q8K_BLOCK_SIZE = %d\n", Q8K_BLOCK_SIZE);
    
    // Create a test: x = {1.0, 0.0, 0.0, ..., 0.0} (256 elements)
    float x[256];
    memset(x, 0, sizeof(x));
    x[0] = 1.0f;
    
    // Quantize x to Q8_K
    uint8_t q8_buf[Q8K_BLOCK_SIZE];
    quantize_row_q8_K(x, q8_buf, 256);
    
    printf("  Q8_K quantized input: d = %f\n", ((float*)q8_buf)[0]);
    
    // Create a Q4_K block where first element = 1.0 (others will be irrelevant
    // since x is only non-zero at position 0)
    // For Q4_K, element i is at position i/2 in qs, with lower/upper 4 bits
    // We want q[0] = 1 (the first 4-bit quant, in the lower 4 bits of qs[0])
    // And scale d such that the dequantized value = 1.0
    // x_reconstructed = d * (q - 8) - dmin for min-based or d * q * scale + dmin
    // Actually: Q4_K uses d * (q - 8) for the main value, but with sub-block scales
    
    // This is too complex to construct by hand. Let me try with a real model weight.
    
    printf("\n  To test with real weights, load output.weight Q4_K data\n");
    printf("  and compute vec_dot directly.\n");
    
    // Let's just verify the function linkage
    printf("\n  ggml_vec_dot_q4_K_q8_K address: %p\n", ggml_vec_dot_q4_K_q8_K);
    printf("  quantize_row_q8_K address: %p\n", quantize_row_q8_K);
    
    return 0;
}
