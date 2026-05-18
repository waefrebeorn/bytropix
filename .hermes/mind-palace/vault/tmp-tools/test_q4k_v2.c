#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// From ggml-common.h: block_q4_K = 144 bytes, block_q8_K = 292 bytes
#define QK_K 256
#define K_SCALE_SIZE 12

// Minimal struct matching libggml-cpu.so's layout
typedef uint16_t ggml_half;

typedef struct {
    union {
        struct {
            ggml_half d;      // 2 bytes
            ggml_half dmin;   // 2 bytes
        };
        uint32_t dm;          // 4 bytes
    };
    uint8_t scales[K_SCALE_SIZE];  // 12 bytes
    uint8_t qs[QK_K/2];           // 128 bytes
} block_q4_K;

typedef struct {
    float   d;              // 4 bytes
    int8_t  qs[QK_K];       // 256 bytes
    int16_t bsums[QK_K/16]; // 32 bytes
} block_q8_K;

// From libggml-cpu.so
extern void ggml_vec_dot_q4_K_q8_K(int n, float *s, size_t bs,
    const void *vx, size_t bx, const void *vy, size_t by, int nrc);
extern void ggml_vec_dot_q4_K_q8_K_generic(int n, float *s, size_t bs,
    const void *vx, size_t bx, const void *vy, size_t by, int nrc);
extern void quantize_row_q8_K(const float *x, void *y, int64_t k);

// fp16 -> fp32
static float fp16_to_f32(uint16_t v) {
    int sign = (v >> 15) & 1;
    int exp = (v >> 10) & 0x1F;
    int mant = v & 0x3FF;
    if (exp == 0) return ldexpf((float)mant / 1024.0f, -14);
    if (exp == 31) return sign ? -INFINITY : INFINITY;
    float f = ldexpf(1.0f + (float)mant / 1024.0f, exp - 15);
    return sign ? -f : f;
}

// BFloat = Bit-cast uint16 to fp16 (same thing on x86)
typedef uint16_t GGML_FP16;
static float ggml_fp16_to_fp32(GGML_FP16 x) {
    return fp16_to_f32(x);
}

int main() {
    printf("=== Q4_K vec_dot Deep Debug ===\n");
    printf("sizeof(block_q4_K) = %zu (expected 144)\n", sizeof(block_q4_K));
    printf("sizeof(block_q8_K) = %zu (expected 292)\n", sizeof(block_q8_K));
    
    // Create known Q4_K data: all values = 42 (approximately)
    // x = d * scale * (q - 8) ... simplified: let's just make d=1.0 in fp16
    // and q=0x0F (max) for all, then result = QK_K * d * scale_subblock * max_q
    // Actually simpler: create Q4_K block where dequant gives exactly 1.0 for all elements
    block_q4_K q4;
    memset(&q4, 0, sizeof(q4));
    q4.d = 0x3C00;  // fp16 1.0
    q4.dmin = 0;     // no min
    
    // scales: for 8 sub-blocks of 32 elements each
    // Q4_K stores scale bits across 12 bytes in a specific layout.
    // On x86 with AVX2, the scale decode is:
    // utmp[0:3] = memcpy(utmp, scales, 12)
    // utmp[3] = ((utmp[2] >> 4) & 0x0f0f0f0f) | (((utmp[1] >> 6) & 0x03030303) << 4)
    // uaux = utmp[1] & 0x3f3f3f3f
    // utmp[1] = (utmp[2] & 0x0f0f0f0f) | (((utmp[0] >> 6) & 0x03030303) << 4)
    // utmp[2] = uaux
    // utmp[0] &= 0x3f3f3f3f
    // 
    // After this, utmp has 16 6-bit scale values (we need first 8).
    // For scale=63 (max 6-bit, meaning scale=1.0 in min-scale encoding):
    // We want each 6-bit field = 0x3F
    // 
    // utmp[0] should be 0x3F3F3F3F (4 scales of 63 in lower 24 bits)
    // utmp[1] similarly
    // 
    // But after the transformation, the raw bytes are different.
    // Let me work backwards from the final utmp values.
    // 
    // Final desired: utmp[0] = 0x3F3F3F3F (scales s0..s3 = 63)
    //                utmp[1] = 0x3F3F3F3F (scales s4..s7 = 63)
    //                utmp[2] = 0x3F3F3F3F (scales s8..s11 = 63)
    // 
    // After transformation:
    // utmp[0] = utmp[0]_raw & 0x3F3F3F3F
    // So utmp[0]_raw must have 0x3F3F3F3F in its low 24 bits of each byte
    // 
    // uaux = utmp[1]_raw & 0x3F3F3F3F
    // utmp[1] = (utmp[2]_raw & 0x0F0F0F0F) | (((utmp[0]_raw >> 6) & 0x03030303) << 4)
    // utmp[2] = uaux
    // utmp[3] = ((utmp[2]_raw >> 4) & 0x0F0F0F0F) | (((utmp[1]_raw >> 6) & 0x03030303) << 4)
    
    // For simplicity: let me just set all 12 scale bytes to 0xFF
    // and see what the resulting scale values are
    memset(q4.scales, 0xFF, K_SCALE_SIZE);  // all ones
    
    // Set qs to all 0x0F (max 4-bit value = 15)
    memset(q4.qs, 0xFF, QK_K/2);  // each byte stores 2 q values
    
    // Now create a Q8_K block from input x = {1, 1, 1, ...} (all ones)
    float x[QK_K];
    for (int i = 0; i < QK_K; i++) x[i] = 1.0f;
    
    block_q8_K q8;
    quantize_row_q8_K(x, &q8, QK_K);
    
    printf("Q8_K: d=%f\n", q8.d);
    printf("  bsums[0]=%d (sum of first 16 qs)\n", q8.bsums[0]);
    printf("  qs[0]=%d\n", q8.qs[0]);
    
    // Compute vec_dot
    float result = 0.0f;
    ggml_vec_dot_q4_K_q8_K(QK_K, &result, 0, &q4, 0, &q8, 0, 1);
    printf("vec_dot(x=all1, q4=all max) = %f\n", (double)result);
    
    // Also try with block_0 of REAL data from Qwen model
    // Read from GGUF file
    FILE *f = fopen("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", "rb");
    if (!f) { printf("Can't open model!\n"); return 1; }
    fseek(f, 10990048, SEEK_SET);
    block_q4_K real_q4;
    fread(&real_q4, sizeof(block_q4_K), 1, f);
    fclose(f);
    
    printf("\n=== REAL MODEL Q4_K BLOCK ===\n");
    printf("d_raw=0x%04X d=%f\n", real_q4.d, (double)fp16_to_f32(real_q4.d));
    printf("dmin_raw=0x%04X dmin=%f\n", real_q4.dmin, (double)fp16_to_f32(real_q4.dmin));
    printf("scales[0:6]=%02x %02x %02x %02x %02x %02x\n",
           real_q4.scales[0], real_q4.scales[1], real_q4.scales[2],
           real_q4.scales[3], real_q4.scales[4], real_q4.scales[5]);
    printf("qs[0:4]=%02x %02x %02x %02x\n",
           real_q4.qs[0], real_q4.qs[1], real_q4.qs[2], real_q4.qs[3]);
    
    // vec_dot of real Q4_K block with all-ones Q8_K input
    memset(x, 0, sizeof(x));
    x[0] = 1.0f;
    quantize_row_q8_K(x, &q8, QK_K);
    
    result = 0.0f;
    ggml_vec_dot_q4_K_q8_K(QK_K, &result, 0, &real_q4, 0, &q8, 0, 1);
    printf("vec_dot(real_q4, x={1,0,...}) = %f\n", (double)result);
    
    // Also try the generic version
    float result_gen = 0.0f;
    ggml_vec_dot_q4_K_q8_K_generic(QK_K, &result_gen, 0, &real_q4, 0, &q8, 0, 1);
    printf("vec_dot_generic(real_q4, x={1,0,...}) = %f\n", (double)result_gen);
    
    // Now test with REAL Q8_K data: real input
    // Let me load the actual hidden state from the model
    // (the last layer output before final proj)
    
    return 0;
}
