#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// from libggml-cpu.so
extern void ggml_vec_dot_q4_K_q8_K(int n, float *s, size_t bs,
    const void *vx, size_t bx, const void *vy, size_t by, int nrc);
extern void quantize_row_q8_K(const float *x, void *y, int64_t k);

#define QK_K 256
#define Q4K_BLOCK_SIZE 144
#define Q8K_BLOCK_SIZE 292

int main() {
    printf("=== REAL MODEL TEST ===\n");
    
    // Read output.weight Q4_K data from file
    // Data section at 10990048, output.weight at offset 0
    FILE *f = fopen("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", "rb");
    if (!f) { printf("Can't open model\n"); return 1; }
    fseek(f, 10990048, SEEK_SET);
    
    // Read first Q4_K block (144 bytes)
    uint8_t q4_block[144];
    fread(q4_block, 1, 144, f);
    fclose(f);
    
    // Decode first block
    uint16_t d_raw = ((uint16_t *)q4_block)[0];
    uint16_t dmin_raw = ((uint16_t *)q4_block)[1];
    
    // fp16 decode using ldexp
    float d_val = 0.0f;
    {
        int sign = (d_raw >> 15) & 1;
        int exp = (d_raw >> 10) & 0x1F;
        int mant = d_raw & 0x3FF;
        if (exp == 0) {
            d_val = ldexpf((float)mant / 1024.0f, -14);
        } else if (exp == 31) {
            d_val = sign ? -INFINITY : INFINITY;
        } else {
            d_val = ldexpf(1.0f + (float)mant / 1024.0f, exp - 15);
        }
        if (sign) d_val = -d_val;
    }
    float dmin_val = 0.0f;
    {
        int sign = (dmin_raw >> 15) & 1;
        int exp = (dmin_raw >> 10) & 0x1F;
        int mant = dmin_raw & 0x3FF;
        if (exp == 0) {
            dmin_val = ldexpf((float)mant / 1024.0f, -14);
        } else if (exp == 31) {
            dmin_val = sign ? -INFINITY : INFINITY;
        } else {
            dmin_val = ldexpf(1.0f + (float)mant / 1024.0f, exp - 15);
        }
        if (sign) dmin_val = -dmin_val;
    }
    
    printf("Q4_K block: d=%f (raw 0x%04X) dmin=%f (raw 0x%04X)\n",
           d_val, d_raw, dmin_val, dmin_raw);
    printf("  scales: %d %d %d %d %d %d %d %d %d %d %d %d\n",
           q4_block[4], q4_block[5], q4_block[6], q4_block[7],
           q4_block[8], q4_block[9], q4_block[10], q4_block[11],
           q4_block[12], q4_block[13], q4_block[14], q4_block[15]);
    printf("  qs[0:8]: %d %d %d %d %d %d %d %d\n",
           q4_block[16], q4_block[17], q4_block[18], q4_block[19],
           q4_block[20], q4_block[21], q4_block[22], q4_block[23]);
    
    // Test vec_dot with test vector x = {1, 0, ..., 0} (256 elements)
    float test_x[256];
    memset(test_x, 0, sizeof(test_x));
    test_x[0] = 1.0f;
    
    uint8_t test_q8[Q8K_BLOCK_SIZE + 64];
    uint8_t *test_q8_a = (uint8_t *)(((uintptr_t)test_q8 + 63) & ~(uintptr_t)63);
    quantize_row_q8_K(test_x, test_q8_a, 256);
    
    float dot_result = 0.0f;
    ggml_vec_dot_q4_K_q8_K(256, &dot_result, 0, q4_block, 0, test_q8_a, 0, 1);
    printf("\nvec_dot(test_x[0]=1.0, Q4_K_block[0]) = %f\n", (double)dot_result);
    
    // Debug: read Q8_K block structure
    // block_q8_K: float d + int8_t qs[256] + int16_t bsums[16]
    float q8_d = *(float *)test_q8_a;
    int8_t *q8_qs = (int8_t *)(test_q8_a + 4);
    printf("  Q8_K d=%f, qs[0:5]={%d,%d,%d,%d,%d}\n",
           (double)q8_d, q8_qs[0], q8_qs[1], q8_qs[2], q8_qs[3], q8_qs[4]);
    
    // Check if bsums are correct
    int16_t *bsums = (int16_t *)(test_q8_a + 260);
    printf("  bsums[0:3]={%d,%d,%d}\n", bsums[0], bsums[1], bsums[2]);
    
    // Dequantize the Q8_K block manually
    float q8_d = *(float *)test_q8_a;
    int8_t *q8_qs = (int8_t *)(test_q8_a + 4);
    printf("  Q8_K: d=%f, q8[0:5]={%d,%d,%d,%d,%d}\n",
           (double)q8_d, q8_qs[0], q8_qs[1], q8_qs[2], q8_qs[3], q8_qs[4]);
    // Dequantized x = d * qs
    float deq0 = q8_d * q8_qs[0];
    printf("  Dequantized x[0] = d * qs[0] = %f * %d = %f (expected 1.0)\n",
           (double)q8_d, q8_qs[0], (double)deq0);
    
    // Also test vec_dot with a manually constructed Q8_K block
    // to verify the function works at all
    uint8_t manual_q8[Q8K_BLOCK_SIZE];
    memset(manual_q8, 0, Q8K_BLOCK_SIZE);
    *(float *)manual_q8 = 1.0f;  // d = 1.0
    // Fill qs with known values: first 16 elements = {1, 0, 0, ...} 
    manual_q8[4] = 1;  // qs[0] = 1
    
    float manual_dot = 0.0f;
    ggml_vec_dot_q4_K_q8_K(256, &manual_dot, 0, q4_block, 0, manual_q8, 0, 1);
    printf("vec_dot(Q4_K, MANUAL Q8_K) = %f\n", (double)manual_dot);
    
    // Also check: maybe the issue is bsums mismatch
    // Set bsums correctly for the manual Q8_K (sum of qs values in each group of 16)
    // manual_q8 has qs[0]=1 and rest=0, so bsums[0] should be 1
    int16_t *manual_bsums = (int16_t *)(manual_q8 + 260);
    manual_bsums[0] = 1;  // sum of first 16 qs values
    
    float manual_dot2 = 0.0f;
    ggml_vec_dot_q4_K_q8_K(256, &manual_dot2, 0, q4_block, 0, manual_q8, 0, 1);
    printf("vec_dot(Q4_K, MANUAL Q8_K with bsums) = %f\n", (double)manual_dot2);
    
    // Test vec_dot with a CUSTOM Q4_K block: all ones
    // If I can make a simple Q4_K block and compute its dot with Q8_K,
    // I can verify the function independently
    // Q4_K: d=1.0 in fp16 = 0x3C00, dmin=0
    // Set d=1.0, dmin=0, scales=63 (max), qs=15 (max 4-bit, for first sub-block at least)
    uint8_t custom_q4[144];
    memset(custom_q4, 0, 144);
    *(uint16_t*)(custom_q4 + 0) = 0x3C00;  // d = 1.0 in fp16
    // For Q4_K, scale encoding: each 6-bit scale value [0..63]
    // Scale encoding: uint32_t utmp[4], utmp[0] = scales[0:3] packed as 6-bit values
    // utmp[0] = s0 | s1<<6 | s2<<12 | s3<<18
    // Each byte of scales[] contributes to 2 partial 6-bit values
    // The AVX2 code does complex unpacking. Let me just set all scales to 63 (max).
    // scales[0:11] control 16*6-bit values, we need first 8 to be 63
    // 63 in 6 bits = 0x3F
    // Bytes for 4 scales of 6 bits each:
    // scale0=63, scale1=63, scale2=63, scale3=63
    // bits for byte 0: s0[5:0] (63=0x3F) and s1[1:0] (63 & 3 = 3)
    // This is getting complex. Let me use the SCALE format directly:
    // In Q4_K, each scale byte encodes parts of 2 scales
    // Actually, from the ggml decode code:
    // The first 4 bytes (utmp[0]) contain 4 scales packed as 6-bit values
    // utmp[0] = scales[0] + scales[1]*(1<<6) + scales[2]*(1<<12) + scales[3]*(1<<18)
    // But the actual bytes are in DIFFERENT order due to the shuffling
    // This is too complex to hand-craft. Let me instead check by comparing with
    // what llama-cli produces.
    
    // Actually, let me just check if the test is even calling the real vec_dot
    // by checking the function address
    printf("  vec_dot address: %p\n", ggml_vec_dot_q4_K_q8_K);
    printf("  quantize_row_q8_K address: %p\n", quantize_row_q8_K);
    
    // Test with full 2048 elements: read 8 consecutive Q4_K blocks (column 0)
    f = fopen("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", "rb");
    fseek(f, 10990048, SEEK_SET);
    uint8_t q4_8blocks[144 * 8];
    fread(q4_8blocks, 1, 144 * 8, f);
    fclose(f);
    
    // 8 blocks = D_MODEL=2048 elements per vocab column
    float full_x[2048];
    for (int i = 0; i < 2048; i++) full_x[i] = (i < 256) ? 1.0f : 0.0f;
    
    uint8_t full_q8[Q8K_BLOCK_SIZE * 8 + 64];
    uint8_t *full_q8_a = (uint8_t *)(((uintptr_t)full_q8 + 63) & ~(uintptr_t)63);
    quantize_row_q8_K(full_x, full_q8_a, 2048);
    
    float full_dot = 0.0f;
    ggml_vec_dot_q4_K_q8_K(2048, &full_dot, 0, q4_8blocks, 0, full_q8_a, 0, 1);
    printf("vec_dot(full_x[0:256]=1.0, Q4_K_col0[0:2048]) = %f\n", (double)full_dot);
    
    // Now test with a random vector
    for (int i = 0; i < 2048; i++) full_x[i] = (float)(rand() % 100) / 100.0f;
    quantize_row_q8_K(full_x, full_q8_a, 2048);
    ggml_vec_dot_q4_K_q8_K(2048, &full_dot, 0, q4_8blocks, 0, full_q8_a, 0, 1);
    printf("vec_dot(random, Q4_K_col0) = %f\n", (double)full_dot);
    
    // Compare with F32 version
    f = fopen("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", "rb");
    fseek(f, 10990048, SEEK_SET);
    // Actually we need the F32 version of the weight. Let me just print the result.
    printf("(Skipping F32 comparison, need to load dequantized weight separately)\n");
    
    return 0;
}
