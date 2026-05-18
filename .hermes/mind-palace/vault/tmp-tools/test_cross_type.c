#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define QK_K 256

// External functions from libggml-cpu.so
extern void ggml_vec_dot_q4_K_q8_K(int n, float *s, size_t bs,
    const void *vx, size_t bx, const void *vy, size_t by, int nrc);
extern void quantize_row_q8_K(const float *x, void *y, int64_t k);

// Let me also test vec_dot for Q5_K (which works in the model)
extern void ggml_vec_dot_q5_K_q8_K(int n, float *s, size_t bs,
    const void *vx, size_t bx, const void *vy, size_t by, int nrc);

// Forward declare the generic
extern void ggml_vec_dot_q4_K_q8_K_generic(int n, float *s, size_t bs,
    const void *vx, size_t bx, const void *vy, size_t by, int nrc);

// Minimal block_q8_K
typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K/16];
} block_q8_K;

int main() {
    printf("=== Cross-type vec_dot comparison ===\n");
    printf("sizeof(block_q8_K) = %zu\n", sizeof(block_q8_K));
    
    // Create Q5_K block (manually, with proper scale encoding)
    // block_q5_K: ggml_half d (2) + ggml_half dmin (2) + scales[12] + qh[32] + qs[128]
    // Total = 176 bytes
    // We want d=1.0=0x3C00 in fp16, dmin=0, qh=all ones, qs=all ones
    // scales = all 0xFF (same decode pattern as Q4_K)
    
    uint8_t q5_block[176];
    memset(q5_block, 0, 176);
    *(uint16_t*)(q5_block + 0) = 0x3C00;  // d = 1.0 in fp16
    *(uint16_t*)(q5_block + 2) = 0;        // dmin = 0
    memset(q5_block + 4, 0xFF, 12);         // scales = all max
    memset(q5_block + 16, 0xFF, 32);        // qh = all ones (high bits)
    memset(q5_block + 48, 0xFF, 128);       // qs = all ones (low 4 bits)
    
    // Create Q8_K block from all-1 input
    float x[QK_K];
    for (int i = 0; i < QK_K; i++) x[i] = 1.0f;
    
    block_q8_K q8;
    quantize_row_q8_K(x, &q8, QK_K);
    
    printf("Q8_K: d=%f qs[0]=%d bsums[0]=%d\n", 
           q8.d, q8.qs[0], q8.bsums[0]);
    
    // Test Q5_K vec_dot (should work according to our model)
    float result_q5 = 0.0f;
    ggml_vec_dot_q5_K_q8_K(QK_K, &result_q5, 0, q5_block, 0, &q8, 0, 1);
    printf("vec_dot_q5_K_q8_K (all max) = %f\n", (double)result_q5);
    
    // Test Q4_K vec_dot
    // Create a proper Q4_K block 
    uint8_t q4_block[144];
    memset(q4_block, 0, 144);
    *(uint16_t*)(q4_block + 0) = 0x3C00;  // d = 1.0 in fp16
    *(uint16_t*)(q4_block + 2) = 0;        // dmin = 0
    memset(q4_block + 4, 0xFF, 12);         // scales = all max
    memset(q4_block + 16, 0xFF, 128);       // qs = all ones
    
    float result_q4 = 0.0f;
    ggml_vec_dot_q4_K_q8_K(QK_K, &result_q4, 0, q4_block, 0, &q8, 0, 1);
    printf("vec_dot_q4_K_q8_K (all max) = %f\n", (double)result_q4);
    
    // Also test with the Q8_K data as bytes to check for struct issues
    // Try passing q8 as a raw uint8_t array (no struct)
    uint8_t q8_raw[292];
    memcpy(q8_raw, &q8, 292);
    
    float result_q4_raw = 0.0f;
    ggml_vec_dot_q4_K_q8_K(QK_K, &result_q4_raw, 0, q4_block, 0, q8_raw, 0, 1);
    printf("vec_dot_q4_K_q8_K (raw q8 buffer) = %f\n", (double)result_q4_raw);
    
    // Maybe the issue is that 0x3C00 is being read as the WRONG type
    // For Q4_K, d and dmin are ggml_half (uint16_t) but the vec_dot might
    // use _mm_cvtph_ps which reads the raw uint16_t directly.
    // 0x3C00 should decode to 1.0. Let me verify.
    
    // Try with REAL Q5_K block from model (a known working Q5_K layer)
    FILE *f = fopen("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", "rb");
    if (f) {
        // Read a Q5_K block (type 13 = Q5_K)
        // The model has 181 Q5_K tensors. Let me read blk.0.attn_qkv.weight
        // which has dims=(2048, 8192) and is type 13.
        // I need to find its data offset.
        // From earlier analysis: blk.0.attn_qkv.weight is Q5_K
        // The first Q5_K tensor in the data section after output.weight
        
        // output.weight (Q4_K) is 286,064,640 bytes = 286 MB
        // So blk.0.attn_qkv.weight starts at offset 286,064,640 from data section start
        // data section = 10990048
        // Q5_K blocks for [2048, 8192]: nb = 8192/QK_K = 32 per row, rows = 2048
        // Total blocks = 2048 * 32 = 65536 blocks, each 176 bytes
        // First Q5_K block offset = 286,064,640 + 0 = 286,064,640
        
        fseek(f, 10990048 + 286064640, SEEK_SET);
        uint8_t real_q5[176];
        fread(real_q5, 1, 176, f);
        fclose(f);
        
        uint16_t d5_raw = *(uint16_t*)(real_q5 + 0);
        uint16_t dmin5_raw = *(uint16_t*)(real_q5 + 2);
        
        printf("\n=== REAL Q5_K BLOCK ===\n");
        printf("d=0x%04X dmin=0x%04X\n", d5_raw, dmin5_raw);
        printf("scales[0:4]=%02x %02x %02x %02x\n", real_q5[4], real_q5[5], real_q5[6], real_q5[7]);
        printf("qh[0:4]=%02x %02x %02x %02x\n", real_q5[16], real_q5[17], real_q5[18], real_q5[19]);
        printf("qs[0:4]=%02x %02x %02x %02x\n", real_q5[48], real_q5[49], real_q5[50], real_q5[51]);
        
        float result_q5_real = 0.0f;
        ggml_vec_dot_q5_K_q8_K(QK_K, &result_q5_real, 0, real_q5, 0, &q8, 0, 1);
        printf("vec_dot_q5_K_q8_K (real, all-1 input) = %f\n", (double)result_q5_real);
    }
    
    // Finally: test with ALL-ZERO Q8_K block to verify vec_dot doesn't have
    // a bug where it's reading from the wrong field
    block_q8_K q8_zero;
    memset(&q8_zero, 0, sizeof(q8_zero));
    
    float result_q4_zero = 0.0f;
    ggml_vec_dot_q4_K_q8_K(QK_K, &result_q4_zero, 0, q4_block, 0, &q8_zero, 0, 1);
    printf("vec_dot_q4_K_q8_K (zero q8) = %f\n", (double)result_q4_zero);
    
    // Test Q5_K with zero q8
    float result_q5_zero = 0.0f;
    ggml_vec_dot_q5_K_q8_K(QK_K, &result_q5_zero, 0, q5_block, 0, &q8_zero, 0, 1);
    printf("vec_dot_q5_K_q8_K (zero q8) = %f\n", (double)result_q5_zero);
    
    return 0;
}
