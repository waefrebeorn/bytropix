#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <dlfcn.h>

#define QK_K 256

typedef void (*vec_dot_fn_t)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
typedef void (*quant_fn_t)(const float *, void *, int64_t);

// For comparison
typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K/16];
} block_q8_K;

int main() {
    printf("=== Dynamic symbol resolution check ===\n");
    
    // Load via dlopen for explicit symbol resolution
    void *handle = dlopen("/home/wubu/llama.cpp/build/bin/libggml-cpu.so", RTLD_LAZY | RTLD_LOCAL);
    if (!handle) { printf("dlopen failed: %s\n", dlerror()); return 1; }
    
    vec_dot_fn_t q4_fn = (vec_dot_fn_t)dlsym(handle, "ggml_vec_dot_q4_K_q8_K");
    vec_dot_fn_t q5_fn = (vec_dot_fn_t)dlsym(handle, "ggml_vec_dot_q5_K_q8_K");
    vec_dot_fn_t q6_fn = (vec_dot_fn_t)dlsym(handle, "ggml_vec_dot_q6_K_q8_K");
    quant_fn_t   q8_quant = (quant_fn_t)dlsym(handle, "quantize_row_q8_K");
    
    printf("q4_fn = %p\n", (void*)q4_fn);
    printf("q5_fn = %p\n", (void*)q5_fn);
    printf("q8_quant = %p\n", (void*)q8_quant);
    
    if (!q4_fn || !q5_fn || !q8_quant) { printf("Symbol not found\n"); return 1; }
    
    // Create known Q4_K data
    uint8_t q4_block[144];
    memset(q4_block, 0, 144);
    *(uint16_t*)(q4_block + 0) = 0x3C00;  // d = 1.0f in fp16
    *(uint16_t*)(q4_block + 2) = 0;        // dmin = 0
    memset(q4_block + 4, 0xFF, 12);         // scales = max
    memset(q4_block + 16, 0xFF, 128);       // qs = all max
    
    // Create known Q5_K data
    uint8_t q5_block[176];
    memset(q5_block, 0, 176);
    *(uint16_t*)(q5_block + 0) = 0x3C00;  // d = 1.0f in fp16
    *(uint16_t*)(q5_block + 2) = 0;        // dmin = 0
    memset(q5_block + 4, 0xFF, 12);         // scales = max
    memset(q5_block + 16, 0xFF, 32);        // qh = max
    memset(q5_block + 48, 0xFF, 128);       // qs = max
    
    // Create known Q6_K data
    uint8_t q6_block[210];
    memset(q6_block, 0, 210);
    *(uint16_t*)(q6_block + 0) = 0x3C00;  // d = 1.0f in fp16? No wait, Q6_K has different layout
    // Q6_K: ql[128] + qh[64] + scales[16] + d(ggml_half)[2]
    // Actually from ggml-common.h:
    // uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; ggml_half d;
    memset(q6_block, 0, 210);
    memset(q6_block + 0, 0xFF, 128);   // ql = all ones
    memset(q6_block + 128, 0xFF, 64);   // qh = all ones
    memset(q6_block + 192, 0x7F, 16);   // scales = all max (int8_t = 127)
    *(uint16_t*)(q6_block + 208) = 0x3C00;  // d = 1.0f in fp16
    
    // Create Q8_K input: all 1.0
    float x[QK_K];
    for (int i = 0; i < QK_K; i++) x[i] = 1.0f;
    
    uint8_t q8_buf[292 + 64];
    uint8_t *q8_aligned = (uint8_t*)(((uintptr_t)q8_buf + 63) & ~(uintptr_t)63);
    q8_quant(x, q8_aligned, QK_K);
    
    block_q8_K *q8 = (block_q8_K *)q8_aligned;
    printf("Q8_K: d=%f qs[0]=%d bsums[0]=%d\n", q8->d, q8->qs[0], q8->bsums[0]);
    
    // Test all three types
    float r4=0, r5=0, r6=0;
    q4_fn(QK_K, &r4, 0, q4_block, 0, q8, 0, 1);
    q5_fn(QK_K, &r5, 0, q5_block, 0, q8, 0, 1);
    
    printf("Q4_K vec_dot = %f\n", (double)r4);
    printf("Q5_K vec_dot = %f\n", (double)r5);
    
    // Now try with REAL model data
    FILE *f = fopen("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", "rb");
    if (!f) { printf("Can't open model\n"); return 1; }
    
    // Read one Q4_K block of output.weight (at data_offset 0)
    fseek(f, 10990048, SEEK_SET);
    uint8_t real_q4[144];
    fread(real_q4, 1, 144, f);
    
    // Read one Q5_K block (first block of blk.0.attn_qkv.weight)
    // output.weight (Q4_K) = 286064640 bytes
    // So Q5_K data starts at data_offset + 286064640
    uint8_t real_q5[176];
    fseek(f, 10990048 + 286064640, SEEK_SET);
    fread(real_q5, 1, 176, f);
    fclose(f);
    
    // Decode scales
    uint16_t d4_raw = *(uint16_t*)(real_q4 + 0);
    uint16_t dmin4_raw = *(uint16_t*)(real_q4 + 2);
    printf("\n=== REAL Q4_K === d=0x%04X dmin=0x%04X\n", d4_raw, dmin4_raw);
    
    uint16_t d5_raw = *(uint16_t*)(real_q5 + 0);
    uint16_t dmin5_raw = *(uint16_t*)(real_q5 + 2);
    printf("=== REAL Q5_K === d=0x%04X dmin=0x%04X\n", d5_raw, dmin5_raw);
    
    // Test with real Q5_K block and Q8_K of all-1
    float r5_real = 0;
    q5_fn(QK_K, &r5_real, 0, real_q5, 0, q8, 0, 1);
    printf("REAL Q5_K vec_dot(all-1 input) = %f\n", (double)r5_real);
    
    // Test with real Q4_K block and Q8_K of all-1
    float r4_real = 0;
    q4_fn(QK_K, &r4_real, 0, real_q4, 0, q8, 0, 1);
    printf("REAL Q4_K vec_dot(all-1 input) = %f\n", (double)r4_real);
    
    // Now test with the MODEL's actual input (hidden state from real run)
    // Load the layer 39 output (the final hidden state before output projection)
    f = fopen("/tmp/dump_layers/our_layer_39_out.bin", "rb");
    if (f) {
        float h[QK_K];
        fread(h, sizeof(float), QK_K, f);
        fclose(f);
        
        // Quantize this real input
        uint8_t real_q8_buf[292 + 64];
        uint8_t *real_q8_a = (uint8_t*)(((uintptr_t)real_q8_buf + 63) & ~(uintptr_t)63);
        q8_quant(h, real_q8_a, QK_K);
        
        block_q8_K *real_q8 = (block_q8_K *)real_q8_a;
        printf("\n=== REAL INPUT H[0:256] ===\n");
        printf("Q8_K: d=%f qs[0]=%d\n", real_q8->d, real_q8->qs[0]);
        printf("h[0]=%f h[1]=%f h[255]=%f\n", (double)h[0], (double)h[1], (double)h[255]);
        
        // vec_dot with real Q5_K and real input
        float r5_real_in = 0;
        q5_fn(QK_K, &r5_real_in, 0, real_q5, 0, real_q8_a, 0, 1);
        printf("REAL Q5_K vec_dot(REAL input) = %f\n", (double)r5_real_in);
        
        // vec_dot with real Q4_K and real input
        float r4_real_in = 0;
        q4_fn(QK_K, &r4_real_in, 0, real_q4, 0, real_q8_a, 0, 1);
        printf("REAL Q4_K vec_dot(REAL input) = %f\n", (double)r4_real_in);
    }
    
    dlclose(handle);
    return 0;
}
