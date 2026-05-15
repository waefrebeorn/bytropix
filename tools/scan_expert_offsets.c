#include "gguf_reader.h"
#include "wubu_moe.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    gguf_ctx *ctx = gguf_open("/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if(!ctx) return 1;
    gguf_buffer_data(ctx);
    
    char mn[256];
    snprintf(mn,sizeof(mn),"blk.0.ffn_gate_exps.weight");
    gguf_tensor_info *t = gguf_find_tensor(ctx, mn);
    if(!t) return 1;
    
    uint8_t *base = (uint8_t*)ctx->data_blob + t->data_offset;
    int64_t raw_one = gguf_raw_size(t->ggml_type, (int64_t)D_MODEL * D_FF);
    int64_t raw_all = gguf_raw_size(t->ggml_type, (int64_t)D_MODEL * D_FF * N_EXPERTS);
    
    printf("raw_one=%lld raw_all=%lld\n", (long long)raw_one, (long long)raw_all);
    
    // Find the first expert boundary by scanning
    // Look for runs of 72 zero bytes (a whole zero block)
    int64_t n_one = (int64_t)D_MODEL * D_FF;
    float *buf = (float*)malloc(n_one * sizeof(float));
    
    // Scan for correct expert offsets: 
    // Dequantize at various offsets and check if the output looks like valid weights
    printf("\nScanning for valid expert data:\n");
    for(int e=0; e<8; e++) {
        int64_t probe = e * raw_one;
        gguf_dequantize(base + probe, t->ggml_type, n_one, buf);
        float mx=-1e30;
        int64_t n_small=0, n_big=0;
        for(int64_t i=0;i<100;i++){
            float v = fabsf(buf[i]);
            if(v > mx) mx = v;
            if(v < 1.0f) n_small++;
            else n_big++;
        }
        printf("  offset %lld (expert %d): max_first_100=%f small=%lld big=%lld\n",
               (long long)probe, e, mx, (long long)n_small, (long long)n_big);
    }
    
    // Now check: what if we read from the ACTUAL TENSOR data as returned by gguf_read_tensor_f32
    // and check what raw bytes are at the expected expert boundaries
    float *all = (float*)malloc((int64_t)D_MODEL * D_FF * N_EXPERTS * sizeof(float));
    gguf_read_tensor_f32(ctx, t, all, (int64_t)D_MODEL * D_FF * N_EXPERTS);
    
    printf("\nActual expert boundaries from F32 read:\n");
    for(int e=0; e<4; e++) {
        float *ex = all + e * n_one;
        printf("  Expert %d: first=%.6f max_in_100=%.6f\n", e, ex[0], fabsf(ex[50]));
    }
    
    // Now binary search: find the correct byte offset where expert 1's data starts
    // by matching the first 5 values from the F32 read
    printf("\nBinary search for expert 1 data:\n");
    float target[5];
    for(int i=0;i<5;i++) target[i] = all[n_one + i];
    printf("  Target: %.6f %.6f %.6f %.6f %.6f\n", target[0], target[1], target[2], target[3], target[4]);
    
    // Try offsets near raw_one
    for(int64_t off = raw_one - 10000; off < raw_one + 10000; off += 72) {
        gguf_dequantize(base + off, t->ggml_type, n_one, buf);
        float diff = 0;
        for(int i=0;i<5;i++) diff += fabsf(buf[i] - target[i]);
        if(diff < 0.001f) {
            printf("  FOUND at offset %lld (diff=%f)\n", (long long)off, diff);
            break;
        }
        if(off % (72*200) == 0) {
            printf("  offset %lld: buf[0]=%.6f diff=%f\n", (long long)off, buf[0], diff);
        }
    }
    
    free(buf); free(all);
    gguf_close(ctx);
    return 0;
}
