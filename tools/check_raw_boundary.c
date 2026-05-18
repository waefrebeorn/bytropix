#include "gguf_reader.h"
#include "wubu_moe.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    
    // Count non-zero bytes in each 294912-byte chunk
    for(int e=0; e<8; e++) {
        int64_t off = e * raw_one;
        int nz = 0;
        for(int64_t i=0; i<raw_one && off+i < raw_all; i++) {
            if(base[off+i] != 0) nz++;
        }
        printf("Expert %d (off=%lld): %d non-zero bytes out of %lld\n", 
               e, (long long)off, nz, (long long)raw_one);
    }
    
    // Binary search: find where data changes from non-zero to all-zero
    printf("\nScanning for data boundary...\n");
    int64_t last_nz = 0;
    int64_t chunk = 4096;
    for(int64_t off=0; off<raw_all; off+=chunk) {
        int nz=0;
        for(int64_t i=0; i<chunk && off+i<raw_all; i++) {
            if(base[off+i]!=0) nz++;
        }
        if(nz>0) last_nz = off+chunk;
        if(off % (raw_one) == 0)
            printf("  offset %lld (expert %d): %d non-zero\n", (long long)off, (int)(off/raw_one), nz);
    }
    printf("Last non-zero byte at ~%lld\n", (long long)last_nz);
    printf("Expected raw_all=%lld\n", (long long)raw_all);
    
    // Read full data and check: does expert 1's data appear BEFORE offset 294912?
    int64_t n_one = (int64_t)D_MODEL * D_FF;
    float *buf = (float*)malloc(n_one * sizeof(float));
    
    // Try offset 0 (expert 0)
    gguf_dequantize(base, t->ggml_type, n_one, buf);
    printf("\nExpert 0 at off=0: first=%.6f last=%.6f\n", buf[0], buf[n_one-1]);
    
    // Try binary search for correct expert 1 offset
    int64_t lo=0, hi=raw_one;
    // Find first byte offset where dequant gives different result from expert 0
    for(int64_t probe=0; probe<raw_one; probe+=72) {  // 72 = IQ2_XXS block size
        gguf_dequantize(base + probe, t->ggml_type, n_one, buf);
        if(fabsf(buf[0] - 0.005362f) > 0.001f) {
            printf("Data changes at offset %lld (buf[0]=%.6f vs 0.005362)\n", (long long)probe, buf[0]);
            break;
        }
    }
    
    free(buf);
    gguf_close(ctx);
    return 0;
}
