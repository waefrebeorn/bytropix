#include "gguf_reader.h"
#include "wubu_moe.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    gguf_ctx *ctx = gguf_open("/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if(!ctx) return 1;
    gguf_buffer_data(ctx);
    
    char mn[256];
    snprintf(mn,sizeof(mn),"blk.0.ffn_gate_exps.weight");
    gguf_tensor_info *t = gguf_find_tensor(ctx, mn);
    if(!t) return 1;
    
    printf("data_offset = %lld\n", (long long)t->data_offset);
    printf("data_blob starts at %p\n", (void*)ctx->data_blob);
    printf("data at tensor = %p\n", (void*)(ctx->data_blob + t->data_offset));
    
    int64_t n_one = (int64_t)D_MODEL * D_FF;
    int64_t raw_one = gguf_raw_size(t->ggml_type, (int64_t)D_MODEL * D_FF * N_EXPERTS) / N_EXPERTS;
    
    printf("raw_one = %lld\n", (long long)raw_one);
    
    uint8_t *base = (uint8_t*)ctx->data_blob + t->data_offset;
    
    // Try different offsets: exact block boundaries
    // For IQ2_M type 16, block structure:
    // block_size = 32, bytes_per_block = ?
    int block_size = 32;
    int64_t n_blocks = n_one / block_size;  // 32768
    int64_t bytes_per_block = raw_one / n_blocks;
    printf("n_blocks per expert = %lld\n", (long long)n_blocks);
    printf("bytes_per_block = %lld\n", (long long)bytes_per_block);
    
    // Dequantize expert 0 from tensor base
    float *buf = (float*)malloc(n_one * sizeof(float));
    gguf_dequantize(base, t->ggml_type, n_one, buf);
    float mn0=1e30,mx0=-1e30;
    for(int64_t i=0;i<n_one;i++){
        if(buf[i]<mn0)mn0=buf[i];
        if(buf[i]>mx0)mx0=buf[i];
    }
    printf("Expert 0 from base[0]: min=%.2e max=%.2e\n", mn0, mx0);
    
    // Dequantize from 294912 offset
    gguf_dequantize(base + raw_one, t->ggml_type, n_one, buf);
    float mn1=1e30,mx1=-1e30;
    for(int64_t i=0;i<n_one;i++){
        if(buf[i]<mn1)mn1=buf[i];
        if(buf[i]>mx1)mx1=buf[i];
    }
    printf("Expert 1 from base[%lld]: min=%.2e max=%.2e\n", (long long)raw_one, mn1, mx1);
    
    // Print first few bytes of expert 0's raw data and expert 1's raw data
    printf("\nExpert 0 raw first 16 bytes: ");
    for(int i=0;i<16;i++) printf("%02x ", base[i]);
    printf("\n");
    printf("Expert 1 raw first 16 bytes: ");
    for(int i=0;i<16;i++) printf("%02x ", base[raw_one + i]);
    printf("\n");
    
    // Check: is there a BLOCK ID mismatch? Print the first block fully
    printf("\nExpert 0 block 0 raw (%lld bytes):\n", (long long)bytes_per_block);
    for(int64_t i=0;i<bytes_per_block;i++) printf("%02x ", base[i]);
    printf("\n");
    printf("Expert 1 block 0 raw (%lld bytes):\n", (long long)bytes_per_block);
    for(int64_t i=0;i<bytes_per_block;i++) printf("%02x ", base[raw_one + i]);
    printf("\n");
    
    // Now try dequantizing experts piecewise but using the full tensor approach
    // Read all at once via gguf_read_tensor_f32
    int64_t n_all = (int64_t)D_MODEL * D_FF * N_EXPERTS;
    float *all = (float*)malloc(n_all * sizeof(float));
    gguf_read_tensor_f32(ctx, t, all, n_all);
    
    // Compare expert 0 from full read with from base[0] dequant
    for(int i=0;i<5;i++){
        printf("Compare [%d]: full_read=%.6f partial_dequant=%.6f\n", i, all[i], buf[i]);
    }
    
    // Check expert 1 from full read
    float *ex1_full = all + n_one;
    printf("\nExpert 1 from full read first 5: ");
    for(int i=0;i<5;i++) printf("%.6f ", ex1_full[i]);
    printf("\n");
    
    free(buf); free(all);
    gguf_close(ctx);
    return 0;
}
