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
    int64_t n_one = (int64_t)D_MODEL * D_FF;
    int64_t n_all = n_one * N_EXPERTS;
    int64_t raw_one = gguf_raw_size(t->ggml_type, n_one);
    
    // Use gguf_read_tensor_f32 to read expert 1 only
    // But first check: what are the first 5 output values from the FULL read for expert 1?
    float *all = (float*)malloc(n_all * sizeof(float));
    gguf_read_tensor_f32(ctx, t, all, n_all);
    
    printf("Full read expert 1 first 5: ");
    for(int i=0;i<5;i++) printf("%.6f ", all[n_one + i]);
    printf("\n");
    
    // Now try to find expert 1's data by checking blocks
    // Each block produces 256 floats
    // Expert 1 starts at block 4096
    
    float *block_ref = (float*)malloc(256 * sizeof(float));
    memcpy(block_ref, all + n_one, 256 * sizeof(float));
    printf("Expert 1 first block (from full read): ");
    for(int i=0;i<5;i++) printf("%.6f ", block_ref[i]);
    printf("\n");
    
    // Manually dequantize raw data at the position where block 4096 SHOULD be
    uint8_t *raw_block = base + 4096 * IQ2_XXS_BLOCK_SIZE;
    printf("Raw bytes at potential expert 1 start (block 4096) pos=%lld: ",
           (long long)(4096 * IQ2_XXS_BLOCK_SIZE));
    for(int i=0;i<10;i++) printf("%02x ", raw_block[i]);
    printf("\n");
    
    // The full read's output for block 4096 must come from somewhere
    // Let's trace: gguf_read_tensor_f32 reads blocks 0..(n_blocks-1)
    // n_blocks = (n_all + 255) / 256 = (268435456 + 255) / 256 = 1048576
    // Block 4096's data is at: data + 4096 * 72 = data + 294912
    
    // But data at 294912 is zeros! So where does the correct data come from?
    
    // UNLESS the gguf_read_tensor_f32 doesn't use data_blob but reads from FILE!
    // Let me check by temporarily nulling data_blob
    void *saved_blob = ctx->data_blob;
    // ctx->data_blob = NULL; // uncomment to test file-read path
    
    float *ex1 = (float*)malloc(n_one * sizeof(float));
    gguf_read_tensor_f32(ctx, t, ex1, n_one); // This reads only 1 expert's worth!
    printf("\nRead with max_elems=n_one (should be expert 0):\n");
    printf("  first 5: ");
    for(int i=0;i<5;i++) printf("%.6f ", ex1[i]);
    printf("\n");
    
    // Now read with full n_all but starting from expert 1 offset
    // Create a fake tensor info that points to expert 1
    gguf_tensor_info fake_t = *t;
    // Can't directly change data_offset as gguf_read_tensor_f32 uses it
    
    // Actually, let me just directly read from FILE at different offsets
    uint64_t tensor_pos = ctx->data_blob_offset + t->data_offset;
    
    // Read expert 2 from file
    int64_t raw_all = gguf_raw_size(t->ggml_type, n_all);
    uint8_t *raw_heap = (uint8_t*)malloc(raw_all);
    fseek(ctx->file, tensor_pos, SEEK_SET);
    fread(raw_heap, 1, raw_all, ctx->file);
    
    // Dequantize from different offset positions
    printf("\nFile read tests:\n");
    for(int64_t off = 0; off < raw_all; off += raw_one) {
        float *buf = (float*)malloc(n_one * sizeof(float));
        gguf_dequantize(raw_heap + off, t->ggml_type, n_one, buf);
        float mn=1e30,mx=-1e30;
        for(int64_t i=0;i<n_one && i<10;i++){
            if(buf[i]<mn)mn=buf[i];
            if(buf[i]>mx)mx=buf[i];
        }
        printf("  off=%lld (expert %d): min=%.2e max=%.2e first=%.6f\n",
               (long long)off, (int)(off/raw_one), mn, mx, buf[0]);
        free(buf);
    }
    
    // Compare with full_tensor_read's expert sections
    printf("\nFull F32 read expert sections:\n");
    for(int e=0; e<8; e++) {
        float *ex = all + e * n_one;
        float mn=1e30,mx=-1e30;
        for(int64_t i=0;i<n_one && i<10;i++){
            if(ex[i]<mn)mn=ex[i];
            if(ex[i]>mx)mx=ex[i];
        }
        printf("  Expert %d: min=%.2e max=%.2e first=%.6f\n", e, mn, mx, ex[0]);
    }
    
    free(all); free(ex1); free(raw_heap); free(block_ref);
    // ctx->data_blob = saved_blob;
    gguf_close(ctx);
    return 0;
}
