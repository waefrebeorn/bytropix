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
    
    int64_t n_one = (int64_t)D_MODEL * D_FF;
    int64_t n_all = n_one * N_EXPERTS;
    
    // Method 1: Read entire tensor via gguf_read_tensor_f32
    float *all = (float*)malloc(n_all * sizeof(float));
    gguf_read_tensor_f32(ctx, t, all, n_all);
    
    printf("Method 1 (gguf_read_tensor_f32 full):\n");
    for(int e=0; e<4; e++) {
        float *ex = all + e * n_one;
        float mn=1e30,mx=-1e30;
        for(int64_t i=0;i<n_one && i<10;i++){
            if(ex[i]<mn)mn=ex[i];
            if(ex[i]>mx)mx=ex[i];
        }
        printf("  Expert %d: first 10: min=%.2e max=%.2e\n", e, mn, mx);
        printf("    values: ");
        for(int i=0;i<5;i++) printf("%.6f ", ex[i]);
        printf("\n");
    }
    
    // Method 2: Read from file directly for expert 1
    // Seek to position and read raw bytes for one expert
    uint64_t tensor_pos = ctx->data_blob_offset + t->data_offset;
    int64_t raw_one = gguf_raw_size(t->ggml_type, n_one);
    
    uint8_t *raw_heap = (uint8_t*)malloc(raw_one);
    fseek(ctx->file, tensor_pos + 0, SEEK_SET);
    fread(raw_heap, 1, raw_one, ctx->file);
    float *buf = (float*)malloc(n_one * sizeof(float));
    dequantize_iq2_xxs_row(raw_heap, buf, n_one);
    printf("\nMethod 2 (file read + dequant, expert 0): buf[0]=%.6f buf[1]=%.6f\n", buf[0], buf[1]);
    
    // Expert 1: seek to tensor_pos + raw_one
    fseek(ctx->file, tensor_pos + raw_one, SEEK_SET);
    fread(raw_heap, 1, raw_one, ctx->file);
    dequantize_iq2_xxs_row(raw_heap, buf, n_one);
    printf("Method 2 (file read + dequant, expert 1 @ off=%lld): buf[0]=%.6f\n", 
           (long long)raw_one, buf[0]);
    
    // Method 3: File read expert 1 at different offset
    // What if offset is raw_all/N_EXPERTS?
    int64_t raw_all = gguf_raw_size(t->ggml_type, n_all);
    int64_t alt_off = raw_all / N_EXPERTS;
    printf("  alt_off=%lld raw_one=%lld\n", (long long)alt_off, (long long)raw_one);
    
    fseek(ctx->file, tensor_pos + alt_off, SEEK_SET);
    fread(raw_heap, 1, raw_one, ctx->file);
    dequantize_iq2_xxs_row(raw_heap, buf, n_one);
    printf("Method 2 (file read + dequant, expert 1 @ alt_off=%lld): buf[0]=%.6f\n",
           (long long)alt_off, buf[0]);
    
    // Method 4: Read from data_blob at raw_one
    gguf_dequantize((const uint8_t*)ctx->data_blob + t->data_offset + raw_one,
                    t->ggml_type, n_one, buf);
    printf("Method 4 (data_blob + raw_one): buf[0]=%.6f\n", buf[0]);
    
    // Method 5: Read from data_blob at raw_one BUT with IQ2_XXS function directly
    dequantize_iq2_xxs_row((const uint8_t*)ctx->data_blob + t->data_offset + raw_one, buf, n_one);
    printf("Method 5 (direct iq2_xxs at data_blob + raw_one): buf[0]=%.6f\n", buf[0]);
    
    free(all); free(buf); free(raw_heap);
    gguf_close(ctx);
    return 0;
}
