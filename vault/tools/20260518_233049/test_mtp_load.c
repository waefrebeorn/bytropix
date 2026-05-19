#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    const char *model_path = "/models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf";
    
    gguf_ctx *ctx = gguf_open(model_path);
    if (!ctx) { printf("FAIL: open\\n"); return 1; }
    
    gguf_buffer_data(ctx);
    printf("Model: %lld tensors, blob=%p size=%zu\\n", 
           (long long)ctx->n_tensors, ctx->data_blob, ctx->data_blob_size);
    
    // Test Q2_K dequant on blk.40 ffn_gate_exps (first expert only)
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.40.ffn_gate_exps.weight");
    if (!t) { printf("FAIL: tensor not found\\n"); gguf_close(ctx); return 1; }
    
    int64_t n_elems_per_expert = (int64_t)t->dims[0] * t->dims[1];
    int64_t raw_size_per_expert = gguf_raw_size(t->ggml_type, n_elems_per_expert);
    
    printf("blk.40.ffn_gate_exps: type=%d raw_size_per_expert=%lld\\n", 
           t->ggml_type, (long long)raw_size_per_expert);
    printf("  Expected: type=10(Q2_K) raw_size=84*8=672 (8 blocks of 256 for 2048x512)\n");
    
    // Dequant expert 0
    const uint8_t *src = (const uint8_t *)ctx->data_blob + t->data_offset;
    float *deq = (float *)malloc(n_elems_per_expert * sizeof(float));
    gguf_dequantize(src, t->ggml_type, n_elems_per_expert, deq);
    
    double sum = 0, sum2 = 0;
    for (int64_t i = 0; i < n_elems_per_expert; i++) {
        sum += deq[i];
        sum2 += deq[i] * deq[i];
    }
    double mean = sum / n_elems_per_expert;
    double var = sum2 / n_elems_per_expert - mean * mean;
    printf("  Expert 0: mean=%f var=%f first5=[%f %f %f %f %f]\\n", 
           mean, var, deq[0], deq[1], deq[2], deq[3], deq[4]);
    free(deq);
    
    // Test Q3_K on blk.40 ffn_down_exps
    t = gguf_find_tensor(ctx, "blk.40.ffn_down_exps.weight");
    if (!t) { printf("FAIL: down tensor not found\\n"); gguf_close(ctx); return 1; }
    
    n_elems_per_expert = (int64_t)t->dims[0] * t->dims[1];
    raw_size_per_expert = gguf_raw_size(t->ggml_type, n_elems_per_expert);
    printf("\\nblk.40.ffn_down_exps: type=%d raw_size_per_expert=%lld\\n",
           t->ggml_type, (long long)raw_size_per_expert);
    printf("  Expected: type=11(Q3_K) raw_size=110*2=220 (2 blocks of 256 for 512x2048)\\n");
    
    src = (const uint8_t *)ctx->data_blob + t->data_offset;
    deq = (float *)malloc(n_elems_per_expert * sizeof(float));
    gguf_dequantize(src, t->ggml_type, n_elems_per_expert, deq);
    
    sum = 0; sum2 = 0;
    for (int64_t i = 0; i < n_elems_per_expert; i++) {
        sum += deq[i];
        sum2 += deq[i] * deq[i];
    }
    mean = sum / n_elems_per_expert;
    var = sum2 / n_elems_per_expert - mean * mean;
    printf("  Expert 0: mean=%f var=%f first5=[%f %f %f %f %f]\\n",
           mean, var, deq[0], deq[1], deq[2], deq[3], deq[4]);
    free(deq);
    
    // Test nextn tensors (F32 and Q8_0)
    const char *nextn_names[] = {
        "blk.40.nextn.hnorm.weight",
        "blk.40.nextn.enorm.weight",
        "blk.40.nextn.eh_proj.weight",
        "blk.40.nextn.shared_head_norm.weight",
        NULL
    };
    for (const char **np = nextn_names; *np; np++) {
        t = gguf_find_tensor(ctx, *np);
        if (!t) { printf("  %s: NOT FOUND\\n", *np); continue; }
        int64_t ne = 1;
        for (int d = 0; d < t->n_dims; d++) ne *= t->dims[d];
        printf("\\n%s: type=%d dims=[", *np, t->ggml_type);
        for (int d = 0; d < t->n_dims; d++) printf("%s%lld", d?",":"", (long long)t->dims[d]);
        printf("] n_elems=%lld\\n", (long long)ne);
        
        float *vals = (float *)malloc(ne * sizeof(float));
        if (gguf_read_tensor_f32(ctx, t, vals, ne)) {
            printf("  first5=[%f %f %f %f %f]\\n", vals[0], vals[1], vals[2], vals[3], vals[4]);
        } else {
            printf("  FAILED to read!\\n");
        }
        free(vals);
    }
    
    // Load blk.40 entire layer via gguf_read_tensor_f32
    printf("\\n=== Testing full layer dequant ===\\n");
    const char *blk40_tensors[] = {
        "blk.40.attn_norm.weight",
        "blk.40.attn_q.weight",
        "blk.40.ffn_gate_inp.weight",
        "blk.40.ffn_gate_shexp.weight",
        "blk.40.ffn_up_shexp.weight",
        "blk.40.ffn_down_shexp.weight",
        NULL
    };
    for (const char **np = blk40_tensors; *np; np++) {
        t = gguf_find_tensor(ctx, *np);
        if (!t) { printf("  %s: NOT FOUND\\n", *np); continue; }
        int64_t ne = 1;
        for (int d = 0; d < t->n_dims; d++) ne *= t->dims[d];
        
        float *vals = (float *)malloc(ne * sizeof(float));
        int ok = gguf_read_tensor_f32(ctx, t, vals, ne);
        if (ok) {
            double s = 0;
            for (int64_t i = 0; i < ne; i++) s += fabs(vals[i]);
            printf("  %-40s type=%3d ok=%d avg_mag=%f\\n", *np, t->ggml_type, ok, s/ne);
        } else {
            printf("  %-40s type=%3d FAILED\\n", *np, t->ggml_type);
        }
        free(vals);
    }
    
    gguf_close(ctx);
    return 0;
}
