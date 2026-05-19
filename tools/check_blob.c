#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) { printf("fail\n"); return 1; }
    
    // Read a small known F32 tensor from GGUF via gguf_read_tensor_f32
    // Use attn_norm.weight (layer 0) — this is F32, small
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.attn_norm.weight");
    if (!t) { printf("no attn_norm\n"); return 1; }
    printf("attn_norm.weight: type=%d offset=%llu dims=%llu\n",
           t->ggml_type, (unsigned long long)t->data_offset,
           (unsigned long long)t->dims[0]);
    
    // Read via gguf_read_tensor_f32
    float *f32_data = (float *)malloc(2048 * sizeof(float));
    gguf_read_tensor_f32(ctx, t, f32_data, 2048);
    printf("F32 read: first 5: %.6f %.6f %.6f %.6f %.6f\n",
           f32_data[0], f32_data[1], f32_data[2], f32_data[3], f32_data[4]);
    
    // Now buffer data and compare
    gguf_buffer_data(ctx);
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;
    printf("blob=%p\n", (void*)blob);
    
    // Read from blob at data_offset
    const float *blob_f32 = (const float *)(blob + t->data_offset);
    printf("Blob read: first 5: %.6f %.6f %.6f %.6f %.6f\n",
           blob_f32[0], blob_f32[1], blob_f32[2], blob_f32[3], blob_f32[4]);
    
    // Compare
    int match = 1;
    for (int i = 0; i < 2048; i++) {
        if (f32_data[i] != blob_f32[i]) { match = 0; printf("Mismatch at %d: F32=%f blob=%f\n", i, f32_data[i], blob_f32[i]); break; }
    }
    printf("Match: %s\n", match ? "YES" : "NO");
    
    // Now check ffn_gate_exps (IQ2_XXS)
    t = gguf_find_tensor(ctx, "blk.0.ffn_gate_exps.weight");
    printf("\nffn_gate_exps: type=%d offset=%llu dims=%llu %llu %llu\n",
           t->ggml_type, (unsigned long long)t->data_offset,
           (unsigned long long)t->dims[0], (unsigned long long)t->dims[1], (unsigned long long)t->dims[2]);
    
    int64_t n_elems = (int64_t)t->dims[0] * t->dims[1] * t->dims[2];  // 2048*512*256
    float *f32_gate = (float *)malloc(n_elems * sizeof(float));
    printf("Reading ffn_gate_exps as F32...\n");
    gguf_read_tensor_f32(ctx, t, f32_gate, n_elems);
    
    // Check blob data — what does the raw IQ2_XXS look like?
    const uint8_t *raw_gate = blob + t->data_offset;
    printf("Raw IQ2_XXS first 16 bytes: ");
    for (int i = 0; i < 16; i++) printf("%02x ", raw_gate[i]);
    printf("\n");
    
    // Compare F32 expert 0 with what we'd get from reading type-specific data
    // First IQ2_XXS block: 256 elements → 66 bytes (34 codebook indices + 32 6-bit codes)
    int64_t expert_elems = (int64_t)t->dims[0] * t->dims[1];  // 2048 * 512
    int64_t raw_size = gguf_raw_size(t->ggml_type, expert_elems);
    printf("Expert 0 bytes (expected): %lld\n", (long long)raw_size);
    
    // Check the F32 values for expert 0
    printf("F32 gate expert 0 first 5: %.6f %.6f %.6f %.6f %.6f\n",
           f32_gate[0], f32_gate[1], f32_gate[2], f32_gate[3], f32_gate[4]);
    
    free(f32_data);
    free(f32_gate);
    gguf_close(ctx);
    return 0;
}
