#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gguf_reader.h"

#define MAX_MODEL_TENSORS 1024

int main() {
    gguf_ctx *ctx = gguf_load("bytropix-v13-q5_k-moe.gguf");
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    // List of key tensors to verify
    const char *key_tensors[] = {
        "token_embd.weight",
        "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
        "blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight",
        "output.weight", "output_norm.weight",
        NULL
    };
    
    printf("=== Tensor Dimension Check ===\n");
    printf("Formula: GGUF dims[0]=outermost, dims[1]=innermost\n");
    printf("For matmul: out[j] = sum_i x[i] * W[j][i]\n");
    printf("W[j][i] at linear index = j * dims[1] + i = j * INNER + i\n\n");
    
    for (int k = 0; key_tensors[k]; k++) {
        gguf_tensor_info *t = gguf_find_tensor(ctx, key_tensors[k]);
        if (!t) {
            printf("%s: NOT FOUND\n", key_tensors[k]);
            continue;
        }
        printf("  %s: dims=[%ld", key_tensors[k], t->dims[0]);
        for (int d = 1; d < t->n_dims; d++) {
            printf(", %ld", t->dims[d]);
        }
        printf("] n_dims=%d type=%d\n", t->n_dims, t->ggml_type);
    }
    
    // Also dump SSM weights
    printf("\n=== SSM Tensors ===\n");
    gguf_tensor_info *t;
    
    // Try to find with blk.0 prefix
    const char *ssm_names[] = {
        "blk.0.ssm.attn_qkv.weight", "blk.0.ssm.attn_gate.weight",
        "blk.0.ssm.beta.weight", "blk.0.ssm.alpha.weight",
        "blk.0.ssm.conv1d.weight", "blk.0.ssm.out.weight",
        "blk.0.ssm.x_proj.weight", "blk.0.ssm.dt_proj.weight",
        "blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight", "blk.0.ffn_down_shexp.weight",
        NULL
    };
    
    for (int k = 0; ssm_names[k]; k++) {
        t = gguf_find_tensor(ctx, ssm_names[k]);
        if (t) {
            printf("  %s: dims=[%ld", ssm_names[k], t->dims[0]);
            for (int d = 1; d < t->n_dims; d++) printf(", %ld", t->dims[d]);
            printf("] n_dims=%d type=%d\n", t->n_dims, t->ggml_type);
        }
    }
    
    // Try alternate naming schemes
    const char *alt_ssm[] = {
        "blk.0.ssm_attn_qkv.weight", "blk.0.ssm_attn_gate.weight",
        "blk.0.ssm_beta.weight", "blk.0.ssm_alpha.weight",
        "blk.0.ssm_conv1d.weight", "blk.0.ssm_out.weight",
        NULL
    };
    printf("\n=== Alternate SSM Names ===\n");
    for (int k = 0; alt_ssm[k]; k++) {
        t = gguf_find_tensor(ctx, alt_ssm[k]);
        if (t) {
            printf("  %s: dims=[%ld", alt_ssm[k], t->dims[0]);
            for (int d = 1; d < t->n_dims; d++) printf(", %ld", t->dims[d]);
            printf("] n_dims=%d type=%d\n", t->n_dims, t->ggml_type);
        }
    }
    
    // Dump ALL tensor names
    printf("\n=== ALL TENSORS (first 50) ===\n");
    for (int i = 0; i < ctx->n_tensors && i < 50; i++) {
        printf("  [%d] %s: dims=[%ld", i, ctx->tensors[i].name, ctx->tensors[i].dims[0]);
        for (int d = 1; d < ctx->tensors[i].n_dims; d++)
            printf(", %ld", ctx->tensors[i].dims[d]);
        printf("] n_dims=%d type=%d\n", ctx->tensors[i].n_dims, ctx->tensors[i].ggml_type);
    }
    
    // Now verify an actual value from a known tensor
    // The old and new indexing formulas give different results
    // Let's compute the actual weight being read for a specific element
    // to see which formula is correct
    
    printf("\n=== Verification of indexing formulas ===\n");
    printf("Using attn_q weight (if available):\n");
    t = gguf_find_tensor(ctx, "blk.0.attn_q.weight");
    if (t) {
        int64_t M = t->dims[0]; // outermost (row count)
        int64_t N = t->dims[1]; // innermost (col count)
        printf("  dims[0]=%ld (outermost), dims[1]=%ld (innermost)\n", M, N);
        printf("  For element W[j][i] where j=output(0..%ld-1), i=input(0..%ld-1):\n", M, N);
        printf("  Row-major: j * dims[1] + i = j*%ld + i\n", N);
        printf("  Col-major: i * dims[0] + j = i*%ld + j\n", M);
        printf("\n");
        printf("  Sample: W[1][5] (j=1, i=5):\n");
        printf("    Row-major: 1*%ld + 5 = %ld\n", N, 1*N + 5);
        printf("    Col-major: 5*%ld + 1 = %ld\n", M, 5*M + 1);
    }
    
    // Check ssm_out.weight specifically
    printf("\n=== SSM out.weight cross-check ===\n");
    t = gguf_find_tensor(ctx, "blk.0.ssm_out.weight");
    if (!t) t = gguf_find_tensor(ctx, "blk.0.ssm.out.weight");
    if (t) {
        int64_t M = t->dims[0]; // outermost
        int64_t N = t->dims[1]; // innermost
        printf("  %s: dims=[%ld, %ld]\n", t->name, M, N);
        printf("  For out[j] = sum_i inp[i] * W[j][i]:\n");
        printf("  Correct (row-major): j * %ld + i\n", N);
        printf("  OLD code (i*D_MODEL+j where D_MODEL=%d): depends on constants\n", 2048);
        printf("  NEW code (i+j*VALUE_DIM where VALUE_DIM=%d): depends\n", 4096);
        
        // Read the first few values
        int64_t n_elems = 1;
        for (int d = 0; d < t->n_dims; d++) n_elems *= t->dims[d];
        
        // Actually verify: read tensor and check
        float *buf = malloc(n_elems * sizeof(float));
        int got = gguf_read_tensor_f32(ctx, t, buf, n_elems);
        if (got > 0) {
            printf("  First 8 values: ");
            for (int k = 0; k < 8 && k < n_elems; k++) printf("%.6f ", buf[k]);
            printf("\n");
            printf("  Values at row-major(W[1][0]..W[1][4]): ");
            for (int j = 0; j < 5; j++) printf("%.6f ", buf[1*N + j]);
            printf("\n");
            printf("  Values at col-major(W[0][1]..W[4][1]): ");
            for (int i = 0; i < 5; i++) printf("%.6f ", buf[i*M + 1]);
            printf("\n");
        }
        free(buf);
    }
    
    // Same check for attn_qkv
    printf("\n=== attn_qkv.weight cross-check ===\n");
    t = gguf_find_tensor(ctx, "blk.0.ssm.attn_qkv.weight");
    if (!t) t = gguf_find_tensor(ctx, "blk.0.ssm_attn_qkv.weight");
    if (t) {
        int64_t M = t->dims[0];
        int64_t N = t->dims[1];
        printf("  %s: dims=[%ld, %ld]\n", t->name, M, N);
        printf("  Assuming matmul: out[j] = sum_i x[i] * W[j][i]\n");
        printf("  Row-major: j * %ld + i\n", N);
        printf("  Col-major: i * %ld + j\n", M);
    }
    
    gguf_free(ctx);
    return 0;
}
