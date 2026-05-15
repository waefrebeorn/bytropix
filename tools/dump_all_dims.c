#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    gguf_ctx *ctx = gguf_open("/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) { fprintf(stderr, "FAIL: open\n"); return 1; }
    
    // Dump ALL tensor names and dims (first 80)
    printf("=== ALL TENSORS ===\n");
    for (int i = 0; i < ctx->n_tensors && i < 80; i++) {
        printf("  [%3d] %-40s dims=[", i, ctx->tensors[i].name);
        for (int d = 0; d < ctx->tensors[i].n_dims && d < 4; d++) {
            if (d > 0) printf(", ");
            printf("%ld", ctx->tensors[i].dims[d]);
        }
        printf("] n=%d\n", ctx->tensors[i].n_dims);
    }
    printf("... (%d total tensors)\n", ctx->n_tensors);
    
    // Specifically find GQA layer tensors
    printf("\n=== GQA LAYER TENSORS (layers 1,3,5...) ===\n");
    for (int i = 0; i < ctx->n_tensors; i++) {
        if (strstr(ctx->tensors[i].name, "attn_q.weight") ||
            strstr(ctx->tensors[i].name, "attn_k.weight") ||
            strstr(ctx->tensors[i].name, "attn_v.weight") ||
            strstr(ctx->tensors[i].name, "attn_output.weight") ||
            strstr(ctx->tensors[i].name, "attn_q_norm") ||
            strstr(ctx->tensors[i].name, "attn_k_norm") ||
            strstr(ctx->tensors[i].name, "ffn_gate.weight") ||
            strstr(ctx->tensors[i].name, "ffn_up.weight") ||
            strstr(ctx->tensors[i].name, "ffn_down.weight")) {
            printf("  %-40s dims=[", ctx->tensors[i].name);
            for (int d = 0; d < ctx->tensors[i].n_dims && d < 4; d++) {
                if (d > 0) printf(", ");
                printf("%ld", ctx->tensors[i].dims[d]);
            }
            printf("]\n");
        }
    }
    
    // Also check MoE expert tensors
    printf("\n=== MoE EXPERT TENSORS (first 3) ===\n");
    int found = 0;
    for (int i = 0; i < ctx->n_tensors && found < 10; i++) {
        if (strstr(ctx->tensors[i].name, "ffn_gate.") ||
            strstr(ctx->tensors[i].name, "ffn_up.") ||
            strstr(ctx->tensors[i].name, "ffn_down.")) {
            printf("  %-40s dims=[", ctx->tensors[i].name);
            for (int d = 0; d < ctx->tensors[i].n_dims && d < 4; d++) {
                if (d > 0) printf(", ");
                printf("%ld", ctx->tensors[i].dims[d]);
            }
            printf("]\n");
            found++;
        }
    }
    
    // MoE router tensor
    printf("\n=== MoE ROUTER ===\n");
    for (int i = 0; i < ctx->n_tensors; i++) {
        if (strstr(ctx->tensors[i].name, "ffn_gate_inp")) {
            printf("  %-40s dims=[", ctx->tensors[i].name);
            for (int d = 0; d < ctx->tensors[i].n_dims && d < 4; d++) {
                if (d > 0) printf(", ");
                printf("%ld", ctx->tensors[i].dims[d]);
            }
            printf("]\n");
        }
    }
    
    gguf_close(ctx);
    return 0;
}
