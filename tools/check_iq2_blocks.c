/**
 * check_iq2_blocks.c — IQ2 dequant verification
 * Reads raw blocks from GGUF and verifies dequant output
 */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QK_K 256

// IQ2_XXS block size: 66 bytes per 256 elements
#define IQ2_XXS_BLOCK_SIZE 66
// IQ2_S block size: 132 bytes per 256 elements
#define IQ2_S_BLOCK_SIZE 132

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open GGUF\n"); return 1; }
    
    // Find IQ2_XXS (type 16) and IQ2_S (type 18) tensors
    gguf_tensor_info *t_iq2xxs = NULL, *t_iq2s = NULL;
    for (int i = 0; i < ctx->n_tensors; i++) {
        if (ctx->tensors[i].ggml_type == 16 && !t_iq2xxs) t_iq2xxs = &ctx->tensors[i];
        if (ctx->tensors[i].ggml_type == 18 && !t_iq2s) t_iq2s = &ctx->tensors[i];
        if (t_iq2xxs && t_iq2s) break;
    }
    
    FILE *f = fopen(path, "rb");
    if (!f) { gguf_close(ctx); return 1; }
    
    // Test IQ2_XXS — read first 5 blocks
    if (t_iq2xxs) {
        printf("=== IQ2_XXS (type 16) ===\n");
        printf("Tensor: %s [%ld,%ld,%ld]\n", t_iq2xxs->name,
               (long)t_iq2xxs->dims[0], (long)t_iq2xxs->dims[1], (long)t_iq2xxs->dims[2]);
        
        int64_t n_blocks = (t_iq2xxs->dims[0] * t_iq2xxs->dims[1] + QK_K - 1) / QK_K;
        printf("  Total blocks: %ld\n", (long)n_blocks);
        
        for (int bi = 0; bi < 5 && bi < n_blocks; bi++) {
            long offset = ctx->data_blob_offset + t_iq2xxs->data_offset + bi * IQ2_XXS_BLOCK_SIZE;
            fseek(f, offset, SEEK_SET);
            
            uint8_t block[IQ2_XXS_BLOCK_SIZE];
            if (fread(block, 1, IQ2_XXS_BLOCK_SIZE, f) != IQ2_XXS_BLOCK_SIZE) {
                printf("  Failed to read block %d\n", bi);
                continue;
            }
            
            float output[QK_K];
            memset(output, 0, sizeof(output));
            dequantize_iq2_xxs_row(block, output, QK_K);
            
            // Check for NaN/Inf
            int bad = 0; float vmin = 1e30f, vmax = -1e30f;
            for (int j = 0; j < QK_K; j++) {
                if (isnan(output[j]) || isinf(output[j])) bad++;
                if (output[j] < vmin) vmin = output[j];
                if (output[j] > vmax) vmax = output[j];
            }
            
            printf("  Block %d: range [%f, %f] NaN/Inf=%d first8=", bi, vmin, vmax, bad);
            for (int j = 0; j < 8; j++) printf("%+.4f ", output[j]);
            printf("\n");
        }
    }
    
    // Test IQ2_S — read first 5 blocks  
    if (t_iq2s) {
        printf("\n=== IQ2_S (type 18) ===\n");
        printf("Tensor: %s [%ld,%ld,%ld]\n", t_iq2s->name,
               (long)t_iq2s->dims[0], (long)t_iq2s->dims[1], (long)t_iq2s->dims[2]);
        
        int64_t n_blocks = (t_iq2s->dims[0] * t_iq2s->dims[1] + QK_K - 1) / QK_K;
        printf("  Total blocks: %ld\n", (long)n_blocks);
        
        for (int bi = 0; bi < 5 && bi < n_blocks; bi++) {
            long offset = ctx->data_blob_offset + t_iq2s->data_offset + bi * IQ2_S_BLOCK_SIZE;
            fseek(f, offset, SEEK_SET);
            
            uint8_t block[IQ2_S_BLOCK_SIZE];
            if (fread(block, 1, IQ2_S_BLOCK_SIZE, f) != IQ2_S_BLOCK_SIZE) {
                printf("  Failed to read block %d\n", bi);
                continue;
            }
            
            float output[QK_K];
            memset(output, 0, sizeof(output));
            dequantize_iq2_s_row(block, output, QK_K);
            
            int bad = 0; float vmin = 1e30f, vmax = -1e30f;
            for (int j = 0; j < QK_K; j++) {
                if (isnan(output[j]) || isinf(output[j])) bad++;
                if (output[j] < vmin) vmin = output[j];
                if (output[j] > vmax) vmax = output[j];
            }
            
            printf("  Block %d: range [%f, %f] NaN/Inf=%d first8=", bi, vmin, vmax, bad);
            for (int j = 0; j < 8; j++) printf("%+.4f ", output[j]);
            printf("\n");
        }
    }
    
    fclose(f);
    gguf_close(ctx);
    return 0;
}
