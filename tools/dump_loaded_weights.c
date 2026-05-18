/**
 * Dump first expert's dequantized gate weight for comparison.
 * Uses our gguf_reader to load and dequantize, then dumps first 64 values.
 */
#include "wubu_model.h"
#include "wubu_moe.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    
    // Load MoE for layer 0
    moe_weights_t moe;
    if (!wubu_moe_load_layer(mdl.gguf_ctx, 0, &moe)) return 1;
    
    // Expert 0, column 0, first 8 values
    printf("Expert 0 gate_w[0..7]:");
    for (int i = 0; i < 8; i++)
        printf(" %.10f", moe.ffn_gate_exps[i]);
    printf("\n");
    
    // Expert 0, column 1, first 8 values
    printf("Expert 0 gate_w[D_MODEL..D_MODEL+7]:");
    for (int i = 0; i < 8; i++)
        printf(" %.10f", moe.ffn_gate_exps[D_MODEL + i]);
    printf("\n");
    
    // Expert 64, column 0, first 8 values
    int64_t off = (int64_t)64 * D_MODEL * D_FF;
    printf("Expert 64 gate_w[0..7]:");
    for (int i = 0; i < 8; i++)
        printf(" %.10f", moe.ffn_gate_exps[off + i]);
    printf("\n");
    
    // Check values against what test_expert_dequant produced
    // Expert 64, column 0, first 8 should be:
    // -0.0059783459 -0.0186823308 0.0186823308 0.0059783459 ...
    
    // Also dump expert 64's full column 0 to file
    float *col0 = (float *)malloc(D_MODEL * sizeof(float));
    for (int i = 0; i < D_MODEL; i++)
        col0[i] = moe.ffn_gate_exps[off + i];
    
    FILE *f = fopen("/tmp/dbg_loaded_gate_e64_c0.bin", "wb");
    if (f) { fwrite(col0, sizeof(float), D_MODEL, f); fclose(f); }
    
    printf("Saved loaded gate E64 C0 to /tmp/dbg_loaded_gate_e64_c0.bin\n");
    printf("D_MODEL=%d D_FF=%d\n", D_MODEL, D_FF);
    
    // Print a range of values from loaded weights
    printf("\nLoaded gate_w expert 0, first 5 columns (first 3 values each):\n");
    for (int j = 0; j < 5; j++) {
        printf("  col %d:", j);
        for (int i = 0; i < 3; i++)
            printf(" %.6f", moe.ffn_gate_exps[0 + j * D_MODEL + i]);
        printf("\n");
    }
    
    wubu_moe_free_layer(&moe);
    wubu_model_free(&mdl);
    return 0;
}
