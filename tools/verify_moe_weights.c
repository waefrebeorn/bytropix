/**
 * Verify loaded MoE weights produce correct expert 64 output.
 * Compare against direct GGUF raw file dequant.
 */
#include "wubu_model.h"
#include "wubu_moe.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    // Load model
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    
    // Load MoE for layer 0
    moe_weights_t moe;
    if (!wubu_moe_load_layer(mdl.gguf_ctx, 0, &moe)) return 1;
    
    // Check down_exps values for expert 0, pos 0
    printf("Expert 0 down_exps[0..7]:");
    for (int i=0;i<8;i++) printf(" %.8f", moe.ffn_down_exps[i]);
    printf("\n");
    
    // Expert 0 down_exps dims: [D_FF, D_MODEL] = [512, 2048]
    // Element at (k=0, j=0) = [0]
    // Element at (k=1, j=0) = [1]
    // Element at (k=0, j=1) = [D_FF] = [512]
    
    // Check if column 1 (d_model index 1) has non-zero values
    printf("Expert 0 down_exps col1: pos=%.8f pos+1=%.8f\n", 
           moe.ffn_down_exps[D_FF], moe.ffn_down_exps[D_FF+1]);
    
    // Expert 64 down_exps
    int64_t off = (int64_t)64 * D_FF * D_MODEL;
    printf("Expert 64 down_exps[0..7]:");
    for (int i=0;i<8;i++) printf(" %.8f", moe.ffn_down_exps[off + i]);
    printf("\n");
    
    // Expert 64 down_exps, d_model idx 1, first few D_FF values
    printf("Expert 64 down_exps[d_ff=0..3, d_model=0]:");
    for (int k=0;k<4;k++) printf(" %.8f", moe.ffn_down_exps[off + k + 0*D_FF]);
    printf("\n");
    printf("Expert 64 down_exps[d_ff=0..3, d_model=1]:");
    for (int k=0;k<4;k++) printf(" %.8f", moe.ffn_down_exps[off + k + 1*D_FF]);
    printf("\n");
    
    // Critical: verify magnitude of down_exps weights
    double sum_sq = 0;
    for (int i=0;i<D_FF*D_MODEL;i++) {
        sum_sq += (double)moe.ffn_down_exps[off + i] * (double)moe.ffn_down_exps[off + i];
    }
    printf("Expert 64 down_exps L2 norm: %.6f (RMS: %.6f)\n", 
           sqrt(sum_sq), sqrt(sum_sq / (D_FF * D_MODEL)));
    
    // Compare: what should the L2 norm be for IQ3_XXS dequant?
    // IQ3_XXS has 98 bytes per 256 elements, 3.0625 bpw
    // Expected RMS ≈ d_scale * sqrt(num_nonzero_blocks / total_blocks)
    
    // Load MoE input and compute expert 64 output
    float *moe_input = (float *)malloc(D_MODEL * sizeof(float));
    FILE *f = fopen("/tmp/dbg_moe_input.bin", "rb");
    fread(moe_input, sizeof(float), D_MODEL, f); fclose(f);
    
    const float *gw = moe.ffn_gate_exps + (int64_t)64 * D_MODEL * D_FF;
    const float *uw = moe.ffn_up_exps + (int64_t)64 * D_MODEL * D_FF;
    const float *dw = moe.ffn_down_exps + (int64_t)64 * D_FF * D_MODEL;
    
    // Gate and up projections
    float gate_out[D_FF], up_out[D_FF], act[D_FF], expert_out[D_MODEL];
    
    for (int j = 0; j < D_FF; j++) {
        double s = 0;
        for (int k = 0; k < D_MODEL; k++) s += moe_input[k] * gw[k + j * D_MODEL];
        gate_out[j] = s;
    }
    for (int j = 0; j < D_FF; j++) {
        double s = 0;
        for (int k = 0; k < D_MODEL; k++) s += moe_input[k] * uw[k + j * D_MODEL];
        up_out[j] = s;
    }
    for (int j = 0; j < D_FF; j++) {
        float g = gate_out[j];
        float silu = (g < -80) ? 0 : g / (1 + expf(-g));
        act[j] = silu * up_out[j];
    }
    for (int j = 0; j < D_MODEL; j++) {
        double s = 0;
        for (int k = 0; k < D_FF; k++) s += act[k] * dw[k + j * D_FF];
        expert_out[j] = s;
    }
    
    printf("\nExpert 64 via loaded weights:\n");
    printf("  Gate_out RMS: %.6f\n", sqrtf(
        ({float s=0;for(int i=0;i<D_FF;i++)s+=gate_out[i]*gate_out[i];s;})/D_FF));
    printf("  Act RMS: %.6f\n", sqrtf(
        ({float s=0;for(int i=0;i<D_FF;i++)s+=act[i]*act[i];s;})/D_FF));
    float m=0,ss=0;
    for(int i=0;i<D_MODEL;i++){m+=expert_out[i];ss+=expert_out[i]*expert_out[i];}
    printf("  Expert out: mean=%.6f std=%.6f\n", m/D_MODEL, sqrtf(ss/D_MODEL-(m/D_MODEL)*(m/D_MODEL)));
    
    wubu_moe_free_layer(&moe);
    wubu_model_free(&mdl);
    free(moe_input);
    return 0;
}
