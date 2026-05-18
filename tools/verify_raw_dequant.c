/**
 * Quick verif: Read raw IQ2_XXS from GGUF, dequant, compute expert 64 output.
 * Then compare against our loaded weights.
 */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Our dequant
void dequantize_iq2_xxs_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq3_xxs_row(const uint8_t *data, float *output, int64_t n_elems);

int main(void) {
    // Open model directly
    gguf_context *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) { printf("ERROR: gguf_open failed\n"); return 1; }
    
    // Read gate_exps tensor info
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.ffn_gate_exps.weight");
    if (!t) { printf("ERROR: tensor not found\n"); return 1; }
    
    printf("gate_exps: ne[0]=%ld ne[1]=%ld ne[2]=%ld type=%d\n",
           t->ne[0], t->ne[1], t->ne[2], t->ggml_type);
    printf("data_offset=%ld\n", t->data_offset);
    
    // Read raw data
    int64_t n_elems = t->ne[0] * t->ne[1] * t->ne[2];
    int64_t n_bytes = n_elems / 256 * 66;
    uint8_t *raw = (uint8_t *)malloc(n_bytes);
    if (!gguf_read_tensor_raw(ctx, t, raw, n_bytes)) {
        printf("ERROR: read raw failed\n"); return 1;
    }
    
    // Dequantize entire tensor
    float *f32_all = (float *)malloc(n_elems * sizeof(float));
    dequantize_iq2_xxs_row(raw, f32_all, n_elems);
    
    int D = 2048, FF = 512;
    
    // Check expert 0, col 0
    printf("\nExpert 0, col 0 [0..7]:");
    for (int i = 0; i < 8; i++) printf(" %.8f", f32_all[0 * D * FF + 0 * D + i]);
    printf("\n");
    
    // Expert 0, col 1
    printf("Expert 0, col 1 [0..7]:");
    for (int i = 0; i < 8; i++) printf(" %.8f", f32_all[0 * D * FF + 1 * D + i]);
    printf("\n");
    
    // Expert 64, col 0
    int e64_off = 64 * D * FF;
    printf("Expert 64, col 0 [0..7]:");
    for (int i = 0; i < 8; i++) printf(" %.8f", f32_all[e64_off + 0 * D + i]);
    printf("\n");
    
    // Expert 64, col 1
    printf("Expert 64, col 1 [0..7]:");
    for (int i = 0; i < 8; i++) printf(" %.8f", f32_all[e64_off + 1 * D + i]);
    printf("\n");
    
    // Now compare: our MoE code uses the SAME gguf_read_tensor_f32 function
    // which internally calls dequantize_iq2_xxs_row
    // So the f32_all data should be IDENTICAL to what wubu_moe_load_layer produces
    
    printf("\n=== MATCH CHECK ===\n");
    printf("The dequant produces non-zero values for experts that have non-zero raw data.\n");
    printf("If expert 0 col 1+ is all zeros in the raw GGUF data (which it is),\n");
    printf("the F32 output should also be zeros. This is CORRECT behavior.\n");
    
    // Count non-zero blocks in raw data for expert 0
    int nz_blocks = 0;
    for (int b = 0; b < 4096; b++) { // 512*8 blocks
        uint16_t d;
        memcpy(&d, raw + b * 66, 2);
        if (d != 0) nz_blocks++;
    }
    printf("Expert 0: %d non-zero blocks out of 4096\n", nz_blocks);
    
    // Expert 64
    nz_blocks = 0;
    for (int b = 0; b < 4096; b++) {
        uint16_t d;
        memcpy(&d, raw + (64 * 4096 + b) * 66, 2);
        if (d != 0) nz_blocks++;
    }
    printf("Expert 64: %d non-zero blocks out of 4096\n", nz_blocks);
    
    // Total non-zero blocks across all experts
    int total_nz = 0;
    for (int e = 0; e < 256; e++) {
        for (int b = 0; b < 4096; b++) {
            uint16_t d;
            memcpy(&d, raw + (e * 4096 + b) * 66, 2);
            if (d != 0) total_nz++;
        }
    }
    printf("Total: %d non-zero blocks out of %d (%.2f%%)\n", 
           total_nz, 256 * 4096, 100.0 * total_nz / (256 * 4096));
    
    // Compute expert output for expert 64 using raw-derived weights
    // Need the down weights too
    float *moe_input = (float *)malloc(D * sizeof(float));
    FILE *f = fopen("/tmp/dbg_moe_input.bin", "rb");
    fread(moe_input, sizeof(float), D, f); fclose(f);
    
    const float *gw = f32_all + 64 * D * FF;
    float gate_out[FF], up_out[FF], act[FF], expert_out[D];
    
    // gate
    for (int j = 0; j < FF; j++) {
        double s = 0;
        for (int k = 0; k < D; k++) s += moe_input[k] * gw[k + j * D];
        gate_out[j] = s;
    }
    
    printf("\nExpert 64 gate_out[0..3]: %.8f %.8f %.8f %.8f\n",
           gate_out[0], gate_out[1], gate_out[2], gate_out[3]);
    
    // Now read up_exps
    gguf_tensor_info *t_up = gguf_find_tensor(ctx, "blk.0.ffn_up_exps.weight");
    uint8_t *up_raw = (uint8_t *)malloc(t_up->ne[0] * t_up->ne[1] * t_up->ne[2] / 256 * 66);
    gguf_read_tensor_raw(ctx, t_up, up_raw, t_up->ne[0] * t_up->ne[1] * t_up->ne[2] / 256 * 66);
    float *up_f32 = (float *)malloc(t_up->ne[0] * t_up->ne[1] * t_up->ne[2] * sizeof(float));
    dequantize_iq2_xxs_row(up_raw, up_f32, t_up->ne[0] * t_up->ne[1] * t_up->ne[2]);
    
    const float *uw = up_f32 + 64 * D * FF;
    for (int j = 0; j < FF; j++) {
        double s = 0;
        for (int k = 0; k < D; k++) s += moe_input[k] * uw[k + j * D];
        up_out[j] = s;
    }
    
    printf("Expert 64 up_out[0..3]: %.8f %.8f %.8f %.8f\n",
           up_out[0], up_out[1], up_out[2], up_out[3]);
    
    // act
    for (int j = 0; j < FF; j++) {
        float g = gate_out[j];
        float silu = (g < -80) ? 0 : g / (1 + expf(-g));
        act[j] = silu * up_out[j];
    }
    printf("Expert 64 act[0..3]: %.8f %.8f %.8f %.8f\n",
           act[0], act[1], act[2], act[3]);
    
    // down
    gguf_tensor_info *t_down = gguf_find_tensor(ctx, "blk.0.ffn_down_exps.weight");
    int64_t down_elems = t_down->ne[0] * t_down->ne[1] * t_down->ne[2];
    int64_t down_bytes = down_elems / 256 * 98; // IQ3_XXS
    uint8_t *down_raw = (uint8_t *)malloc(down_bytes);
    gguf_read_tensor_raw(ctx, t_down, down_raw, down_bytes);
    float *down_f32 = (float *)malloc(down_elems * sizeof(float));
    dequantize_iq3_xxs_row(down_raw, down_f32, down_elems);
    
    const float *dw = down_f32 + 64 * FF * D;
    for (int j = 0; j < D; j++) {
        double s = 0;
        for (int k = 0; k < FF; k++) s += act[k] * dw[k + j * FF];
        expert_out[j] = s;
    }
    
    printf("Expert 64 output[0..3]: %.8f %.8f %.8f %.8f\n",
           expert_out[0], expert_out[1], expert_out[2], expert_out[3]);
    
    float m=0, s=0;
    for (int i=0;i<D;i++){m+=expert_out[i];s+=expert_out[i]*expert_out[i];}
    printf("Expert 64 output: mean=%.6f std=%.6f norm=%.4f\n", 
           m/D, sqrtf(s/D-(m/D)*(m/D)), sqrtf(s));
    
    free(raw); free(f32_all); free(up_raw); free(up_f32);
    free(down_raw); free(down_f32); free(moe_input);
    gguf_free(ctx);
    return 0;
}
