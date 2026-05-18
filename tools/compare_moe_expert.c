/**
 * compare_moe_expert.c
 *
 * Direct comparison of one MoE expert computation between bytropix and llama.cpp.
 *
 * 1. Read raw IQ2_XXS data for expert 64 from the GGUF
 * 2. Dequantize using bytropix AND llama.cpp
 * 3. Compute gate/up/act/down with bytropix AND llama.cpp
 * 4. Compare at every step
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// llama.cpp headers for struct defs + dequant
#include "ggml.h"
#include "ggml-common.h"
#include "ggml-quants.h"

// bytropix dequant functions
void dequantize_iq2_xxs_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq3_xxs_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq4_xs_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_q5_K_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_q6_K_row(const uint8_t *data, float *output, int64_t n_elems);

#define D_MODEL 2048
#define D_FF 512
#define N_EXPERTS 256
#define QK_K 256
#define IQ2_BLOCK_SIZE 66

static double max_abs_diff(const float *a, const float *b, int n) {
    double max_d = 0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

static double cos_sim(const float *a, const float *b, int n) {
    double dot=0, na=0, nb=0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    return dot / (sqrt(na) * sqrt(nb) + 1e-30);
}

static double calc_silU(float g) {
    if (g < -80.0f) return 0.0f;
    return g / (1.0f + expf(-g));
}

int main(void) {
    printf("=== Expert Computation Comparison: bytropix vs llama.cpp ===\n\n");
    
    // Load MoE input
    FILE *f = fopen("/tmp/dbg_moe_input.bin", "rb");
    if (!f) { printf("ERROR: no moe_input\n"); return 1; }
    float *x = (float *)malloc(D_MODEL * sizeof(float));
    fread(x, sizeof(float), D_MODEL, f); fclose(f);
    
    // Load raw expert data
    f = fopen("/tmp/dbg_expert_gate_raw.bin", "rb");
    if (!f) { printf("ERROR: no gate raw\n"); return 1; }
    fseek(f, 0, SEEK_END);
    long gate_sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *gate_raw = (uint8_t *)malloc(gate_sz);
    fread(gate_raw, 1, gate_sz, f); fclose(f);
    
    f = fopen("/tmp/dbg_expert_up_raw.bin", "rb");
    fseek(f, 0, SEEK_END);
    long up_sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *up_raw = (uint8_t *)malloc(up_sz);
    fread(up_raw, 1, up_sz, f); fclose(f);
    
    f = fopen("/tmp/dbg_expert_down_raw.bin", "rb");
    fseek(f, 0, SEEK_END);
    long down_sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *down_raw = (uint8_t *)malloc(down_sz);
    fread(down_raw, 1, down_sz, f); fclose(f);
    
    printf("Raw data sizes: gate=%ld up=%ld down=%ld\n", gate_sz, up_sz, down_sz);
    
    // Expert 64's gate weight: [D_MODEL, D_FF] = [2048, 512]
    // Each column of 2048 elements = 8 blocks of 66 bytes = 528 bytes
    // Total: 512 * 528 = 270,336 bytes
    // Same for up and down (down: [D_FF, D_MODEL] = [512, 2048] = 2*66*2048 = 270,336)
    
    int gate_f32_elems = D_MODEL * D_FF;
    int down_f32_elems = D_FF * D_MODEL;
    
    float *gate_byt = (float *)malloc(gate_f32_elems * sizeof(float));
    float *gate_llm = (float *)malloc(gate_f32_elems * sizeof(float));
    float *up_byt   = (float *)malloc(gate_f32_elems * sizeof(float));
    float *up_llm   = (float *)malloc(gate_f32_elems * sizeof(float));
    float *down_byt = (float *)malloc(down_f32_elems * sizeof(float));
    float *down_llm = (float *)malloc(down_f32_elems * sizeof(float));
    
    // Dequantize using both implementations
    dequantize_iq2_xxs_row(gate_raw, gate_byt, gate_f32_elems);
    dequantize_iq2_xxs_row(up_raw, up_byt, gate_f32_elems);
    // down is IQ3_XXS: block_size=98, blocks_per_col=2, bytes_per_col=196
    // Total: 2048 * 196 = 401,408 bytes
    dequantize_iq3_xxs_row(down_raw, down_byt, down_f32_elems);
    dequantize_row_iq3_xxs((const block_iq3_xxs *)down_raw, down_llm, down_f32_elems);
    
    // Compare dequantized weights
    printf("\n=== Dequant Comparison ===\n");
    printf("Gate dequant:  max_diff=%.10f  cos_sim=%.10f\n", 
           max_abs_diff(gate_byt, gate_llm, gate_f32_elems),
           cos_sim(gate_byt, gate_llm, gate_f32_elems));
    printf("Up dequant:    max_diff=%.10f  cos_sim=%.10f\n",
           max_abs_diff(up_byt, up_llm, gate_f32_elems),
           cos_sim(up_byt, up_llm, gate_f32_elems));
    printf("Down dequant:  max_diff=%.10f  cos_sim=%.10f\n",
           max_abs_diff(down_byt, down_llm, down_f32_elems),
           cos_sim(down_byt, down_llm, down_f32_elems));
    
    // Print first 4 values
    printf("\nGate[0..3]: byt=(%.6f %.6f %.6f %.6f) llm=(%.6f %.6f %.6f %.6f)\n",
           gate_byt[0], gate_byt[1], gate_byt[2], gate_byt[3],
           gate_llm[0], gate_llm[1], gate_llm[2], gate_llm[3]);
    
    // =============== Expert Forward ===============
    printf("\n=== Expert Forward (gate=SiLU(x@g)*x@u, output=act@down) ===\n");
    
    // Gate: x[D_MODEL] @ gate_w[D_MODEL, D_FF] -> gate_out[D_FF]
    // Our layout: gate_w[k + j*D_MODEL]
    float gate_out_byt[D_FF], gate_out_llm[D_FF];
    float up_out_byt[D_FF], up_out_llm[D_FF];
    float act_byt[D_FF], act_llm[D_FF];
    float out_byt[D_MODEL], out_llm[D_MODEL];
    
    // Gate projection
    for (int j = 0; j < D_FF; j++) {
        double s_byt = 0, s_llm = 0;
        for (int k = 0; k < D_MODEL; k++) {
            s_byt += x[k] * gate_byt[k + j * D_MODEL];
            s_llm += x[k] * gate_llm[k + j * D_MODEL];
        }
        gate_out_byt[j] = s_byt;
        gate_out_llm[j] = s_llm;
    }
    printf("Gate:         max_diff=%.10f  cos_sim=%.10f\n",
           max_abs_diff(gate_out_byt, gate_out_llm, D_FF),
           cos_sim(gate_out_byt, gate_out_llm, D_FF));
    
    // Up projection
    for (int j = 0; j < D_FF; j++) {
        double s_byt = 0, s_llm = 0;
        for (int k = 0; k < D_MODEL; k++) {
            s_byt += x[k] * up_byt[k + j * D_MODEL];
            s_llm += x[k] * up_llm[k + j * D_MODEL];
        }
        up_out_byt[j] = s_byt;
        up_out_llm[j] = s_llm;
    }
    printf("Up:           max_diff=%.10f  cos_sim=%.10f\n",
           max_abs_diff(up_out_byt, up_out_llm, D_FF),
           cos_sim(up_out_byt, up_out_llm, D_FF));
    
    // Activation: SiLU(gate) * up
    for (int j = 0; j < D_FF; j++) {
        act_byt[j] = calc_silU(gate_out_byt[j]) * up_out_byt[j];
        act_llm[j] = calc_silU(gate_out_llm[j]) * up_out_llm[j];
    }
    printf("Act(SiLU*g):  max_diff=%.10f  cos_sim=%.10f\n",
           max_abs_diff(act_byt, act_llm, D_FF),
           cos_sim(act_byt, act_llm, D_FF));
    
    // Output: act[D_FF] @ down_w[D_FF, D_MODEL] -> out[D_MODEL]
    for (int j = 0; j < D_MODEL; j++) {
        double s_byt = 0, s_llm = 0;
        for (int k = 0; k < D_FF; k++) {
            s_byt += act_byt[k] * down_byt[k + j * D_FF];
            s_llm += act_llm[k] * down_llm[k + j * D_FF];
        }
        out_byt[j] = s_byt;
        out_llm[j] = s_llm;
    }
    printf("Output:       max_diff=%.10f  cos_sim=%.10f\n",
           max_abs_diff(out_byt, out_llm, D_MODEL),
           cos_sim(out_byt, out_llm, D_MODEL));
    
    printf("\nOutput[0..3]: byt=(%.6f %.6f %.6f %.6f) llm=(%.6f %.6f %.6f %.6f)\n",
           out_byt[0], out_byt[1], out_byt[2], out_byt[3],
           out_llm[0], out_llm[1], out_llm[2], out_llm[3]);
    
    printf("\n=== VERDICT ===\n");
    double gate_diff = max_abs_diff(gate_byt, gate_llm, gate_f32_elems);
    double out_cos = cos_sim(out_byt, out_llm, D_MODEL);
    if (gate_diff < 1e-6 && out_cos > 0.999) {
        printf("PASS: bytropix and llama.cpp produce identical expert outputs\n");
    } else {
        printf("FAIL: dequant or computation differs!\n");
    }
    
    free(x);
    free(gate_raw); free(up_raw); free(down_raw);
    free(gate_byt); free(gate_llm);
    free(up_byt); free(up_llm);
    free(down_byt); free(down_llm);
    return 0;
}
