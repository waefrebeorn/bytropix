/**
 * test_moe_layer.c — Compare CPU vs GPU MoE output for a single layer.
 * Tests layer 0 MoE with a fixed input to find the cos-sim discrepancy source.
 * Build: gcc -O3 -I include -DGPU_SUPPORT -o test_moe_layer tools/test_moe_layer.c \
 *   src/wubu_model.o src/wubu_ssm.o src/wubu_moe.o src/wubu_tokenizer.o \
 *   src/gguf_reader.o src/dequant_iq2_xxs.o src/quantized_matmul.o \
 *   src/quantized_dot_generic.o src/wubu_ssm_chunked.o src/wubu_mobius.o \
 *   src/wubu_nested_ssm.o src/wubu_nested_ssm_backward.o src/wubu_moe_backward.o \
 *   src/wubu_moe_hyperbolic.o src/wubu_poincare_ssm_backward.o src/wubu_poincare_gqa.o \
 *   src/wubu_poincare_gqa_backward.o src/wubu_mobius_linear.o src/wubu_hyperbolic_output_proj.o \
 *   src/wubu_vision.o src/qlearner.o src/rsgd.o src/wubu_tst.o -lm -fopenmp
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_ERR_IDX 20

int main(void) {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    mdl.enable_moe = true;

    // Load F32 output weight so we can read MoE params (not used otherwise)
    int gpu_ok = 0;
    if (getenv("GPU")) {
        gpu_ok = wubu_model_gpu_init(&mdl, 4096, 256);
        printf("GPU init: %s\n", gpu_ok ? "OK" : "FAILED");
    }

    // Use a known input: embedding of token 279 (" the")
    float x[D_MODEL];
    if (mdl.token_embd) {
        memcpy(x, mdl.token_embd + 279LL * D_MODEL, D_MODEL * sizeof(float));
    } else {
        for (int i = 0; i < D_MODEL; i++) x[i] = (float)(i % 100) / 100.0f;
    }

    // Run layer 0 pre-attention RMSNorm (same as model forward does)
    float normed[D_MODEL];
    wubu_rms_norm(1, 1, D_MODEL, x, mdl.layers[0].attn_norm_weight, 1e-6f, normed);

    printf("Input RMS norm: %.6f\n", sqrtf(normed[0]*normed[0]+normed[1]*normed[1]));

    // Get MoE weights for layer 0
    moe_weights_t *moe = &mdl.layers[0].moe;
    printf("MoE loaded: %d\n", moe->loaded);
    printf("Gate type: %d, Up type: %d, Down type: %d\n",
           moe->ffn_gate_exps_q_type, moe->ffn_up_exps_q_type, moe->ffn_down_exps_q_type);
    printf("Gate q ptr: %p, Up q ptr: %p, Down q ptr: %p\n",
           (void*)moe->ffn_gate_exps_q, (void*)moe->ffn_up_exps_q, (void*)moe->ffn_down_exps_q);

    // === CPU MoE forward ===
    float cpu_ffn[D_MODEL];
    memset(cpu_ffn, 0, sizeof(cpu_ffn));
    moe->gpu_ctx = NULL;  // force CPU path
    wubu_moe_forward(normed, 1, 1, moe, cpu_ffn, NULL);
    printf("CPU MoE: [0]=%.6f [1]=%.6f [2]=%.6f [3]=%.6f max=%.6f\n",
           cpu_ffn[0], cpu_ffn[1], cpu_ffn[2], cpu_ffn[3], 
           (float)(fabsf(cpu_ffn[0]) > fabsf(cpu_ffn[1]) ? fabsf(cpu_ffn[0]) : fabsf(cpu_ffn[1])));

    // === GPU MoE forward ===
    float gpu_ffn[D_MODEL];
    memset(gpu_ffn, 0, sizeof(gpu_ffn));
    if (gpu_ok) {
        moe->gpu_ctx = (void*)&mdl;  // force GPU path
        wubu_moe_forward(normed, 1, 1, moe, gpu_ffn, NULL);
        printf("GPU MoE: [0]=%.6f [1]=%.6f [2]=%.6f [3]=%.6f max=%.6f\n",
               gpu_ffn[0], gpu_ffn[1], gpu_ffn[2], gpu_ffn[3],
               (float)(fabsf(gpu_ffn[0]) > fabsf(gpu_ffn[1]) ? fabsf(gpu_ffn[0]) : fabsf(gpu_ffn[1])));
        
        // Compare
        double dot = 0, n1 = 0, n2 = 0;
        double max_diff = 0;
        int max_diff_idx = -1;
        int n_zeros_cpu = 0, n_zeros_gpu = 0;
        for (int i = 0; i < D_MODEL; i++) {
            dot += (double)cpu_ffn[i] * (double)gpu_ffn[i];
            n1  += (double)cpu_ffn[i] * (double)cpu_ffn[i];
            n2  += (double)gpu_ffn[i] * (double)gpu_ffn[i];
            double diff = fabs((double)cpu_ffn[i] - (double)gpu_ffn[i]);
            if (diff > max_diff) { max_diff = diff; max_diff_idx = i; }
            if (fabsf(cpu_ffn[i]) < 1e-10f) n_zeros_cpu++;
            if (fabsf(gpu_ffn[i]) < 1e-10f) n_zeros_gpu++;
        }
        printf("Cos-sim: %.6f\n", dot / (sqrt(n1) * sqrt(n2)));
        printf("Max diff at [%d]: cpu=%.6f gpu=%.6f diff=%.6e\n",
               max_diff_idx, cpu_ffn[max_diff_idx], gpu_ffn[max_diff_idx], max_diff);
        printf("Zeros: cpu=%d gpu=%d\n", n_zeros_cpu, n_zeros_gpu);
        
        // Show top errors
        printf("\nTop %d absolute errors:\n", MAX_ERR_IDX);
        int err_indices[D_MODEL];
        double err_vals[D_MODEL];
        for (int i = 0; i < D_MODEL; i++) {
            err_indices[i] = i;
            err_vals[i] = fabs((double)cpu_ffn[i] - (double)gpu_ffn[i]);
        }
        // Simple bubble sort for top 20
        for (int i = 0; i < MAX_ERR_IDX && i < D_MODEL; i++) {
            int best = i;
            for (int j = i+1; j < D_MODEL; j++)
                if (err_vals[j] > err_vals[best]) best = j;
            double tmpv = err_vals[i]; err_vals[i] = err_vals[best]; err_vals[best] = tmpv;
            int tmpi = err_indices[i]; err_indices[i] = err_indices[best]; err_indices[best] = tmpi;
        }
        for (int i = 0; i < MAX_ERR_IDX && i < D_MODEL; i++) {
            int idx = err_indices[i];
            printf("  [%4d] cpu=%.6f gpu=%.6f err=%.6e rel=%.4f\n",
                   idx, cpu_ffn[idx], gpu_ffn[idx], err_vals[i],
                   err_vals[i] / (fabs(cpu_ffn[idx]) + 1e-30));
        }
        
        // Dump raw values for offline analysis
        FILE *fc = fopen("/tmp/cpu_moe_layer0.bin", "wb");
        if (fc) { fwrite(cpu_ffn, sizeof(float), D_MODEL, fc); fclose(fc); }
        FILE *fg = fopen("/tmp/gpu_moe_layer0.bin", "wb");
        if (fg) { fwrite(gpu_ffn, sizeof(float), D_MODEL, fg); fclose(fg); }
        printf("\nDumped to /tmp/cpu_moe_layer0.bin and /tmp/gpu_moe_layer0.bin\n");
    }

    moe->gpu_ctx = NULL;
    wubu_model_free(&mdl);
    return 0;
}
