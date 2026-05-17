/**
 * dump_moe_for_python.c — Dump MoE weights + test input for Python comparison.
 * Dumps dequantized expert 0 weights and router weights for layer 0.
 * Build: gcc -O2 -I include -o dump_moe_for_python tools/dump_moe_for_python.c \
 *        src/gguf_reader.o src/wubu_ssm.o src/wubu_mobius.o \
 *        src/wubu_moe.o src/wubu_model.o src/wubu_tokenizer.o \
 *        src/qlearner.o src/dequant_iq2_xxs.o -lm -fopenmp
 */
#include "wubu_model.h"
#include "wubu_moe.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }

    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) return 1;

    moe_weights_t moe;
    if (!wubu_moe_load_layer(ctx, 0, &moe)) return 1;
    fprintf(stderr, "Layer 0 MoE weights loaded\n");

    // Create simple test input: all 1s
    float x[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) x[i] = 1.0f;
    
    // Try to load the REAL MoE input (post-attention RMSNorm output)
    // from infer_text layer 0 dump
    FILE *real_f = fopen("/tmp/moe_input_layer0.bin", "rb");
    float real_x[D_MODEL];
    float *moe_in = x;  // default: all-1s
    if (real_f) {
        fread(real_x, sizeof(float), D_MODEL, real_f);
        fclose(real_f);
        moe_in = real_x;
        double s = 0; for (int i = 0; i < D_MODEL; i++) s += moe_in[i] * moe_in[i];
        fprintf(stderr, "Using REAL MoE input from layer 0 (rms=%.4f)\n", sqrt(s/D_MODEL));
    }

    // 1. Dump input
    FILE *f = fopen("/tmp/moe_test_input.bin", "wb");
    fwrite(moe_in, sizeof(float), D_MODEL, f);
    fclose(f);

    // 2. Dump router weights
    f = fopen("/tmp/moe_router_weights.bin", "wb");
    fwrite(moe.ffn_gate_inp, sizeof(float), D_MODEL * N_EXPERTS, f);
    fclose(f);

    // 3. Dump expert 0 gate weights
    int64_t ne = (int64_t)D_MODEL * D_FF;
    float *exp0_gate = (float *)malloc(ne * sizeof(float));
    memcpy(exp0_gate, moe.ffn_gate_exps + 0 * ne, ne * sizeof(float));
    f = fopen("/tmp/moe_exp0_gate.bin", "wb");
    fwrite(exp0_gate, sizeof(float), ne, f);
    fclose(f);

    // 4. Dump expert 0 up weights
    float *exp0_up = (float *)malloc(ne * sizeof(float));
    for (int64_t i = 0; i < ne; i++)
        exp0_up[i] = moe.ffn_gate_exps[i + 0 * ne]; // same as gate since identical dims
    memcpy(exp0_up, moe.ffn_up_exps + 0 * ne, ne * sizeof(float));
    f = fopen("/tmp/moe_exp0_up.bin", "wb");
    fwrite(exp0_up, sizeof(float), ne, f);
    fclose(f);

    // 5. Dump expert 0 down weights
    int64_t nd = (int64_t)D_FF * D_MODEL;
    float *exp0_down = (float *)malloc(nd * sizeof(float));
    for (int64_t i = 0; i < nd; i++)
        exp0_down[i] = moe.ffn_down_exps[i + 0 * nd];
    f = fopen("/tmp/moe_exp0_down.bin", "wb");
    fwrite(exp0_down, sizeof(float), nd, f);
    fclose(f);

    // 6. Dump shared expert weights
    f = fopen("/tmp/moe_sh_gate.bin", "wb");
    fwrite(moe.ffn_gate_shexp, sizeof(float), D_MODEL * SHARED_D_FF, f);
    fclose(f);
    f = fopen("/tmp/moe_sh_up.bin", "wb");
    fwrite(moe.ffn_up_shexp, sizeof(float), D_MODEL * SHARED_D_FF, f);
    fclose(f);
    f = fopen("/tmp/moe_sh_down.bin", "wb");
    fwrite(moe.ffn_down_shexp, sizeof(float), SHARED_D_FF * D_MODEL, f);
    fclose(f);
    if (moe.ffn_gate_inp_shexp) {
        f = fopen("/tmp/moe_sh_gate_proj.bin", "wb");
        fwrite(moe.ffn_gate_inp_shexp, sizeof(float), D_MODEL, f);
        fclose(f);
    }

    // 7. Run our full MoE forward and dump output
    float our_out[D_MODEL];
    wubu_moe_forward(moe_in, 1, 1, &moe, our_out);
    f = fopen("/tmp/moe_our_output.bin", "wb");
    fwrite(our_out, sizeof(float), D_MODEL, f);
    fclose(f);

    double s = 0; for (int i = 0; i < D_MODEL; i++) s += (double)our_out[i] * our_out[i];
    fprintf(stderr, "Our MoE layer 0 output: rms=%.6f\n", sqrt(s/D_MODEL));

    // 8. Dump top-8 expert indices from the MoE forward
    // First run the router to find top-8 experts
    float scores[N_EXPERTS];
    int topk_idx[N_ACTIVE_EXPTS];
    float topk_w[N_ACTIVE_EXPTS];
    
    wubu_moe_router(moe_in, 1, 1, moe.ffn_gate_inp, scores);
    
    // Softmax + topk
    float mx = scores[0];
    for (int e = 1; e < N_EXPERTS; e++) if (scores[e] > mx) mx = scores[e];
    float sum_exp = 0;
    for (int e = 0; e < N_EXPERTS; e++) sum_exp += expf(scores[e] - mx);
    float inv = 1.0f / sum_exp;
    float sm[N_EXPERTS];
    for (int e = 0; e < N_EXPERTS; e++) sm[e] = expf(scores[e] - mx) * inv;
    
    for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
        int best = -1; float best_v = -1e30f;
        for (int e = 0; e < N_EXPERTS; e++) {
            int used = 0;
            for (int p = 0; p < k; p++) if (topk_idx[p] == e) { used = 1; break; }
            if (!used && sm[e] > best_v) { best_v = sm[e]; best = e; }
        }
        topk_idx[k] = best; topk_w[k] = sm[best];
    }
    float sw = 0; for (int k = 0; k < N_ACTIVE_EXPTS; k++) sw += topk_w[k];
    for (int k = 0; k < N_ACTIVE_EXPTS; k++) topk_w[k] /= sw;
    
    // Write top-8 indices and weights
    f = fopen("/tmp/moe_top8_info.bin", "wb");
    fwrite(topk_idx, sizeof(int), N_ACTIVE_EXPTS, f);
    fwrite(topk_w, sizeof(float), N_ACTIVE_EXPTS, f);
    fclose(f);
    
    // Dump all 8 experts' weights
    for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
        int eid = topk_idx[k];
        char fn[256];
        
        snprintf(fn, sizeof(fn), "/tmp/moe_exp%d_gate.bin", eid);
        f = fopen(fn, "wb");
        fwrite(moe.ffn_gate_exps + eid * ne, sizeof(float), ne, f);
        fclose(f);
        
        snprintf(fn, sizeof(fn), "/tmp/moe_exp%d_up.bin", eid);
        f = fopen(fn, "wb");
        fwrite(moe.ffn_up_exps + eid * ne, sizeof(float), ne, f);
        fclose(f);
        
        snprintf(fn, sizeof(fn), "/tmp/moe_exp%d_down.bin", eid);
        f = fopen(fn, "wb");
        fwrite(moe.ffn_down_exps + eid * nd, sizeof(float), nd, f);
        fclose(f);
    }
    
    fprintf(stderr, "Top-8 experts dumped:");
    for (int k = 0; k < N_ACTIVE_EXPTS; k++) fprintf(stderr, " %d(%.4f)", topk_idx[k], topk_w[k]);
    fprintf(stderr, "\n");
    
    free(exp0_gate);
    free(exp0_up);
    free(exp0_down);
    wubu_moe_free_layer(&moe);
    gguf_close(ctx);
    return 0;
}
