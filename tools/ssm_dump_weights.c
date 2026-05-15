/**
 * ssm_dump_weights.c — Dump SSM layer 0 weight types, dequant stats, and
 * run a single SSM forward with intermediate traces.
 */
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void print_stats(const float *data, int64_t n, const char *name) {
    if (n <= 0 || !data) { printf("  %s: NULL or empty\n", name); return; }
    double mean = 0, sq = 0;
    float min = data[0], max = data[0];
    for (int64_t i = 0; i < n; i++) {
        float v = data[i];
        mean += v;
        sq += (double)v * v;
        if (v < min) min = v;
        if (v > max) max = v;
    }
    mean /= n;
    double rms = sqrt(sq / n);
    int print_n = n < 12 ? (int)n : 12;
    printf("  %s [%ld]: mean=%.4f rms=%.4f min=%.4f max=%.4f first=%d:",
           name, (long)n, mean, rms, min, max, print_n);
    for (int i = 0; i < print_n && i < n; i++) printf(" %.4f", data[i]);
    printf("\n");
}

static void dump_tensor_info(gguf_ctx *ctx, const char *tname) {
    gguf_tensor_info *t = gguf_find_tensor(ctx, tname);
    if (!t) { printf("  %s: NOT FOUND\n", tname); return; }
    printf("  %s: type=%d dims=[%ld", tname, t->ggml_type, (long)t->dims[0]);
    for (int d = 1; d < t->n_dims; d++) printf(", %ld", (long)t->dims[d]);
    printf("]\n");
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1]
        : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";

    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Can't open %s\n", path); return 1; }
    gguf_buffer_data(ctx);
    printf("=== SSM Layer 0 Weight Dump ===\n\n");

    // === Type dump ===
    printf("-- Tensor Types --\n");
    dump_tensor_info(ctx, "blk.0.attn_qkv.weight");
    dump_tensor_info(ctx, "blk.0.attn_gate.weight");
    dump_tensor_info(ctx, "blk.0.ssm_beta.weight");
    dump_tensor_info(ctx, "blk.0.ssm_alpha.weight");
    dump_tensor_info(ctx, "blk.0.ssm_dt.bias");
    dump_tensor_info(ctx, "blk.0.ssm_a");
    dump_tensor_info(ctx, "blk.0.ssm_conv1d.weight");
    dump_tensor_info(ctx, "blk.0.ssm_norm.weight");
    dump_tensor_info(ctx, "blk.0.ssm_out.weight");
    dump_tensor_info(ctx, "output.weight");
    dump_tensor_info(ctx, "token_embd.weight");
    printf("\n");

    // === Dequant + stats ===
    printf("-- Dequant Stats --\n");
    float *buf;
    int64_t n;
    gguf_tensor_info *t;

    // ssm_dt.bias [32] F32 — just read directly
    t = gguf_find_tensor(ctx, "blk.0.ssm_dt.bias");
    buf = (float *)malloc(32 * sizeof(float));
    n = gguf_read_tensor_f32(ctx, t, buf, 32);
    print_stats(buf, n, "ssm_dt.bias");
    free(buf);

    // ssm_a [32] F32
    t = gguf_find_tensor(ctx, "blk.0.ssm_a");
    buf = (float *)malloc(32 * sizeof(float));
    n = gguf_read_tensor_f32(ctx, t, buf, 32);
    print_stats(buf, n, "ssm_a (-A_log)");
    free(buf);

    // ssm_norm.weight [128] F32
    t = gguf_find_tensor(ctx, "blk.0.ssm_norm.weight");
    buf = (float *)malloc(128 * sizeof(float));
    n = gguf_read_tensor_f32(ctx, t, buf, 128);
    print_stats(buf, n, "ssm_norm.weight");
    free(buf);

    // ssm_beta.weight [2048, 32] F32
    t = gguf_find_tensor(ctx, "blk.0.ssm_beta.weight");
    n = D_MODEL * DT_RANK;
    buf = (float *)malloc(n * sizeof(float));
    n = gguf_read_tensor_f32(ctx, t, buf, n);
    print_stats(buf, n, "ssm_beta.weight");
    free(buf);

    // ssm_alpha.weight [2048, 32] F32
    t = gguf_find_tensor(ctx, "blk.0.ssm_alpha.weight");
    n = D_MODEL * DT_RANK;
    buf = (float *)malloc(n * sizeof(float));
    n = gguf_read_tensor_f32(ctx, t, buf, n);
    print_stats(buf, n, "ssm_alpha.weight");
    free(buf);

    // ssm_conv1d.weight [4, 8192] F32
    t = gguf_find_tensor(ctx, "blk.0.ssm_conv1d.weight");
    n = CONV_KERNEL * CONV_DIM;
    buf = (float *)malloc(n * sizeof(float));
    n = gguf_read_tensor_f32(ctx, t, buf, n);
    print_stats(buf, n, "ssm_conv1d.weight");
    free(buf);

    // attn_qkv.weight [2048, 8192] — quantized
    t = gguf_find_tensor(ctx, "blk.0.attn_qkv.weight");
    n = (int64_t)D_MODEL * (KEY_DIM * 2 + VALUE_DIM);
    buf = (float *)malloc(n * sizeof(float));
    n = gguf_read_tensor_f32(ctx, t, buf, n);
    print_stats(buf, n, "attn_qkv.weight");
    free(buf);

    // attn_gate.weight [2048, 4096] — quantized
    t = gguf_find_tensor(ctx, "blk.0.attn_gate.weight");
    n = (int64_t)D_MODEL * VALUE_DIM;
    buf = (float *)malloc(n * sizeof(float));
    n = gguf_read_tensor_f32(ctx, t, buf, n);
    print_stats(buf, n, "attn_gate.weight");
    free(buf);

    // ssm_out.weight [4096, 2048] Q6_K — quantized
    t = gguf_find_tensor(ctx, "blk.0.ssm_out.weight");
    n = (int64_t)VALUE_DIM * D_MODEL;
    buf = (float *)malloc(n * sizeof(float));
    n = gguf_read_tensor_f32(ctx, t, buf, n);
    print_stats(buf, n, "ssm_out.weight");
    free(buf);
    
    // output.weight [2048, 248320] — quantized (verify fix)
    t = gguf_find_tensor(ctx, "output.weight");
    if (t) {
        int64_t ne = 1; for (int d = 0; d < t->n_dims; d++) ne *= t->dims[d];
        printf("  output.weight: type=%d total_elems=%ld\n", t->ggml_type, (long)ne);
        // Quick dequant check using public API
        float deq[256];
        gguf_dequantize((const uint8_t *)ctx->data_blob + t->data_offset,
                        t->ggml_type, 256, deq);
        double m=0; for(int i=0;i<256;i++) m+=deq[i];
        printf("  output.weight first 256: mean=%.4f\n", m/256);
        printf("  first 8: ");
        for(int i=0;i<8;i++) printf("%.4f ", deq[i]);
        printf("\n");
    }

    // === Load SSM layer and run forward with tracing ===
    printf("\n-- SSM Forward Trace (T=2, no state) --\n");
    
    // Load all SSM weights
    ssm_layer_weights w;
    memset(&w, 0, sizeof(w));
    int qkv_dim = KEY_DIM * 2 + VALUE_DIM;
    
    // Manual load (same as wubu_model_init)
    t = gguf_find_tensor(ctx, "blk.0.attn_qkv.weight");
    if (t) { w.attn_qkv_weight = (float *)malloc(D_MODEL * qkv_dim * sizeof(float));
             gguf_read_tensor_f32(ctx, t, w.attn_qkv_weight, D_MODEL * qkv_dim); }
    t = gguf_find_tensor(ctx, "blk.0.attn_gate.weight");
    if (t) { w.attn_gate_weight = (float *)malloc(D_MODEL * VALUE_DIM * sizeof(float));
             gguf_read_tensor_f32(ctx, t, w.attn_gate_weight, D_MODEL * VALUE_DIM); }
    t = gguf_find_tensor(ctx, "blk.0.ssm_beta.weight");
    if (t) { w.ssm_beta_weight = (float *)malloc(D_MODEL * DT_RANK * sizeof(float));
             gguf_read_tensor_f32(ctx, t, w.ssm_beta_weight, D_MODEL * DT_RANK); }
    t = gguf_find_tensor(ctx, "blk.0.ssm_alpha.weight");
    if (t) { w.ssm_alpha_weight = (float *)malloc(D_MODEL * DT_RANK * sizeof(float));
             gguf_read_tensor_f32(ctx, t, w.ssm_alpha_weight, D_MODEL * DT_RANK); }
    t = gguf_find_tensor(ctx, "blk.0.ssm_dt.bias");
    if (t) { w.ssm_dt_bias = (float *)malloc(DT_RANK * sizeof(float));
             gguf_read_tensor_f32(ctx, t, w.ssm_dt_bias, DT_RANK); }
    t = gguf_find_tensor(ctx, "blk.0.ssm_a");
    if (t) { w.ssm_a = (float *)malloc(DT_RANK * sizeof(float));
             gguf_read_tensor_f32(ctx, t, w.ssm_a, DT_RANK); }
    t = gguf_find_tensor(ctx, "blk.0.ssm_conv1d.weight");
    if (t) { w.ssm_conv1d_weight = (float *)malloc(CONV_KERNEL * CONV_DIM * sizeof(float));
             gguf_read_tensor_f32(ctx, t, w.ssm_conv1d_weight, CONV_KERNEL * CONV_DIM); }
    t = gguf_find_tensor(ctx, "blk.0.ssm_norm.weight");
    if (t) { w.ssm_norm_weight = (float *)malloc(SSM_D_STATE * sizeof(float));
             gguf_read_tensor_f32(ctx, t, w.ssm_norm_weight, SSM_D_STATE); }
    t = gguf_find_tensor(ctx, "blk.0.ssm_out.weight");
    if (t) { w.ssm_out_weight = (float *)malloc(VALUE_DIM * D_MODEL * sizeof(float));
             gguf_read_tensor_f32(ctx, t, w.ssm_out_weight, VALUE_DIM * D_MODEL); }
    
    // Create small input (2 tokens, batch 1) with realistic embedding
    int B = 1, T = 2;
    float x[2048 * 2];
    float ssm_state[32 * 128 * 128] = {0};
    float conv_state[3 * 8192] = {0};
    float output[2048 * 2];
    
    // Simple sinusoidal input (not random — deterministic)
    for (int s = 0; s < B * T; s++)
        for (int i = 0; i < D_MODEL; i++)
            x[s * D_MODEL + i] = 0.02f * sinf((float)(s * D_MODEL + i) * 0.1f);
    
    print_stats(x, B * T * D_MODEL, "Input x");
    
    // Run SSM forward
    wubu_ssm_forward(x, B, T, &w, ssm_state, conv_state, output);
    print_stats(output, B * T * D_MODEL, "SSM output");
    
    // Also check conv_state after forward
    print_stats(conv_state, (CONV_KERNEL-1) * CONV_DIM, "conv_state after");
    
    // Check ssm_state after forward — just first head first few values
    printf("  ssm_state[0][0..7]:");
    for (int i = 0; i < 8; i++) printf(" %.4f", ssm_state[i]);
    printf("\n");
    
    // Cleanup
    free(w.attn_qkv_weight); free(w.attn_gate_weight);
    free(w.ssm_beta_weight); free(w.ssm_alpha_weight);
    free(w.ssm_dt_bias); free(w.ssm_a);
    free(w.ssm_conv1d_weight); free(w.ssm_norm_weight); free(w.ssm_out_weight);
    
    gguf_close(ctx);
    return 0;
}
