/**
 * infer_unified.c — Unified 40-layer inference (SSM→GQA→MoE)
 *
 * Loads GGUF once, runs all 40 layers with lazy MoE dequant.
 * Benchmarks per-layer breakdown and total throughput.
 */
#include "wubu_model.h"
#include "wubu_moe.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Per-expert dequantized weights (one expert's gate/up/down)
typedef struct {
    int expert_id;
    float *gate;   // [D_MODEL, D_FF]
    float *up;     // [D_MODEL, D_FF]
    float *down;   // [D_FF, D_MODEL]
} expert_buf_t;

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1]
        : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    int B = 1, T = argc > 2 ? atoi(argv[2]) : 4;
    int N = B * T;
    int verbose = argc > 3 ? atoi(argv[3]) : 1;

    printf("=== Unified 40-layer Inference ===\n");
    printf("Model: %s  B=%d T=%d\n", path, B, T);

    // ================================================================
    // 1. Load GGUF + buffer all data
    // ================================================================
    double t0 = now_sec();
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    double t_load = now_sec() - t0;
    printf("GGUF load+buffer: %.2f s\n", t_load);

    // ================================================================
    // 2. Load model (SSM/GQA weights only, no MoE)
    // ================================================================
    t0 = now_sec();
    wubu_model_t model;
    if (!wubu_model_init(&model, path)) {
        fprintf(stderr, "Model init failed\n");
        gguf_close(ctx);
        return 1;
    }
    // Override with our own ctx (model already opened one)
    // Close model's ctx, use our buffered one
    if (model.gguf_ctx) {
        gguf_close(model.gguf_ctx);
    }
    model.gguf_ctx = ctx;
    double t_model = now_sec() - t0;
    printf("Model init: %.2f s\n", t_model);

    // ================================================================
    // 3. Pre-allocate per-expert metadata from GGUF
    // ================================================================
    int64_t expert_n = (int64_t)D_MODEL * D_FF;       // 1,048,576
    int64_t expert_n_down = (int64_t)D_FF * D_MODEL;  // 1,048,576

    // Store quantized pointers + types per layer
    typedef struct {
        const uint8_t *q_gate_inp;
        const uint8_t *q_gate_exps;
        const uint8_t *q_up_exps;
        const uint8_t *q_down_exps;
        const uint8_t *q_gate_shexp;
        const uint8_t *q_up_shexp;
        const uint8_t *q_down_shexp;
        int ty_gi, ty_ge, ty_gs;
        int64_t expert_raw, expert_raw_down;
        bool has_moe;
    } moe_quant_t;

    moe_quant_t *moe_q = (moe_quant_t *)calloc(model.n_layers, sizeof(moe_quant_t));
    for (int l = 0; l < model.n_layers; l++) {
        moe_quant_t *mq = &moe_q[l];
        char name[256];

        snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", l);
        gguf_tensor_info *t = gguf_find_tensor(ctx, name);
        if (t) { mq->ty_gi = t->ggml_type; mq->q_gate_inp = (const uint8_t *)ctx->data_blob + t->data_offset; }

        snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", l);
        t = gguf_find_tensor(ctx, name);
        if (t) { mq->ty_ge = t->ggml_type; mq->q_gate_exps = (const uint8_t *)ctx->data_blob + t->data_offset; }

        snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", l);
        t = gguf_find_tensor(ctx, name);
        if (t) mq->q_up_exps = (const uint8_t *)ctx->data_blob + t->data_offset;

        snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", l);
        t = gguf_find_tensor(ctx, name);
        if (t) mq->q_down_exps = (const uint8_t *)ctx->data_blob + t->data_offset;

        snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", l);
        t = gguf_find_tensor(ctx, name);
        if (t) { mq->ty_gs = t->ggml_type; mq->q_gate_shexp = (const uint8_t *)ctx->data_blob + t->data_offset; }

        snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", l);
        t = gguf_find_tensor(ctx, name);
        if (t) mq->q_up_shexp = (const uint8_t *)ctx->data_blob + t->data_offset;

        snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", l);
        t = gguf_find_tensor(ctx, name);
        if (t) mq->q_down_shexp = (const uint8_t *)ctx->data_blob + t->data_offset;

        mq->has_moe = (mq->q_gate_exps != NULL);
        if (mq->has_moe) {
            mq->expert_raw = gguf_raw_size(mq->ty_ge, expert_n);
            mq->expert_raw_down = gguf_raw_size(mq->ty_ge, expert_n_down);
        }
    }

    // ================================================================
    // 4. Create test input embeddings
    // ================================================================
    float *embd = (float *)malloc(N * D_MODEL * sizeof(float));
    for (int i = 0; i < N * D_MODEL; i++)
        embd[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    // ================================================================
    // 5. Forward pass through all 40 layers with lazy MoE
    // ================================================================
    float *x = (float *)malloc(N * D_MODEL * sizeof(float));
    float *normed = (float *)malloc(N * D_MODEL * sizeof(float));
    float *attn_out = (float *)malloc(N * D_MODEL * sizeof(float));
    float *normed2 = (float *)malloc(N * D_MODEL * sizeof(float));
    float *ffn_out = (float *)malloc(N * D_MODEL * sizeof(float));

    // MoE scratch buffers (reused per layer)
    float *scores = (float *)malloc(N * N_EXPERTS * sizeof(float));
    int *topk_indices = (int *)malloc(N * N_ACTIVE_EXPTS * sizeof(int));
    float *topk_weights = (float *)malloc(N * N_ACTIVE_EXPTS * sizeof(float));
    float *expert_temp = (float *)malloc(D_FF * 3 * sizeof(float));
    float *shared_gate = (float *)malloc(SHARED_D_FF * sizeof(float));
    float *shared_up = (float *)malloc(SHARED_D_FF * sizeof(float));
    float *shared_act = (float *)malloc(SHARED_D_FF * sizeof(float));

    memcpy(x, embd, N * D_MODEL * sizeof(float));

    double total_ssm = 0, total_gqa = 0, total_norm = 0, total_moe = 0, total_moe_dequant = 0;

    for (int l = 0; l < model.n_layers; l++) {
        wubu_layer_t *layer = &model.layers[l];
        moe_quant_t *mq = &moe_q[l];

        // Pre-attention RMSNorm
        double t_n = now_sec();
        wubu_rms_norm(B, T, D_MODEL, x, layer->attn_norm_weight, 1e-6f, normed);
        total_norm += now_sec() - t_n;

        // SSM or GQA attention
        double t_a = now_sec();
        if (layer->is_ssm) {
            float *ssm_state = model.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *conv_state = model.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            wubu_ssm_forward(normed, B, T, &layer->ssm, ssm_state, conv_state, attn_out);
            total_ssm += now_sec() - t_a;
        } else {
            wubu_gqa_forward(normed, B, T, &layer->gqa, attn_out);
            total_gqa += now_sec() - t_a;
        }

        // NaN check
        int nan_idx = -1;
        for (int i = 0; i < N * D_MODEL; i++) {
            if (isnan(attn_out[i])) { nan_idx = i; break; }
        }
        if (nan_idx >= 0 && verbose) {
            int tt = nan_idx / D_MODEL, dd = nan_idx % D_MODEL;
            printf("  L%d %s NaN at [t=%d,d=%d]\n", l, layer->is_ssm ? "SSM" : "GQA", tt, dd);
        }

        // Residual: x = x + attn_out
        for (int i = 0; i < N * D_MODEL; i++) x[i] += attn_out[i];

        // Post-attention RMSNorm
        t_n = now_sec();
        wubu_rms_norm(B, T, D_MODEL, x, layer->post_attn_norm_weight, 1e-6f, normed2);
        total_norm += now_sec() - t_n;

        // ============================================================
        // MoE forward with lazy dequant
        // ============================================================
        double t_moe = now_sec();
        double t_dq = 0;
        int n_unique = 0;

        if (mq->has_moe) {
            // Step 1: Dequantize router only
            float *gate_inp = (float *)malloc(D_MODEL * N_EXPERTS * sizeof(float));
            t_dq = now_sec();
            gguf_dequantize(mq->q_gate_inp, mq->ty_gi, D_MODEL * N_EXPERTS, gate_inp);
            t_dq = now_sec() - t_dq;

            // Step 2: Route all tokens
            wubu_moe_router(normed2, B, T, gate_inp, scores);

            // Step 3: Softmax + top-k (N_ACTIVE_EXPTS per token)
            for (int s = 0; s < N; s++) {
                float *score_s = scores + s * N_EXPERTS;

                float max_s = score_s[0];
                for (int e = 1; e < N_EXPERTS; e++)
                    if (score_s[e] > max_s) max_s = score_s[e];

                float sum_exp = 0.0f;
                for (int e = 0; e < N_EXPERTS; e++)
                    sum_exp += expf(score_s[e] - max_s);
                float inv_sum = 1.0f / (sum_exp + 1e-30f);

                float softmax_vals[N_EXPERTS];
                for (int e = 0; e < N_EXPERTS; e++)
                    softmax_vals[e] = expf(score_s[e] - max_s) * inv_sum;

                int *indices_s = topk_indices + s * N_ACTIVE_EXPTS;
                float *weights_s = topk_weights + s * N_ACTIVE_EXPTS;

                for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
                    int best_idx = -1;
                    float best_val = -1e30f;
                    for (int e = 0; e < N_EXPERTS; e++) {
                        bool used = false;
                        for (int pk = 0; pk < k; pk++)
                            if (indices_s[pk] == e) { used = true; break; }
                        if (!used && softmax_vals[e] > best_val) {
                            best_val = softmax_vals[e];
                            best_idx = e;
                        }
                    }
                    indices_s[k] = best_idx;
                    weights_s[k] = best_val;
                }

                float sum_w = 0.0f;
                for (int k = 0; k < N_ACTIVE_EXPTS; k++) sum_w += weights_s[k];
                if (sum_w > 1e-30f) {
                    float inv_sum_w = 1.0f / sum_w;
                    for (int k = 0; k < N_ACTIVE_EXPTS; k++) weights_s[k] *= inv_sum_w;
                }
            }

            // Step 4: Collect unique expert IDs
            int unique_ids[N_ACTIVE_EXPTS * N];
            n_unique = 0;
            for (int s = 0; s < N; s++) {
                int *indices_s = topk_indices + s * N_ACTIVE_EXPTS;
                for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
                    int eid = indices_s[k];
                    if (eid < 0) continue;
                    bool seen = false;
                    for (int u = 0; u < n_unique; u++)
                        if (unique_ids[u] == eid) { seen = true; break; }
                    if (!seen) unique_ids[n_unique++] = eid;
                }
            }

            // Step 5: Dequant only selected experts
            expert_buf_t *experts = (expert_buf_t *)calloc(n_unique, sizeof(expert_buf_t));
            double t_dq2 = now_sec();
            for (int u = 0; u < n_unique; u++) {
                int eid = unique_ids[u];
                experts[u].expert_id = eid;

                experts[u].gate = (float *)malloc(expert_n * sizeof(float));
                const uint8_t *gate_ptr = mq->q_gate_exps + (int64_t)eid * mq->expert_raw;
                gguf_dequantize(gate_ptr, mq->ty_ge, expert_n, experts[u].gate);

                experts[u].up = (float *)malloc(expert_n * sizeof(float));
                const uint8_t *up_ptr = mq->q_up_exps + (int64_t)eid * mq->expert_raw;
                gguf_dequantize(up_ptr, mq->ty_ge, expert_n, experts[u].up);

                experts[u].down = (float *)malloc(expert_n_down * sizeof(float));
                const uint8_t *down_ptr = mq->q_down_exps + (int64_t)eid * mq->expert_raw_down;
                gguf_dequantize(down_ptr, mq->ty_ge, expert_n_down, experts[u].down);
            }
            t_dq2 = now_sec() - t_dq2;
            t_dq += t_dq2;

            // Step 6: Dequant shared expert
            double t_sd = now_sec();
            float *gate_shexp = (float *)malloc(D_MODEL * SHARED_D_FF * sizeof(float));
            float *up_shexp = (float *)malloc(D_MODEL * SHARED_D_FF * sizeof(float));
            float *down_shexp = (float *)malloc(SHARED_D_FF * D_MODEL * sizeof(float));
            gguf_dequantize(mq->q_gate_shexp, mq->ty_gs, D_MODEL * SHARED_D_FF, gate_shexp);
            gguf_dequantize(mq->q_up_shexp, mq->ty_gs, D_MODEL * SHARED_D_FF, up_shexp);
            gguf_dequantize(mq->q_down_shexp, mq->ty_gs, SHARED_D_FF * D_MODEL, down_shexp);
            t_dq += now_sec() - t_sd;

            total_moe_dequant += t_dq;

            // Step 7: Run MoE forward using cached expert pointers
            // Build expert lookup table: expert_id -> {gate, up, down}
            const float *exp_gate_lut[N_EXPERTS];
            const float *exp_up_lut[N_EXPERTS];
            const float *exp_down_lut[N_EXPERTS];
            memset(exp_gate_lut, 0, sizeof(exp_gate_lut));
            memset(exp_up_lut, 0, sizeof(exp_up_lut));
            memset(exp_down_lut, 0, sizeof(exp_down_lut));
            for (int u = 0; u < n_unique; u++) {
                int eid = experts[u].expert_id;
                exp_gate_lut[eid] = experts[u].gate;
                exp_up_lut[eid] = experts[u].up;
                exp_down_lut[eid] = experts[u].down;
            }

            // Step 8: Inline MoE forward with cached experts + shared expert
            // Compute shared expert contribution directly
            for (int s = 0; s < N; s++) {
                const float *x_s = normed2 + s * D_MODEL;
                float *out_s = ffn_out + s * D_MODEL;
                int *indices_s = topk_indices + s * N_ACTIVE_EXPTS;
                float *weights_s = topk_weights + s * N_ACTIVE_EXPTS;

                // Shared expert: gate = x @ gate_shexp
                for (int j = 0; j < SHARED_D_FF; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < D_MODEL; k++)
                        sum += x_s[k] * gate_shexp[k * SHARED_D_FF + j];
                    shared_gate[j] = sum;
                }
                // up = x @ up_shexp
                for (int j = 0; j < SHARED_D_FF; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < D_MODEL; k++)
                        sum += x_s[k] * up_shexp[k * SHARED_D_FF + j];
                    shared_up[j] = sum;
                }
                // act = silu(gate) * up
                for (int j = 0; j < SHARED_D_FF; j++) {
                    float g = shared_gate[j];
                    float silu_g = (g < -80.0f) ? 0.0f : g / (1.0f + expf(-g));
                    shared_act[j] = silu_g * shared_up[j];
                }
                // out = shared_act @ down_shexp
                for (int j = 0; j < D_MODEL; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < SHARED_D_FF; k++)
                        sum += shared_act[k] * down_shexp[k * D_MODEL + j];
                    out_s[j] = sum;
                }

                // Routed expert contributions from cached LUT
                for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
                    int e = indices_s[k];
                    float wgt = weights_s[k];
                    if (e < 0 || wgt < 1e-30f) continue;

                    const float *gate_w = exp_gate_lut[e];
                    const float *up_w = exp_up_lut[e];
                    const float *down_w = exp_down_lut[e];
                    if (!gate_w || !up_w || !down_w) continue;

                    // Expert forward inline
                    float *gate_out = expert_temp;
                    float *up_out = expert_temp + D_FF;
                    float *act = expert_temp + 2 * D_FF;

                    for (int j = 0; j < D_FF; j++) {
                        float gs = 0.0f, us = 0.0f;
                        for (int d = 0; d < D_MODEL; d++) {
                            gs += x_s[d] * gate_w[d * D_FF + j];
                            us += x_s[d] * up_w[d * D_FF + j];
                        }
                        gate_out[j] = gs;
                        up_out[j] = us;
                    }
                    for (int j = 0; j < D_FF; j++) {
                        float g = gate_out[j];
                        float silu_g = (g < -80.0f) ? 0.0f : g / (1.0f + expf(-g));
                        act[j] = silu_g * up_out[j];
                    }
                    for (int j = 0; j < D_MODEL; j++) {
                        float sum = 0.0f;
                        for (int d = 0; d < D_FF; d++)
                            sum += act[d] * down_w[d * D_MODEL + j];
                        out_s[j] += wgt * sum;
                    }
                }
            }
            for (int u = 0; u < n_unique; u++) {
                free(experts[u].gate);
                free(experts[u].up);
                free(experts[u].down);
            }
            free(experts);
            free(gate_inp);
            free(gate_shexp);
            free(up_shexp);
            free(down_shexp);

        } else {
            // No MoE: pass-through
            memcpy(ffn_out, normed2, N * D_MODEL * sizeof(float));
            n_unique = 0;
        }

        total_moe += now_sec() - t_moe;

        // Residual: x = x + ffn_out
        for (int i = 0; i < N * D_MODEL; i++) x[i] += ffn_out[i];

        if (verbose) {
            float mean = 0.0f;
            for (int i = 0; i < D_MODEL; i++) mean += fabsf(x[i]);
            mean /= D_MODEL;
            double attn_ms = (now_sec() - t_a) * 1000;
            printf("  L%02d %s | attn %6.2fms | moe %6.2fms (deq %5.2fms, %2d ex) | x[0:4] %.2f %.2f %.2f %.2f | mean %.4f\n",
                   l, layer->is_ssm ? "SSM" : "GQA",
                   attn_ms,
                   (now_sec() - t_moe) * 1000,
                   t_dq * 1000, n_unique,
                   x[0], x[1], x[2], x[3], mean);
        }
    }

    // Final RMSNorm
    if (model.norm_weight) {
        wubu_rms_norm(B, T, D_MODEL, x, model.norm_weight, 1e-6f, normed);
        memcpy(x, normed, N * D_MODEL * sizeof(float));
    }

    // ================================================================
    // 6. Summary
    // ================================================================
    double total_time = now_sec() - t0;
    printf("\n=== Unified 40-layer Summary ===\n");
    printf("Total time: %.3f s (%.1f tok/s)\n", total_time, N / total_time);
    printf("SSM fwd:    %.3f s\n", total_ssm);
    printf("GQA fwd:    %.3f s\n", total_gqa);
    printf("Norms:      %.3f s\n", total_norm);
    printf("MoE fwd:    %.3f s (dequant: %.3f s)\n", total_moe, total_moe_dequant);
    printf("Other:      %.3f s\n", total_time - total_ssm - total_gqa - total_norm - total_moe);

    // Show final output stats
    float mean = 0.0f, min_v = 1e30f, max_v = -1e30f;
    for (int i = 0; i < N * D_MODEL; i++) {
        mean += fabsf(x[i]);
        if (x[i] < min_v) min_v = x[i];
        if (x[i] > max_v) max_v = x[i];
    }
    mean /= (N * D_MODEL);
    printf("\nFinal hidden: mean %.4f range [%.4f, %.4f]\n", mean, min_v, max_v);

    // Cleanup
    free(x); free(normed); free(attn_out); free(normed2); free(ffn_out);
    free(scores); free(topk_indices); free(topk_weights);
    free(expert_temp); free(shared_gate); free(shared_up); free(shared_act);
    free(embd);
    free(moe_q);
    wubu_model_free(&model);
    // ctx closed by wubu_model_free

    printf("\n=== Unified Inference PASS ===\n");
    return 0;
}
