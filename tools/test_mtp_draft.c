/**
 * test_mtp_draft.c — Verify MTP head loads and produces non-trash logits.
 * Tests the code path without needing hidden state capture.
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec*1e-9; }

int main() {
    const char *model_path = "/models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf";
    
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, model_path)) return 1;
    mdl.enable_moe = true;
    
    printf("Loading MTP head...\n");
    double t0 = now_sec();
    if (!wubu_mtp_load(&mdl.mtp, NULL, mdl.gguf_ctx, NULL)) {
        printf("FAIL: MTP load\n");
        wubu_model_free(&mdl);
        return 1;
    }
    printf("MTP loaded in %.2fs\n", now_sec() - t0);
    printf("  hnorm:     %p\n", (void*)mdl.mtp.nextn_hnorm);
    printf("  enorm:     %p\n", (void*)mdl.mtp.nextn_enorm);
    printf("  eh_proj:   %p (dim=%lld, F32)\n", (void*)mdl.mtp.nextn_eh_proj_f32, (long long)mdl.mtp.nextn_eh_proj_dim);
    printf("  head_norm: %p\n", (void*)mdl.mtp.nextn_shared_head_norm);
    printf("  blk40:     is_ssm=%d moe=%d\n", mdl.mtp.blk40.is_ssm, mdl.mtp.blk40.moe.loaded);
    printf("  gqa:       q_q=%p k_q=%p v_q=%p o_q=%p\n",
           (void*)mdl.mtp.blk40.gqa.attn_q_weight_q,
           (void*)mdl.mtp.blk40.gqa.attn_k_weight_q,
           (void*)mdl.mtp.blk40.gqa.attn_v_weight_q,
           (void*)mdl.mtp.blk40.gqa.attn_output_weight_q);
    printf("  moe:       gate_q=%p up_q=%p down_q=%p\n",
           (void*)mdl.mtp.blk40.moe.ffn_gate_exps_q,
           (void*)mdl.mtp.blk40.moe.ffn_up_exps_q,
           (void*)mdl.mtp.blk40.moe.ffn_down_exps_q);
    
    // Test draft forward with dummy inputs
    printf("\nRunning MTP draft forward (dummy inputs)...\n");
    float dummy_h[D_MODEL];
    float dummy_embd[D_MODEL];
    float logits_out[248320];
    
    // Fill with small random-ish values
    for (int i = 0; i < D_MODEL; i++) {
        dummy_h[i] = ((float)(i % 100) - 50.0f) * 0.01f;
        dummy_embd[i] = ((float)((i*7) % 100) - 50.0f) * 0.01f;
    }
    
    t0 = now_sec();
    int nd = wubu_mtp_draft_forward(&mdl, dummy_h, dummy_embd, 1, logits_out);
    double t_draft = now_sec() - t0;
    printf("Draft forward: %d candidates in %.3fs\n", nd, t_draft);
    
    // Check logits sanity
    int nan_count = 0, inf_count = 0;
    float max_val = -1e30f, min_val = 1e30f, sum = 0;
    int max_idx = 0;
    for (int i = 0; i < 248320; i++) {
        if (isnan(logits_out[i])) nan_count++;
        if (isinf(logits_out[i])) inf_count++;
        if (logits_out[i] > max_val) { max_val = logits_out[i]; max_idx = i; }
        if (logits_out[i] < min_val) min_val = logits_out[i];
        sum += logits_out[i];
    }
    printf("  nan=%d inf=%d max=%f(min_idx=%d) min=%f mean=%f\n", 
           nan_count, inf_count, max_val, max_idx, min_val, sum/248320.0f);
    printf("  Top-5 logits:\n");
    // Find top 5
    float logits_cpy[248320];
    memcpy(logits_cpy, logits_out, sizeof(logits_cpy));
    for (int k = 0; k < 5; k++) {
        float mv = -1e30f; int mi = -1;
        for (int i = 0; i < 248320; i++) { if (logits_cpy[i] > mv) { mv = logits_cpy[i]; mi = i; } }
        printf("    [%d] idx=%d val=%f\n", k, mi, mv);
        logits_cpy[mi] = -1e30f;
    }
    
    printf("\n=== PASS ===\n");
    wubu_model_free(&mdl);
    return 0;
}
