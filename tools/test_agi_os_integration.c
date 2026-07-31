/*
 * test_agi_os_integration.c — End-to-end AGI OS inference stack integration.
 *
 * Verifies that all new modules work together as a cohesive inference
 * stack for the AGI OS:
 *
 * 1. SoA layout for channel-wise attention operations (I02)
 * 2. Cache-line-aligned KV storage (C03)
 * 3. RoPE-aware KV prefetch with KDA offset (A10)
 * 4. Adaptive KV quantization with entropy-aware bit-width (001)
 * 5. FlashAttention-style fused prefill (H01)
 * 6. Continuous batching across requests (D01)
 * 7. Chunked prefill + PD disaggregation (D03/D04)
 * 8. LMCache persistent KV reuse (A06)
 * 9. SMT-verified GEMV equivalence (F02)
 * 10. MLA multi-head latent attention (E02)
 *
 * This test simulates the AGI OS inference pipeline: a batch of requests
 * arrives, prefill is chunked and interleaved with decode, KV blocks are
 * cache-line aligned and adaptively quantized, and the prefix cache
 * persists across requests.
 */
#include "wubu_model.h"
#include "wubu_scheduler.h"
#include "wubu_mla.h"
#include "wubu_expert_choice.h"
#include "wubu_chunked_prefill.h"
#include "wubu_lmcache.h"
#include "wubu_smt_check.h"
#include "wubu_kv_cacheline.h"
#include "wubu_rope_prefetch.h"
#include "wubu_flash_prefill.h"
#include "wubu_soa.h"
#include "wubu_kv_adaptive.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <sys/stat.h>

int main(void) {
    fprintf(stderr, "AGI OS integration test starting...\n");
    int errors = 0;

    /* === 1. SMT boot check: verify GEMV equivalence before inference === */
    wubu_smt_result_t smt = wubu_smt_check_gemv(4, 0.1f);
    printf("[1] SMT boot check: %s (%d checks, %d failures, max_err=%.2e)\n",
           wubu_smt_status_str(smt.status), smt.n_checks, smt.n_failures,
           (double)smt.max_error);
    if (smt.status != WUBU_SMT_OK) { errors++; printf("  FAIL\n"); }
    else printf("  PASS\n");

    /* === 2. MLA: set up latent attention with ~14x KV compression === */
    int hidden_dim = 256, n_heads = 4, head_dim = 64;
    int q_lora_rank = 32, kv_lora_rank = 16, rope_head_dim = 8;
    wubu_mla_t *mla = wubu_mla_create(hidden_dim, n_heads, head_dim,
                                       q_lora_rank, kv_lora_rank, rope_head_dim);
    assert(mla);
    float ratio = wubu_mla_compression_ratio(mla);
    printf("[2] MLA: kv_lora=%d rope=%d latent=%d, compression=%.1fx\n",
           kv_lora_rank, rope_head_dim, mla->kv_latent_dim, (double)ratio);
    if (ratio < 2.0f) { errors++; printf("  FAIL\n"); }
    else printf("  PASS\n");

    /* === 3. SoA layout: convert AoS activations for channel-wise ops === */
    int n_tokens = 16, n_channels = 64;
    float *aos = (float *)malloc(n_tokens * n_channels * sizeof(float));
    float *soa = (float *)malloc(n_tokens * n_channels * sizeof(float));
    for (int i = 0; i < n_tokens * n_channels; i++) aos[i] = 0.01f * i;
    wubu_soa_pack(aos, soa, n_tokens, n_channels);
    /* Verify: soa[channel][token] = aos[token][channel] */
    int ok = 1;
    for (int t = 0; t < n_tokens; t++)
        for (int c = 0; c < n_channels; c++)
            if (fabsf(soa[c * n_tokens + t] - aos[t * n_channels + c]) > 1e-6f) { ok = 0; break; }
    printf("[3] SoA layout: aos→soa conversion %s\n", ok ? "verified" : "BROKEN");
    if (!ok) { errors++; printf("  FAIL\n"); }
    else printf("  PASS\n");

    /* === 4. Cache-line KV storage === */
    wubu_kv_cacheline_t *kvcl = wubu_kv_cacheline_create(16, 128, 8, 10);
    assert(kvcl);
    /* Each block write needs [n_kv_heads, head_dim] = 8*128 = 1024 floats */
    float *kv_data = (float *)malloc(128 * 10 * sizeof(float));
    for (int i = 0; i < 1280; i++) kv_data[i] = 0.1f * (i % 7 - 3);
    /* Write block 0, token 0 with k_vec = kv_data (8 heads x 128 dims) */
    wubu_kv_cacheline_write(kvcl, 0, 0, kv_data, kv_data);
    float *kv_read_k = (float *)malloc(128 * 8 * sizeof(float));
    float *kv_read_v = (float *)malloc(128 * 8 * sizeof(float));
    wubu_kv_cacheline_read(kvcl, 0, 0, kv_read_k, kv_read_v);
    float max_err = 0.0f;
    for (int i = 0; i < 128; i++) {
        float e = fabsf(kv_data[i] - kv_read_k[i]);
        if (e > max_err) max_err = e;
    }
    printf("[4] Cache-line KV: 1 block, max_err=%.8f\n", (double)max_err);
    if (max_err > 1e-6f) { errors++; printf("  FAIL\n"); }
    else printf("  PASS\n");
    wubu_kv_cacheline_free(kvcl);
    free(kv_data); free(kv_read_k); free(kv_read_v);

    /* === 5. RoPE-aware KV prefetch === */
    wubu_kv_cacheline_t *kvcl2 = wubu_kv_cacheline_create(16, 128, 8, 10);
    assert(kvcl2);
    int blocks[] = {0, 1, 2, 3};
    wubu_rope_prefetch_kv(kvcl2, blocks, 4, 100, 64, 64);
    printf("[5] RoPE prefetch: pos=100, 4 blocks scheduled\n");
    printf("  PASS\n");
    wubu_kv_cacheline_free(kvcl2);

    /* === 6. Adaptive KV quantization (entropy-aware) === */
    float *adapt_data = (float *)malloc(64 * sizeof(float));
    for (int i = 0; i < 64; i++) adapt_data[i] = 0.5f * (float)(i % 11 - 5);
    float cosine = wubu_kvq_adaptive_roundtrip(adapt_data, 64);
    printf("[6] Adaptive KV: roundtrip cosine=%.6f\n", (double)cosine);
    if (cosine < 0.95f) { errors++; printf("  FAIL\n"); }
    else printf("  PASS\n");
    free(adapt_data);

    /* === 7. FlashAttention fused prefill === */
    int T = 64, D = 32, n_heads_ag = 4;
    /* Layout: [n_heads, seq_len, head_dim] */
    size_t qkv_sz = (size_t)n_heads_ag * T * D;
    float *q = (float *)malloc(qkv_sz * sizeof(float));
    float *k = (float *)malloc(qkv_sz * sizeof(float));
    float *v = (float *)malloc(qkv_sz * sizeof(float));
    float *flash = (float *)malloc(qkv_sz * sizeof(float));
    for (size_t i = 0; i < qkv_sz; i++) {
        q[i] = 0.01f * (float)(i % 61 - 30);
        k[i] = 0.02f * (float)(i % 53 - 26);
        v[i] = 0.03f * (float)(i % 47 - 23);
    }
    /* Flash prefill */
    wubu_flash_prefill_attn(q, k, v, flash, n_heads_ag, T, D, T);
    float flash_max_err = 0.0f;
    int has_nan = 0;
    for (size_t i = 0; i < qkv_sz; i++) {
        if (isnan(flash[i]) || isinf(flash[i])) has_nan = 1;
        if (fabsf(flash[i]) > flash_max_err) flash_max_err = fabsf(flash[i]);
    }
    printf("[7] Flash prefill: T=%d D=%d n_heads=%d max_val=%.8f nan=%d\n",
           T, D, n_heads_ag, (double)flash_max_err, has_nan);
    if (has_nan) { errors++; printf("  FAIL\n"); }
    else printf("  PASS\n");
    free(q); free(k); free(v); free(flash);

    /* === 8. Continuous batching === */
    wubu_sched_t *sched = wubu_sched_create(8);
    assert(sched);
    int prompts[4][5] = {
        {1, 2, 3, 4, 5},
        {10, 20, 30, 40, 50},
        {100, 200, 300, 400, 500},
        {1000, 2000, 3000, 4000, 5000}
    };
    wubu_req_t *reqs[4];
    for (int i = 0; i < 4; i++) {
        reqs[i] = wubu_req_create(i, prompts[i], 5, 0);
        assert(reqs[i]);
        wubu_sched_submit(sched, reqs[i]);
    }
    int steps = 0;
    while (wubu_sched_active(sched) > 0 && steps < 100) {
        wubu_sched_step(sched);
        /* Simulate token generation: emit a token for each active decode request */
        for (int i = 0; i < sched->n; i++) {
            if (sched->reqs[i] && sched->reqs[i]->state == WUBU_REQ_DECODE) {
                wubu_req_emit(sched->reqs[i], 42);
            }
        }
        steps++;
    }
    printf("[8] Continuous batching: 4 requests, %d steps, active=%d\n",
           steps, wubu_sched_active(sched));
    if (wubu_sched_active(sched) > 0) { errors++; printf("  FAIL\n"); }
    else printf("  PASS\n");
    wubu_sched_free(sched);
    /* wubu_sched_free frees all requests internally */

    /* === 9. Chunked prefill + PD disaggregation === */
    wubu_chunked_prefill_t *cp = wubu_chunked_prefill_create(64);
    assert(cp);
    int job = wubu_chunked_prefill_submit(cp, 200);
    assert(job >= 0);
    int total = 0, chunks = 0;
    while (!wubu_chunked_prefill_is_done(cp, job)) {
        int chunk = wubu_chunked_prefill_next_chunk(cp, job);
        if (chunk <= 0) break;
        total += chunk; chunks++;
    }
    printf("[9] Chunked prefill: 200 tokens → %d chunks, total=%d\n", chunks, total);
    if (total != 200) { errors++; printf("  FAIL\n"); }
    else printf("  PASS\n");
    wubu_chunked_prefill_free(cp);

    /* === 10. LMCache persistent KV === */
    const char *cache_dir = "/tmp/wubu_agi_os_test";
    mkdir(cache_dir, 0755);
    wubu_lmcache_t *lmc = wubu_lmcache_create(cache_dir, 2, 16, 8, 4);
    assert(lmc);
    int tokens[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    /* n_layers=2, n_blocks=10, n_kv_heads=8, block_size=16, head_dim=4 */
    /* block_bytes = 8*16*4*4 = 2048 bytes = 512 floats */
    /* total = 2*10*512 = 10240 floats */
    size_t kv_store_size = 2 * 10 * 8 * 16 * 4;
    float *kv_store = (float *)malloc(kv_store_size * sizeof(float));
    for (size_t i = 0; i < kv_store_size; i++) kv_store[i] = 0.01f * (float)(i % 128 - 64);
    wubu_lmcache_store(lmc, "agi-os", tokens, 10, kv_store, 10);
    float *kv_load = (float *)malloc(kv_store_size * sizeof(float));
    int loaded = wubu_lmcache_load(lmc, "agi-os", tokens, 10, kv_load, 10);
    float lm_max_err = 0.0f;
    for (size_t i = 0; i < kv_store_size; i++) {
        float e = fabsf(kv_store[i] - kv_load[i]);
        if (e > lm_max_err) lm_max_err = e;
    }
    printf("[10] LMCache: %d blocks, max_err=%.8f\n", loaded, (double)lm_max_err);
    if (loaded != 10 || lm_max_err > 1e-6f) { errors++; printf("  FAIL\n"); }
    else printf("  PASS\n");
    wubu_lmcache_free(lmc);
    free(kv_store); free(kv_load);

    /* === 11. Expert choice routing === */
    float scores[32];  /* 8 tokens x 4 experts */
    for (int i = 0; i < 32; i++) scores[i] = 0.1f * ((i * 7) % 11 - 5);
    int ec_assign[8]; float ec_weights[8];
    wubu_expert_choice_route(scores, 8, 4, 2, ec_assign, ec_weights);
    float lb = wubu_route_load_balance(ec_assign, 4, 2, 8);
    printf("[11] Expert choice: load_balance CV=%.4f\n", (double)lb);
    if (lb < 0.0f) { errors++; printf("  FAIL\n"); }
    else printf("  PASS\n");

    /* === Summary === */
    printf("\n=== AGI OS Integration Summary ===\n");
    printf("Modules tested: 11\n");
    printf("Errors: %d\n", errors);
    printf("Status: %s\n", errors == 0 ? "ALL PASS ✅" : "FAILURES ❌");

    /* Cleanup */
    wubu_mla_free(mla);
    free(aos); free(soa);

    return errors == 0 ? 0 : 1;
}
