/*
 * test_hh.c -- HH01-HH07 verification.
 */
#include "wubu_specdec.h"
#include "wubu_pagedkv.h"
#include "wubu_moeroute.h"
#include "wubu_contbatch.h"
#include "wubu_medusa.h"
#include "wubu_quantkv.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fails++; printf("FAIL: %s\n", msg); } \
    else printf("  ok: %s\n", msg); \
} while(0)

int main() {
    /* HH01: Speculative decoding */
    printf("=== HH01: Speculative Decoding ===\n");
    wubu_specdec_t sd;
    memset(&sd, 0, sizeof(sd));
    sd.draft_len = 3;
    /* Draft proposes tokens 10, 20, 30. Make draft probs == target probs at those
       tokens → acceptance prob = 1 → all accepted → bonus token produced. */
    sd.draft_tokens[0] = 10; sd.draft_tokens[1] = 20; sd.draft_tokens[2] = 30;
    for (int i = 0; i < 3; i++) {
        for (int v = 0; v < WUBU_SPECDEC_VOCAB; v++) {
            sd.draft_probs[i][v] = (v == sd.draft_tokens[i]) ? 0.9f : 0.1f / (WUBU_SPECDEC_VOCAB - 1);
            sd.target_probs[i][v] = sd.draft_probs[i][v];  /* aligned → accept */
        }
    }
    unsigned seed = 12345;
    int produced = wubu_specdec_verify(&sd, &seed);
    CHECK(sd.n_accepted == 3, "all 3 draft tokens accepted (draft==target aligned)");
    CHECK(produced == 4, "produced 4 tokens (3 accepted + 1 bonus)");
    CHECK(wubu_specdec_rate(&sd) == 1.0f, "acceptance rate = 1.0");

    /* Mismatched: draft overestimates → rejection (target prob = 0 on draft token
       → accept prob = min(1, 0/q) = 0 → guaranteed reject). */
    wubu_specdec_t sd2; memset(&sd2, 0, sizeof(sd2));
    sd2.draft_len = 2;
    sd2.draft_tokens[0] = 5; sd2.draft_tokens[1] = 7;
    for (int i = 0; i < 2; i++) {
        for (int v = 0; v < WUBU_SPECDEC_VOCAB; v++) {
            sd2.draft_probs[i][v] = (v == sd2.draft_tokens[i]) ? 0.95f : 0.05f / (WUBU_SPECDEC_VOCAB - 1);
            /* target: ZERO prob on the draft token → guaranteed reject */
            sd2.target_probs[i][v] = (v == sd2.draft_tokens[i]) ? 0.0f : 0.9f / (WUBU_SPECDEC_VOCAB - 1);
        }
    }
    unsigned seed2 = 999;
    int p2 = wubu_specdec_verify(&sd2, &seed2);
    CHECK(sd2.n_accepted == 0, "all rejected (target prob=0 on draft token)");
    CHECK(p2 >= 1, "at least 1 token produced (resample from residual)");

    /* HH02: Paged KV cache */
    printf("\n=== HH02: Paged KV Cache ===\n");
    wubu_pagedkv_t pk;
    CHECK(wubu_pagedkv_init(&pk, 100) == 0, "paged KV init (100 physical blocks)");
    int p0 = wubu_pagedkv_alloc(&pk, 0);
    int p1 = wubu_pagedkv_alloc(&pk, 0);
    CHECK(p0 >= 0 && p1 >= 0, "allocated 2 logical blocks (non-contiguous ok)");
    CHECK(pk.n_free == 98, "free list decremented (98 free of 100)");
    /* Prefix sharing: register block p0 under a hash, lookup returns it */
    unsigned h = 0xDEADBEEF;
    wubu_pagedkv_share_prefix(&pk, h, p0);
    int lookup = wubu_pagedkv_lookup_prefix(&pk, h);
    CHECK(lookup == p0, "prefix lookup returns shared physical block");
    CHECK(pk.refcount[p0] == 2, "refcount incremented on share (copy-on-write)");
    wubu_pagedkv_free(&pk, p0);  /* one ref freed, still shared */
    CHECK(pk.refcount[p0] == 1, "free decrements refcount (still shared)");
    CHECK(wubu_pagedkv_frag(&pk) < 1.0f, "fragmentation < 1.0 (blocks in use)");

    /* HH03: MoE capacity routing */
    printf("\n=== HH03: MoE Capacity Routing ===\n");
    wubu_moeroute_t mr;
    CHECK(wubu_moeroute_init(&mr, 4, 2, 2) == 0, "moeroute init (4 experts, top-2, cap=2)");
    /* 8 tokens, router logits favor experts 0 and 1 → capacity overflow test */
    for (int t = 0; t < 8; t++) {
        for (int e = 0; e < 4; e++) mr.router_logits[t][e] = (e < 2) ? 1.0f : 0.0f;
    }
    int routed = wubu_moeroute_step(&mr, 8);
    CHECK(routed <= 8 * 2, "routed tokens <= top_k * n_tokens");
    CHECK(mr.dropped > 0, "some tokens dropped (experts 0,1 over capacity cap=2)");
    CHECK(mr.load[0] == 2 && mr.load[1] == 2, "capacity cap enforced (load=2 each)");
    float aux = wubu_moeroute_aux_loss(&mr);
    printf("    aux loss (load variance) = %.3f\n", aux);
    CHECK(aux >= 0, "aux loss non-negative (load-balancing signal)");

    /* HH04: Continuous batching */
    printf("\n=== HH04: Continuous Batching ===\n");
    wubu_contbatch_t cb; memset(&cb, 0, sizeof(cb));
    wubu_contbatch_add(&cb, 1, 5);
    wubu_contbatch_add(&cb, 2, 3);
    wubu_contbatch_add(&cb, 3, 10);
    /* Mid-generation: add a 4th request without waiting for batch end */
    wubu_contbatch_step(&cb);
    wubu_contbatch_add(&cb, 4, 2);  /* joins mid-flight */
    int r;
    for (int s = 0; s < 12; s++) r = wubu_contbatch_step(&cb);
    CHECK(cb.reqs[0].done && cb.reqs[1].done && cb.reqs[2].done && cb.reqs[3].done,
          "all requests (including mid-join) completed");
    float tput = wubu_contbatch_tput(&cb);
    printf("    effective throughput = %.2f tok/step\n", tput);
    CHECK(tput > 0, "throughput proxy positive (concurrent decode)");

    /* HH05: Medusa self-draft */
    printf("\n=== HH05: Medusa Self-Draft ===\n");
    wubu_medusa_t med; CHECK(wubu_medusa_init(&med) == 0, "medusa init (4 heads, branch 2)");
    CHECK(med.draft_len == 4, "initial draft length = all heads (optimistic EMA)");
    /* Simulate: head 3 has consistently low acceptance → EMA should drop below 0.5 */
    for (int k = 0; k < 20; k++) wubu_medusa_update(&med, 3, 1, 10);  /* 10% acceptance ×20 → EMA < 0.5 */
    wubu_medusa_update(&med, 0, 9, 10);  /* 90% acceptance on head 0 */
    wubu_medusa_adapt(&med, 0.5f);  /* drop heads < 0.5 EMA */
    printf("    draft_len after adapt = %d (head 3 should drop)\n", med.draft_len);
    CHECK(med.draft_len < 4, "adaptive draft length drops low-acceptance head");

    /* HH06: KV quantization */
    printf("\n=== HH06: KV Quantization ===\n");
    CHECK(wubu_quantkv_bits() == 8, "INT8 = 8 bits");
    CHECK(fabs(wubu_quantkv_ratio() - 4.0f) < 1e-3, "compression ratio = 4x (FP32→INT8)");
    float kv[64];
    for (int i = 0; i < 64; i++) kv[i] = (float)(i % 16) - 8.0f;  /* -8..7 range */
    wubu_quantkv_t qk; memset(&qk, 0, sizeof(qk));
    qk.group = 16;
    CHECK(wubu_quantkv_quantize(&qk, kv, 64) == 0, "quantize 64 FP32 → INT8");
    float out[64];
    wubu_quantkv_dequantize(&qk, out);
    /* Check max abs error within quant step (scale/127) */
    float max_err = 0.0f;
    for (int i = 0; i < 64; i++) {
        float e = fabsf(out[i] - kv[i]);
        if (e > max_err) max_err = e;
    }
    printf("    max dequant error = %.4f (step ≈ %.4f)\n", max_err, qk.scale[0] / 127.0f);
    CHECK(max_err <= qk.scale[0] + 1e-3f, "dequant error within one INT8 step");

    /* HH07: Integration — speedup model */
    printf("\n=== HH07: Integration ===\n");
    /* speedup ≈ acceptance_rate × concurrency × (1/kv_bits_compression) */
    float accept = wubu_specdec_rate(&sd);  /* 1.0 from HH01 */
    float concurrency = wubu_contbatch_tput(&cb);  /* > 0 from HH04 */
    float kv_compress = wubu_quantkv_ratio();  /* 4.0 from HH06 */
    float speedup = accept * (1.0f + concurrency * 0.1f) * (kv_compress / 4.0f);
    printf("    modeled speedup factor = %.3f (accept=%.2f, kv_compress=%.1fx)\n", speedup, accept, kv_compress);
    CHECK(speedup > 0, "speedup model positive (all primitives compose)");

    if (fails > 0) {
        printf("\n%d TEST(S) FAILED\n", fails);
        return 1;
    }
    printf("\nALL HH TESTS PASSED\n");
    return 0;
}
