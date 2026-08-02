/* test_evict2026.c -- Theme IO: 2026 eviction frontier mechanisms. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_evict2026.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_evict2026 (IO frontier) ===\n");

    /* SnapKV pooling */
    {
        float a[8] = { 0.1f, 0.9f, 0.2f, 0.3f, 0.05f, 0.8f, 0.4f, 0.6f };
        float out[4];
        int m = wubu_ev_pool_obs(a, 8, 2, out);
        CHECK(m == 4, "pooled count 4");
        NEAR(out[0], 0.9f, 1e-6f);
        NEAR(out[1], 0.3f, 1e-6f);
        NEAR(out[2], 0.8f, 1e-6f);
        NEAR(out[3], 0.6f, 1e-6f);
        CHECK(wubu_ev_pool_obs(NULL, 8, 2, out) == 0, "null rejected");
    }

    /* Proxy-token batch eviction */
    {
        float s[5] = { 0.8f, 0.1f, 0.5f, 0.2f, 0.9f };
        int keep[5];
        int k = wubu_ev_proxy_evict(s, 5, 3, keep);
        CHECK(k == 3, "keep 3");
        CHECK(keep[0] == 4 && keep[1] == 0 && keep[2] == 2, "top-3 by score");
        CHECK(wubu_ev_proxy_evict(s, 5, 0, keep) == 0, "keep 0");
        CHECK(wubu_ev_proxy_evict(s, 5, 9, keep) == 5, "keep capped at n");
    }

    /* InfiniPot novelty */
    {
        float proto[6] = { 1, 0, 0,  0, 1, 0 };  /* two prototypes, 3-d */
        NEAR(wubu_ev_novelty(proto, 2, 3, (float[]){ 1, 0, 0 }), 0.0f, 1e-5f);
        NEAR(wubu_ev_novelty(proto, 2, 3, (float[]){ 0, 0, 2 }), sqrtf(5.0f), 1e-4f);
        NEAR(wubu_ev_novelty(proto, 2, 3, (float[]){ 0.5f, 0.5f, 0 }), sqrtf(0.5f), 1e-4f);
    }

    /* HASHEVICT simhash + hamming */
    {
        float plane[4] = { 1, 1, 1, 1 };
        uint32_t a = wubu_ev_simhash((float[]){ 1, 1, 1, 1 }, 4, plane, 0);
        uint32_t b = wubu_ev_simhash((float[]){ -1, -1, -1, -1 }, 4, plane, 0);
        CHECK(a != 0 && b == 0, "simhash sign separation");
        CHECK(wubu_ev_hamming(a, b) >= 4, "opposite signs -> far hamming");
        CHECK(wubu_ev_hamming(a, a) == 0, "self hamming 0");
        uint32_t c = wubu_ev_simhash((float[]){ 1, 1, 1, -1 }, 4, plane, 0);
        CHECK(wubu_ev_hamming(a, c) == 1, "one-bit difference");
    }

    /* RocketKV two-stage */
    {
        float coarse[5] = { 0.9f, 0.1f, 0.8f, 0.2f, 0.7f };
        float qsim[5] = { 0.1f, 0.1f, 0.6f, 0.1f, 0.9f };
        int out[5];
        int k = wubu_ev_twostage(coarse, qsim, 5, 3, 2, out);
        /* stage1: {0,2,4}; stage2 by qsim: {4,2} */
        CHECK(k == 2 && out[0] == 4 && out[1] == 2, "two-stage selection");
    }

    /* Ada-KV head-adaptive budget */
    {
        float disp[3] = { 1.0f, 3.0f, 6.0f };  /* sum 10 */
        CHECK(wubu_ev_adakv_budget(disp, 3, 100, 0) == 10, "head0 gets 10");
        CHECK(wubu_ev_adakv_budget(disp, 3, 100, 1) == 30, "head1 gets 30");
        CHECK(wubu_ev_adakv_budget(disp, 3, 100, 2) == 60, "head2 gets 60");
        CHECK(wubu_ev_adakv_budget(disp, 3, 100, 5) == 0, "OOB head -> 0");
    }

    /* KeyDiff key-similarity */
    {
        CHECK(wubu_ev_keysim_redundant((float[]){ 1, 0 }, (float[]){ 1, 0 }, 2, 0.9f) == 1,
              "identical keys redundant");
        CHECK(wubu_ev_keysim_redundant((float[]){ 1, 0 }, (float[]){ 0, 1 }, 2, 0.9f) == 0,
              "orthogonal keys kept");
        CHECK(wubu_ev_keysim_redundant((float[]){ 1, 0 }, (float[]){ 0, 1 }, 2, -0.5f) == 1,
              "negative thresh -> everything redundant");
    }

    /* Sink discovery + semantic sponsor */
    {
        CHECK(wubu_ev_sink_pos((float[]){ 0.1f, 0.8f, 0.2f }, 3) == 1, "sink at 1");
        CHECK(wubu_ev_sink_pos((float[]){ 0.9f, 0.1f, 0.2f }, 3) == 0, "sink at 0");
        CHECK(wubu_ev_semantic_sponsor(0.8f, 0.5f) == 1, "sponsored");
        CHECK(wubu_ev_semantic_sponsor(0.2f, 0.5f) == 0, "not sponsored");
    }

    /* Loss bound + block drift */
    {
        NEAR(wubu_ev_loss_bound(2.0f, 10.0f), 0.2f, 1e-6f);
        NEAR(wubu_ev_loss_bound(10.0f, 5.0f), 1.0f, 1e-6f);  /* capped */
        NEAR(wubu_ev_block_drift(0.3f, 0.2f, 0.4f), 0.4f, 1e-6f); /* capped */
        NEAR(wubu_ev_block_drift(0.3f, 0.1f, 1.0f), 0.4f, 1e-6f);
    }

    /* Sink reservation + hybrid + streaming softmax + hysteresis + disparity */
    {
        CHECK(wubu_ev_reserve_sink(100, 4, 6) == 10, "reserve 10");
        CHECK(wubu_ev_reserve_sink(8, 4, 6) == 8, "reserve capped at budget");
        CHECK(wubu_ev_hybrid_choose(0.2f, 1.0f, 0.5f) == 1, "low value -> evict");
        CHECK(wubu_ev_hybrid_choose(0.9f, 1.0f, 0.5f) == 0, "high value -> compress");
        float mx = -1e30f, sum = 0;
        float w1 = wubu_ev_stream_softmax(&mx, &sum, 0.0f);
        float w2 = wubu_ev_stream_softmax(&mx, &sum, 1.0f);
        NEAR(w1, 1.0f, 1e-5f);               /* the only token -> weight 1 */
        NEAR(w2, 1.0f / (1.0f + expf(-1.0f)), 1e-4f);  /* softmax(1|{0,1}) */
        /* hysteresis */
        CHECK(wubu_ev_hysteresis(0.55f, 0.5f, 0.1f, 1) == 1, "hyst keeps within band");
        CHECK(wubu_ev_hysteresis(0.45f, 0.5f, 0.1f, 0) == 0, "hyst rejects below band");
        CHECK(wubu_ev_hysteresis(0.45f, 0.5f, 0.1f, 1) == 1, "hyst sticks while above lower edge");
        /* head disparity */
        NEAR(wubu_ev_head_disparity((float[]){ 1, 2, 4 }, 3), 4.0f, 1e-6f);
        NEAR(wubu_ev_head_disparity((float[]){ 2, 2 }, 2), 1.0f, 1e-6f);
        NEAR(wubu_ev_head_disparity((float[]){ 0, 2 }, 2), 1e9f, 1e3f);
    }

    if (failures == 0) printf("ALL EVICT2026 TESTS PASSED\n");
    else printf("%d EVICT2026 FAILURES\n", failures);
    return failures ? 1 : 0;
}
