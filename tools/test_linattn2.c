/* test_linattn2.c -- Theme IU complete: the linear-attention frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_linattn2.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_linattn2 (IU complete) ===\n");
    {
        float st[2] = { 0, 0 };
        float k[2] = { 1, 0 }, v[2] = { 5, 0 };
        wubu_la2_delta_write(st, 2, k, v, 0.5f);
        NEAR(st[0], 2.5f, 1e-4f);
    }
    CHECK(wubu_la2_kernel_pick((float[]){ 3, 1, 2 }, 3) == 1, "cheapest kernel");
    CHECK(wubu_la2_precision(0.05f, 0.01f, 0.1f) == 16, "mid drift -> fp16");
    CHECK(wubu_la2_precision(0.2f, 0.01f, 0.1f) == 32, "high drift -> fp32");
    NEAR(wubu_la2_energy(57000, 1.0f, 0.25f), 0.25f, 1e-5f);
    CHECK(wubu_la2_layer_sched(2, 10, 0.3f) == 1, "SSM layer");
    CHECK(wubu_la2_layer_sched(5, 10, 0.3f) == 0, "attention layer");
    {
        float st[3] = { 1, 2, 3 }, buf[3], st2[3] = { 0, 0, 0 };
        wubu_la2_ckpt(st, 3, buf);
        wubu_la2_restore(st2, 3, buf);
        NEAR(st2[2], 3.0f, 1e-6f);
    }
    NEAR(wubu_la2_recall_gap(0.6f, 0.9f), 0.3f, 1e-5f);
    {
        float st[2] = { 1, 1 }, out[2];
        wubu_la2_stream((float[]){ 2, 3 }, st, 2, out);
        NEAR(out[0], 3.0f, 1e-5f);
    }
    {
        float x[4][1] = { {1},{2},{3},{4} }, st = 0;
        CHECK(wubu_la2_chunk_prefill(&x[0][0], 4, 1, 2, &st) == 2, "chunks");
        NEAR(st, 10.0f, 1e-4f);
    }
    NEAR(wubu_la2_forget(1.0f, 1.0f), 0.5f, 1e-5f);
    {
        float st[2] = { 3, 4 };
        wubu_la2_normalize(st, 2);
        NEAR(sqrtf(st[0]*st[0] + st[1]*st[1]), 1.0f, 1e-4f);
    }
    NEAR(wubu_la2_update_energy(64, 0.5f), 32.0f, 1e-4f);
    {
        float st[2] = { 1, 1 };
        wubu_la2_decay(st, 2, 0.5f);
        NEAR(st[0], 0.5f, 1e-5f);
    }
    {
        int32_t out[2];
        float st[2] = { 1, -1 };
        wubu_la2_quant_state(st, 2, 4, out);
        CHECK(out[0] == 7 && out[1] == -7, "quantized state");
    }
    {
        float ratio = 4.0f;
        wubu_la2_expansion(0.5f, 0.8f, &ratio);
        CHECK(ratio > 4.0f, "under-recall expands");
        wubu_la2_expansion(0.9f, 0.8f, &ratio);
        CHECK(ratio < 16.0f, "over-recall shrinks");
    }
    {
        float logits[2];
        CHECK(wubu_la2_draft((float[]){ 1, 2 }, 2, logits) == 2, "drafter");
        NEAR(logits[1], 2.0f, 1e-5f);
    }
    CHECK(wubu_la2_chunk_par(10, 4) == 4, "parallel chunks");
    {
        float s0[2] = { 1, 0 }, s1[2] = { 0, 1 }, out[2];
        const float *states[2] = { s0, s1 };
        wubu_la2_mux(states, 2, 2, (float[]){ 0.5f, 0.5f }, out);
        NEAR(out[0], 0.5f, 1e-5f);
        NEAR(out[1], 0.5f, 1e-5f);
    }
    CHECK(wubu_la2_watchdog(2.0f, 1.5f) == 1, "state-norm watchdog");
    {
        int scheme = -1;
        wubu_la2_pos_head(0, 4, &scheme);
        CHECK(scheme == 0, "RoPE head");
        wubu_la2_pos_head(3, 4, &scheme);
        CHECK(scheme == 1, "PaTH head");
    }
    CHECK(wubu_la2_span(0.9f, 0.5f) > 0, "finite span");
    CHECK(wubu_la2_o1(64, 1000000) == 1, "O(1) state");
    CHECK(wubu_la2_slot_cap(64, 8) == 512, "slot capacity");
    CHECK(wubu_la2_needle((float[]){ 1, 0 }, 2, (float[]){ 1, 0 }, 0.5f) == 1,
          "needle found");
    {
        int keep[3];
        CHECK(wubu_la2_prune((float[]){ 0.9f, 0.1f, 0.5f }, 3, 0.3f, keep) == 2,
              "low-importance pruned");
    }
    NEAR(wubu_la2_layer_cost(1, 0, 2.0f, 0.5f), 2.0f, 1e-5f);

    if (failures == 0) printf("ALL LINATTN2 TESTS PASSED\n");
    else printf("%d LINATTN2 FAILURES\n", failures);
    return failures ? 1 : 0;
}
