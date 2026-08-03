/* test_hopfield4.c -- Theme IP ABSOLUTE final: completes the 26 remaining gaps. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_hopfield3.h"  /* the impl is in wubu_hopfield3.c — wubu_hopfield4.h re-exports */
#include "wubu_hopfield4.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_hopfield4 (IP ABSOLUTE final) ===\n");

    /* IP05: attention-as-Hopfield retrieval */
    {
        float q[4] = { 1, 0, 0, 0 }, kv[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
        float out[4];
        int best = wubu_hop3_attention_read(q, kv, 4, 4, out);
        CHECK(best == 0, "attention read picks first pattern");
    }

    /* IP22: manifold curvature estimation */
    {
        float patterns[8] = { 1,0, 0,1, 1,0, 0,1 };
        float c = wubu_hop3_curvature(patterns, 4, 2);
        CHECK(c > 0, "curvature is positive");
    }

    /* IP23: federated memory sharing */
    {
        float patterns[4] = { 1, 2, 3, 4 };
        int merged = 0;
        CHECK(wubu_hop3_federated(patterns, 4, 42, &merged) == 0 && merged == 42004, "federated merge");
    }

    /* IP32: memory stabilization */
    {
        float pattern[4] = { 1, 1, 1, 1 }, anchored[4];
        CHECK(wubu_hop3_stabilize(pattern, 4, 1.0f, anchored) == 0, "stabilize");
        float norm = sqrtf(anchored[0]*anchored[0] + anchored[1]*anchored[1] + anchored[2]*anchored[2] + anchored[3]*anchored[3]);
        NEAR(norm, 1.0f, 1e-4f);
    }

    /* IP41: cue embedding quality monitor */
    {
        float cue[4] = { 0.3f, 0.4f, 0.5f, 0.6f };
        CHECK(wubu_hop3_cue_quality(cue, 4, 0.5f) == 1, "cue quality OK");
        CHECK(wubu_hop3_cue_quality(cue, 4, 1.5f) == 0, "cue quality bad");
    }

    /* IP42: memory write batching */
    {
        float patterns[16] = { 0 };
        CHECK(wubu_hop3_write_batch(patterns, 4, 4, 2) == 2, "write batches");
    }

    /* IP43: memory read batching */
    {
        float patterns[16] = { 0 };
        CHECK(wubu_hop3_read_batch(patterns, 4, 4, 2) == 2, "read batches");
    }

    /* IP46: associative outlier tolerance */
    {
        float pattern[4] = { 0.5f, 0.5f, 0.5f, 0.5f };
        CHECK(wubu_hop3_outlier_tol(pattern, 4, 1.0f) == 1, "outlier tolerated");
    }

    /* IP49: memory ANN search */
    {
        float query[4] = { 1, 0, 0, 0 }, memory[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
        int idx = -1;
        CHECK(wubu_hop3_ann(query, memory, 4, 4, 0.5f, &idx) == 1, "ANN found");
        CHECK(idx == 0, "ANN best match");
    }

    /* IP50: write/read asymmetry */
    NEAR(wubu_hop3_asymmetry(100, 200), 0.5f, 1e-5f);

    /* IP52: decay vs consolidation arbitration */
    CHECK(wubu_hop3_decay_arbitrate(0.1f, 0.05f, 0.02f) == 1, "decay wins");
    CHECK(wubu_hop3_decay_arbitrate(0.03f, 0.05f, 0.02f) == 0, "consolidation wins");

    /* IP53: retrieval-augmented memory */
    {
        float corpus[8] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f };
        float pattern[2];
        CHECK(wubu_hop3_rag(corpus, 4, 2, pattern) == 2, "RAG pattern");
        NEAR(pattern[0], 0.4f, 1e-4f);
    }

    /* IP54: memory provenance */
    {
        float pattern[4] = { 0 };
        char meta[32];
        CHECK(wubu_hop3_provenance(pattern, 12345, meta, 32) == 0, "provenance");
        CHECK(strstr(meta, "12345") != NULL, "id in metadata");
    }

    /* IP55: memory privacy (forget-set) */
    {
        float pattern[4] = { 1, 2, 3, 4 };
        int forget_ids[2] = { 0, 2 };
        CHECK(wubu_hop3_forget(pattern, forget_ids, 2, 2) == 1, "forgotten");
        CHECK(wubu_hop3_forget(pattern, forget_ids, 2, 1) == 0, "not forgotten");
    }

    /* IP56: load balancing across tiers */
    {
        float access[4] = { 10, 100, 5, 50 };
        int hot = -1;
        CHECK(wubu_hop3_balance(access, 4, &hot) == 0 && hot == 1, "hot tier");
    }

    /* IP57: world-model updates via associative memory */
    {
        float state[4] = { 1, 0, 0, 0 }, obs[4] = { 0, 1, 0, 0 }, next[4];
        CHECK(wubu_hop3_world_update(state, 4, obs, next) == 0, "world update");
        NEAR(next[0], 1.0f, 1e-5f);
        NEAR(next[1], 0.1f, 1e-5f);
    }

    /* IP58: capacity warning */
    CHECK(wubu_hop3_capacity_warning(2000, 1000) == 1, "capacity warning");
    CHECK(wubu_hop3_capacity_warning(500, 1000) == 0, "within capacity");

    /* IP59: pattern importance weighting */
    {
        float patterns[8] = { 1,0, 0,1, 1,1, 0,0 };
        float weights[4];
        wubu_hop3_weight(patterns, 4, 2, weights);
        CHECK(weights[0] == 1.0f && weights[2] == sqrtf(2.0f), "pattern weights");
    }

    /* IP60: session coherence */
    {
        float a[4] = { 1, 0, 1, 0 }, b[4] = { 1, 0, 1, 0.01f };
        float score;
        CHECK(wubu_hop3_coherence(a, b, 2, 2, &score) == 0, "coherence");
        CHECK(score > 0.9f, "high coherence");
    }

    /* IP61: momentum Hopfield update */
    {
        float current[4] = { 1, 1, 0, 0 }, target[4] = { 0, 0, 1, 1 }, next[4];
        CHECK(wubu_hop3_momentum(current, target, 4, 0.5f, next) == 0, "momentum");
        NEAR(next[0], 0.5f, 1e-5f);
        NEAR(next[2], 0.5f, 1e-5f);
    }

    /* IP62: sparse Hopfield */
    {
        float patterns[16] = { 0 };
        int selected[4];
        CHECK(wubu_hop3_sparse(patterns, 16, 4, selected) == 4, "sparse select");
    }

    /* IP63: continuous-time Hopfield */
    NEAR(wubu_hop3_continuous(10.0f, 1.0f, 0.0f), 0.1f, 1e-4f);

    /* IP64: energy function */
    {
        float state[2] = { 1, 0 }, weights[4] = { 0, 1, 1, 0 };
        float e = wubu_hop3_energy(state, weights, 2);
        CHECK(e < 2.0f, "energy computed");
    }

    /* IP65: capacity scaling */
    CHECK(wubu_hop3_scaling(100, 0.5f) == 50, "scaling");

    /* IP66: noise robustness */
    {
        float clean[4] = { 1, 1, 1, 1 }, noisy[4] = { 1.01f, 0.99f, 1.0f, 1.0f };
        CHECK(wubu_hop3_noise(clean, noisy, 4, 0.02f) == 1, "noise tolerated");
    }

    /* IP67: pattern completion */
    {
        float partial[2] = { 1, 0 }, memory[8] = { 1, 0, 0, 1, 1, 1, 0, 0 };
        float completed[2];
        int best = wubu_hop3_complete(partial, 2, memory, 4, completed);
        CHECK(best >= 0, "pattern completed");
    }

    if (failures == 0) printf("ALL HOPFIELD4 TESTS PASSED\n");
    else printf("%d HOPFIELD4 FAILURES\n", failures);
    return failures ? 1 : 0;
}
