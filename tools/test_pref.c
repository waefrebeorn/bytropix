/* test_pref.c -- Theme IQ batch 1: the preference-opt frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_pref.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_pref (IQ batch 1) ===\n");

    /* IQ01: SimPO -- the win should produce a SMALLER loss */
    {
        float l_win = wubu_pref_simpo(-2.0f, -4.0f, 20, 20, 1.0f, 0.5f);
        float l_lose = wubu_pref_simpo(-4.0f, -2.0f, 20, 20, 1.0f, 0.5f);
        CHECK(l_win < l_lose, "win pair losses less");
        CHECK(l_win >= 0 && l_win < 1, "bounded");
    }

    /* IQ03: IPO -- zero at the tau threshold */
    NEAR(wubu_pref_ipo(1.0f, 0.0f, 1.0f, 1.0f), 0.0f, 1e-6f);
    CHECK(wubu_pref_ipo(2.0f, 0.0f, 1.0f, 1.0f) > 0, "above tau -> loss");

    /* IQ08: length normalization */
    {
        float n1 = wubu_pref_len_norm(-20.0f, 20, 1.0f);
        float n2 = wubu_pref_len_norm(-10.0f, 10, 1.0f);
        NEAR(n1, -1.0f, 1e-5f);
        NEAR(n2, -1.0f, 1e-5f);
    }

    /* IQ06: margin sampling prefers near-margin pairs */
    {
        float near_m = wubu_pref_margin_score(1.2f, 0.2f, 1.0f);
        float far = wubu_pref_margin_score(3.0f, -3.0f, 1.0f);
        CHECK(near_m > far, "near-margin pair scores higher");
    }

    /* IQ10: difficulty weight */
    NEAR(wubu_pref_difficulty_weight(0.0f), 1.0f, 1e-5f);
    CHECK(wubu_pref_difficulty_weight(3.0f) < 0.01f, "easy pair down-weighted");

    /* IQ11: reward accuracy */
    {
        float w[3] = { 1, 2, 3 }, l[3] = { 0, 5, 2 };
        NEAR(wubu_pref_accuracy(w, l, 3), 2.0f / 3.0f, 1e-6f);
    }

    /* IQ12: pair dedup */
    {
        float k0[2] = { 1, 1 }, k1[2] = { 0, 0 }, nk[2] = { 1, 1 };
        const float *keys[2] = { k0, k1 };
        CHECK(wubu_pref_dedup(keys, 2, 2, nk, 1e-4f) == 0, "dup found");
    }

    /* IQ13: mixing grows the online fraction */
    {
        float m0 = wubu_pref_mix(0.5f, 0, 10);
        float m1 = wubu_pref_mix(0.5f, 10, 10);
        CHECK(m0 < m1, "online share grows");
        NEAR(m1, 1.0f, 1e-5f);
    }

    /* IQ14: consensus + disagreement */
    {
        float v1[3] = { 0.9f, 0.85f, 0.95f }, v2[3] = { 0.9f, 0.1f, 0.8f };
        float m, s;
        CHECK(wubu_pref_consensus(v1, 3, &m, &s) == 0, "agree -> no flag");
        CHECK(wubu_pref_consensus(v2, 3, &m, &s) == 1, "disagree -> flag");
        NEAR(m, 0.6f, 1e-4f);
    }

    /* IQ15: margin anneal */
    NEAR(wubu_pref_margin_schedule(0.1f, 0.9f, 0.5f), 0.5f, 1e-5f);
    NEAR(wubu_pref_margin_schedule(0.1f, 0.9f, 0.0f), 0.1f, 1e-5f);

    /* IQ16: noise-robust loss is positive + grows with logit */
    CHECK(wubu_pref_noise_loss(2.0f, 0.1f) > 0, "positive");
    CHECK(wubu_pref_noise_loss(2.0f, 0.4f) > wubu_pref_noise_loss(2.0f, 0.1f),
          "noisier -> larger loss");

    /* IQ17: token-level reward */
    {
        float tw[3] = { 0.5f, 0.6f, 0.4f }, tl[3] = { 0.3f, 0.2f, 0.4f };
        NEAR(wubu_pref_token_reward(tw, tl, 3), 0.2f, 1e-5f);
    }

    /* IQ19: the gradient cache */
    {
        wubu_pref_cache_t c = { 0, 0, 0 };
        NEAR(wubu_pref_cache_get(&c, 1.0f, 0.5f), 0.5f, 1e-6f);
        wubu_pref_cache_put(&c, 1.0f, 0.9f);
        NEAR(wubu_pref_cache_get(&c, 1.0f, 0.5f), 0.9f, 1e-6f);
    }

    /* IQ20: early stopping */
    {
        int stale = 0;
        CHECK(wubu_pref_early_stop(0.9f, 0.8f, 3, &stale) == 0 && stale == 0,
              "good accuracy resets");
        wubu_pref_early_stop(0.5f, 0.8f, 3, &stale);
        wubu_pref_early_stop(0.5f, 0.8f, 3, &stale);
        CHECK(wubu_pref_early_stop(0.5f, 0.8f, 3, &stale) == 1, "patience hit");
    }

    /* IQ22: staleness decay */
    NEAR(wubu_pref_staleness(0.0f, 10.0f), 1.0f, 1e-5f);
    NEAR(wubu_pref_staleness(10.0f, 10.0f), 0.5f, 1e-4f);

    if (failures == 0) printf("ALL PREF TESTS PASSED\n");
    else printf("%d PREF FAILURES\n", failures);
    return failures ? 1 : 0;
}
