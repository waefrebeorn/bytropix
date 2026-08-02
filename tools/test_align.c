/* test_align.c -- Theme IM: preference alignment + unlearning. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_align.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_align (IM01-IM07) ===\n");

    /* IM01: DPO -- the win must dominate; a win with a HIGHER model
     * prob than the reference gives a POSITIVE reward */
    {
        float r_ok = wubu_dpo_reward(-1.0f, -2.0f, -3.0f, -2.0f, 1.0f);
        /* r = 1*( ( -1 - -2 ) - ( -3 - -2 ) ) = 1*( 1 - -1 ) = 2 */
        NEAR(r_ok, 2.0f, 1e-4f);
        /* a win with a LOWER model prob than the reference -> negative */
        float r_bad = wubu_dpo_reward(-3.0f, -2.0f, -1.0f, -2.0f, 1.0f);
        NEAR(r_bad, -2.0f, 1e-4f);
        /* the loss is monotone decreasing in the reward */
        float l_ok = wubu_dpo_loss(-1.0f, -2.0f, -3.0f, -2.0f, 1.0f);
        float l_bad = wubu_dpo_loss(-3.0f, -2.0f, -1.0f, -2.0f, 1.0f);
        CHECK(l_ok < l_bad, "DPO loss lower for the positive reward");
        NEAR(wubu_dpo_loss(0, 0, 0, 0, 1.0f), 0.6931472f, 1e-4f); /* log 2 */
        NEAR(wubu_dpo_reward(0, 0, 0, 0, 0), 0.0f, 1e-6f);
    }

    /* IM02: KTO -- the desirable loss falls as the reward rises; the
     * undesirable rises */
    {
        float d_hi = wubu_kto_loss(1, 2.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        float d_lo = wubu_kto_loss(1, -2.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        CHECK(d_hi < d_lo, "desirable with high reward: low loss");
        float u_hi = wubu_kto_loss(0, 2.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        float u_lo = wubu_kto_loss(0, -2.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        CHECK(u_hi > u_lo, "undesirable with high reward: high loss");
        NEAR(wubu_kto_loss(1, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f), 0.5f, 1e-4f);
        NEAR(wubu_kto_loss(0, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f), 0.5f, 1e-4f);
    }

    /* IM03/IM04: unlearning updates */
    {
        NEAR(wubu_unlearn_ascent(0.1f, 2.0f), 0.2f, 1e-6f);
        NEAR(wubu_unlearn_ascent(0.0f, 2.0f), 0.0f, 1e-6f);
        NEAR(wubu_unlearn_anchor_weight(0.5f, 4.0f), 2.0f, 1e-6f);
        NEAR(wubu_unlearn_anchor_weight(-1.0f, 4.0f), 0.0f, 1e-6f);
        NEAR(wubu_unlearn_anchor_weight(0.5f, -1.0f), 0.0f, 1e-6f);
    }

    /* IM05: alignment replay */
    {
        wubu_align_buffer_t b;
        memset(&b, 0, sizeof(b));
        CHECK(wubu_align_push(&b, 0.1f) == 0, "push 1");
        CHECK(wubu_align_push(&b, 0.9f) == 0, "push 2");
        CHECK(wubu_align_push(&b, 0.5f) == 0, "push 3");
        CHECK(b.count == 3 && b.min_pref == 0.1f, "count + min");
        NEAR(wubu_align_mean(&b), 0.5f, 1e-5f);
        int idx[3];
        int n = wubu_align_topk(&b, 2, idx);
        CHECK(n == 2 && b.pref[idx[0]] == 0.9f && b.pref[idx[1]] == 0.5f,
              "topk by preference");
        /* ring: 300 pushes keep the count capped */
        wubu_align_buffer_t r;
        memset(&r, 0, sizeof(r));
        for (int i = 0; i < 300; i++) wubu_align_push(&r, (float)(i % 10) / 10.0f);
        CHECK(r.count == WUBU_ALIGN_BUFSZ, "ring capped");
        CHECK(wubu_align_topk(&r, 3, idx) == 3, "ring topk works");
    }

    /* IM06: drift monitor -- reward hacking detected */
    {
        wubu_align_monitor_t m;
        CHECK(wubu_align_monitor_init(&m, 3.0f) == 0, "monitor init");
        /* warm-up window (the baseline) */
        for (int i = 0; i < 50; i++) wubu_align_monitor_feed(&m, 0.5f);
        CHECK(wubu_align_monitor_drifted(&m) == 0, "stable window: no drift");
        /* seed the baseline from the warm window */
        m.baseline_mean = 0.5f;
        m.baseline_std = 0.05f;
        /* a spike: rewards jump to 2.0 (mean moves >> 3 sigma) */
        for (int i = 0; i < 50; i++) wubu_align_monitor_feed(&m, 2.0f);
        CHECK(wubu_align_monitor_drifted(&m) == 1, "reward spike flagged");
        CHECK(wubu_align_monitor_init(&m, 0) == -1, "bad sigma rejected");
    }

    /* IM07: the operator config pick */
    {
        const float al[4] = { 0.5f, 0.8f, 0.7f, 0.9f };
        const float co[4] = { 1.0f, 2.0f, 0.5f, 3.0f };
        /* max cost 1.0: configs 0 (0.5) and 2 (0.7) -> pick 2 */
        CHECK(wubu_align_pick_config(al, co, 4, 1.0f) == 2, "best under cost");
        /* max cost 3.0 -> config 3 (0.9) */
        CHECK(wubu_align_pick_config(al, co, 4, 3.0f) == 3, "best under 3.0");
        /* max cost 0.4 -> none */
        CHECK(wubu_align_pick_config(al, co, 4, 0.4f) == -1, "none affordable");
        CHECK(wubu_align_pick_config(al, co, 4, -1.0f) == -1, "neg cost");
    }

    if (failures == 0) printf("ALL ALIGN TESTS PASSED\n");
    else printf("%d ALIGN FAILURES\n", failures);
    return failures ? 1 : 0;
}
