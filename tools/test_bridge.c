/* test_bridge.c -- Theme JF: cross-resource bridges (first batch). */
#include <stdio.h>
#include <math.h>
#include "wubu_bridge.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_bridge (JF first batch) ===\n");

    /* JF01: mood retrieval */
    {
        float moods[3] = { -1.0f, 0.0f, 1.0f };  /* sad, neutral, happy */
        float out = 0;
        CHECK(wubu_br_mood_retrieve(moods, 3, 0.8f, &out) == 2, "happy mood");
        NEAR(out, 1.0f, 1e-5f);
        CHECK(wubu_br_mood_retrieve(moods, 3, -0.9f, &out) == 0, "sad mood");
    }

    /* JF02: confidence gate */
    {
        CHECK(wubu_br_confidence_gate(0.9f, 0.8f, 5) == 1, "confident + budget");
        CHECK(wubu_br_confidence_gate(0.7f, 0.8f, 5) == 0, "below thresh");
        CHECK(wubu_br_confidence_gate(0.9f, 0.8f, 0) == 0, "no budget");
    }

    /* JF04: persona guard */
    {
        CHECK(wubu_br_persona_guard(0.95f, 0.8f) == 1, "safe");
        CHECK(wubu_br_persona_guard(0.5f, 0.8f) == 0, "unsafe");
    }

    /* JF06: calibrated credit */
    {
        NEAR(wubu_br_credit(1.0f, 1.0f), 1.0f, 1e-6f);
        NEAR(wubu_br_credit(0.5f, 1.0f), 0.5f, 1e-6f);
        NEAR(wubu_br_credit(0.5f, 0.0f), 0.0f, 1e-6f);
    }

    /* JF07: chat pruning */
    {
        CHECK(wubu_br_chat_prune(10, 5) == 5, "evict 5 of 10");
        CHECK(wubu_br_chat_prune(3, 10) == 0, "under budget -> no evict");
        CHECK(wubu_br_chat_prune(10, 0) == 10, "zero budget -> all evicted");
    }

    /* JF10: regulation */
    {
        CHECK(wubu_br_regulate(0.1f, 5, 0.3f, 0.7f) == 0, "low conf -> stop");
        CHECK(wubu_br_regulate(0.5f, 5, 0.3f, 0.7f) == 1, "mid conf -> retry");
        CHECK(wubu_br_regulate(0.9f, 5, 0.3f, 0.7f) == 2, "high conf -> delegate");
        CHECK(wubu_br_regulate(0.9f, 0, 0.3f, 0.7f) == 0, "no budget -> stop");
    }

    /* JF15: mood prediction */
    {
        NEAR(wubu_br_mood_predict(1.0f, 0, 10.0f), 1.0f, 1e-5f);
        NEAR(wubu_br_mood_predict(1.0f, 10, 10.0f), expf(-1.0f), 1e-4f);
    }

    /* JF18/JF23: forgetting */
    {
        NEAR(wubu_br_forget_retain(0, 10), 1.0f, 1e-6f);
        NEAR(wubu_br_forget_retain(10, 10), 0.5f, 1e-4f);
        NEAR(wubu_br_memory_weight(2.0f, 10, 10), 1.0f, 1e-4f);
    }

    /* JF27: self-pattern */
    {
        float s = 0;
        CHECK(wubu_br_self_pattern(0.7f, &s) == 0 && s == 0.7f, "self slot");
        CHECK(wubu_br_self_pattern(-1.0f, &s) == 0 && s == 0.0f, "clamped low");
        CHECK(wubu_br_self_pattern(0.7f, NULL) == -1, "null slot");
    }

    /* JF31/JF36/JF45 */
    {
        CHECK(wubu_br_verify_output(0.9f, 0.8f) == 1, "verified");
        CHECK(wubu_br_verify_output(0.5f, 0.8f) == 0, "rejected");
        int log_len = 3;
        CHECK(wubu_br_monitor_log(&log_len, 10) == 4, "log appended");
        CHECK(wubu_br_monitor_log(&log_len, 4) == 4, "log capped");
        CHECK(wubu_br_mood_anomaly(0.9f, 0.5f) == 1, "anomaly");
        CHECK(wubu_br_mood_anomaly(0.2f, 0.5f) == 0, "calm");
    }

    /* JF59/JF74/JF85/JF100 */
    {
        NEAR(wubu_br_empathy_reward(0.8f, 0.3f, 2.0f), 1.0f, 1e-5f);
        CHECK(wubu_br_tier(0.2f, 0.4f, 0.8f) == 0, "small tier");
        CHECK(wubu_br_tier(0.6f, 0.4f, 0.8f) == 1, "mid tier");
        CHECK(wubu_br_tier(0.9f, 0.4f, 0.8f) == 2, "large tier");
        float comp[3];
        CHECK(wubu_br_monitor_component(comp, 3, (float[]){ 1, 2, 3 }) == 3, "component");
        NEAR(wubu_br_close_rate(0.0f, 3, 10), 0.3f, 1e-5f);
        NEAR(wubu_br_close_rate(0.5f, 3, 10), 0.44f, 1e-4f);
    }

    if (failures == 0) printf("ALL BRIDGE TESTS PASSED\n");
    else printf("%d BRIDGE FAILURES\n", failures);
    return failures ? 1 : 0;
}
