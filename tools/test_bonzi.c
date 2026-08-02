/* test_bonzi.c -- Theme JE: Bonzi Buddy companion core (batch 1). */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_bonzi.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_bonzi (JE batch 1) ===\n");

    /* JE01/JE13: mood state machine */
    {
        wubu_bonzi_mood_t m = { 0.0f, 0.3f, 0 };
        wubu_bonzi_mood_step(&m, 1.0f, 0.9f, 0.5f);
        NEAR(m.valence, 0.5f, 1e-5f);
        NEAR(m.arousal, 0.6f, 1e-5f);
        CHECK(m.seq == 1, "seq advanced");
        /* clamps */
        wubu_bonzi_mood_t m2 = { -0.9f, 0.95f, 0 };
        wubu_bonzi_mood_step(&m2, 1.0f, 1.0f, 2.0f);
        CHECK(m2.valence <= 1.0f && m2.arousal <= 1.0f, "clamped");
        CHECK(wubu_bonzi_mood_step(NULL, 0, 0, 0.5f) == -1, "null rejected");
    }

    /* JE02: persona fit */
    {
        NEAR(wubu_bonzi_persona_score("warm", "warm", 4), 1.0f, 1e-5f);
        NEAR(wubu_bonzi_persona_score("warm", "ward", 4), 0.75f, 1e-5f);
        NEAR(wubu_bonzi_persona_score("warm", "cold", 4), 0.0f, 1e-5f);
    }

    /* JE03/JE17: idle scheduling + affordability */
    {
        CHECK(wubu_bonzi_idle(20, 10, 1.0f, 0.5f) == 1, "idle plays");
        CHECK(wubu_bonzi_idle(5, 10, 1.0f, 0.5f) == 0, "too soon");
        CHECK(wubu_bonzi_idle(20, 10, 0.2f, 0.5f) == 0, "no energy (ties IJ)");
        CHECK(wubu_bonzi_idle_afford(1.0f, 0.5f) == 1, "afford");
        CHECK(wubu_bonzi_idle_afford(0.2f, 0.5f) == 0, "can't afford");
    }

    /* JE04: prosody */
    {
        wubu_bonzi_mood_t happy = { 1.0f, 1.0f, 0 };
        wubu_bonzi_mood_t sad = { -1.0f, 0.1f, 0 };
        float p1, s1, e1, p2, s2, e2;
        wubu_bonzi_prosody(&happy, &p1, &s1, &e1);
        wubu_bonzi_prosody(&sad, &p2, &s2, &e2);
        CHECK(p1 > p2 && s1 > s2 && e1 > e2, "happy is brighter + faster");
    }

    /* JE06: empathy */
    {
        float w_sad = wubu_bonzi_empathy(-1.0f, 1.0f);
        float w_happy = wubu_bonzi_empathy(1.0f, 1.0f);
        CHECK(w_sad > w_happy, "sadder user gets more warmth");
        NEAR(w_sad, 1.0f, 1e-5f);
        NEAR(w_happy, 0.5f, 1e-5f);
    }

    /* JE07/JE22: turn delay */
    {
        int d1 = wubu_bonzi_turn_delay(1.0f, 10);
        int d2 = wubu_bonzi_turn_delay(0.0f, 10);
        CHECK(d1 < d2, "aroused -> faster response");
        CHECK(d1 >= 3, "bounded latency floor");
    }

    /* JE08: mood log */
    {
        wubu_bonzi_mood_t log[4], s = { 0.5f, 0.5f, 1 };
        int n = 0;
        wubu_bonzi_mood_log(log, &n, 4, &s);
        wubu_bonzi_mood_log(log, &n, 4, &s);
        CHECK(n == 2, "two samples");
        wubu_bonzi_mood_log(log, &n, 4, &s);
        wubu_bonzi_mood_log(log, &n, 4, &s);
        CHECK(n == 4, "capped");
    }

    /* JE12: notify tone */
    {
        float t_hi = wubu_bonzi_notify_tone(1.0f, 1.0f);
        float t_lo = wubu_bonzi_notify_tone(0.1f, -1.0f);
        CHECK(t_hi > t_lo, "urgent notifies louder");
    }

    /* JE14/JE15 */
    {
        float slots[3] = { 0 }, imp[3] = { 0 };
        CHECK(wubu_bonzi_mem_write(slots, imp, 3, 1, 0.7f, 0.9f) == 0, "mem write");
        NEAR(slots[1], 0.7f, 1e-5f);
        NEAR(imp[1], 0.9f, 1e-5f);
        CHECK(wubu_bonzi_mem_write(slots, imp, 3, 5, 1, 1) == -1, "OOB slot");
        CHECK(wubu_bonzi_chat_prune(10, 5) == 5, "chat prune");
        CHECK(wubu_bonzi_chat_prune(3, 10) == 0, "under budget");
    }

    /* JE20/JE21 */
    {
        NEAR(wubu_bonzi_user_mood(0.0f, 1.0f, 0.5f), 0.5f, 1e-5f);
        NEAR(wubu_bonzi_user_mood(0.5f, 0.5f, 0.5f), 0.5f, 1e-5f);
        wubu_bonzi_mood_t m = { 0.5f, 0.8f, 0 };
        NEAR(wubu_bonzi_self_mood(&m, 1.0f), 0.8f, 1e-5f);
        NEAR(wubu_bonzi_self_mood(&m, 0.5f), 0.4f, 1e-5f);
    }

    /* JE26: session continuity */
    {
        uint32_t a = wubu_bonzi_session_id("wubu", 42);
        uint32_t b = wubu_bonzi_session_id("wubu", 42);
        uint32_t c = wubu_bonzi_session_id("other", 42);
        CHECK(a == b, "same user -> same session");
        CHECK(a != c, "different user -> different session");
        CHECK(a != 0, "non-zero id");
    }

    /* JE32/JE37/JE42/JE67 */
    {
        float win[3] = { 0 };
        /* rolling mean over the whole window: first feed = 1/3 */
        NEAR(wubu_bonzi_engagement(win, 3, 1.0f), 1.0f / 3.0f, 1e-5f);
        /* window now {0, 1.0, 0.5} -> mean 0.5 */
        NEAR(wubu_bonzi_engagement(win, 3, 0.5f), 0.5f, 1e-5f);
        NEAR(wubu_bonzi_honesty(1.0f, 1.0f), 1.0f, 1e-5f);
        NEAR(wubu_bonzi_honesty(1.0f, 0.0f), 0.5f, 1e-5f);
        wubu_bonzi_mood_t p = { 0.0f, 0.3f, 0 }, c = { 0.9f, 0.9f, 1 };
        CHECK(wubu_bonzi_mood_anomaly(&p, &c, 0.5f) == 1, "mood jump flagged");
        CHECK(wubu_bonzi_mood_anomaly(&p, &c, 2.0f) == 0, "within bound");
        NEAR(wubu_bonzi_mem_decay(1.0f, 10, 10), 0.5f, 1e-4f);
        NEAR(wubu_bonzi_mem_decay(1.0f, 0, 10), 1.0f, 1e-6f);
    }

    if (failures == 0) printf("ALL BONZI TESTS PASSED\n");
    else printf("%d BONZI FAILURES\n", failures);
    return failures ? 1 : 0;
}
