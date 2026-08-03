/* test_bonzi2.c -- Theme JE complete: the companion frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_bonzi2.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_bonzi2 (JE complete) ===\n");
    {
        float text[2] = { 1, 0 }, voice[2] = { 0, 1 }, s;
        CHECK(wubu_bonzi_sentiment(text, voice, 2, &s) == 0, "sentiment");
        NEAR(s, 0.5f, 1e-5f);
    }
    {
        float anim[3];
        CHECK(wubu_bonzi_react(0.8f, anim) == 0, "react");
        NEAR(anim[0], 0.8f, 1e-5f);
    }
    {
        float params[3];
        CHECK(wubu_bonzi_avatar(params, 3) == 3, "avatar");
        NEAR(params[0], 0.5f, 1e-5f);
    }
    NEAR(wubu_bonzi_presence(0.8f, 0.6f), 0.7f, 1e-5f);
    CHECK(wubu_bonzi_guardrail("hello", "cheerful", 100) == 1, "guardrail ok");
    {
        float req[4] = { 1, 2, 3, 4 };
        int out = 0;
        CHECK(wubu_bonzi_batch(req, 4, 2, &out) == 0 && out == 2, "batch");
    }
    {
        float depth = 0;
        CHECK(wubu_bonzi_empathy(0.8f, 0.5f, &depth) == 0, "empathy");
        NEAR(depth, 0.4f, 1e-5f);
    }
    {
        char vocab[32];
        CHECK(wubu_bonzi_vocab("happy", vocab, 32) == 5, "vocab");
        CHECK(strcmp(vocab, "happy") == 0, "vocab copied");
    }
    {
        float expr[3] = { 1, 2, 3 };
        CHECK(wubu_bonzi_micro(expr, 3, 0.5f) == 0, "micro-expr");
        NEAR(expr[0], 0.5f, 1e-5f);
    }
    {
        float params[3];
        CHECK(wubu_bonzi_voice(params, 3, "cheerful") == 3, "voice");
    }
    {
        int action = 0;
        CHECK(wubu_bonzi_proactive(10.0f, 0.5f, &action) == 0 && action == 1, "proactive");
        CHECK(wubu_bonzi_proactive(1.0f, 0.5f, &action) == 0 && action == 0, "not idle");
    }
    {
        float rgb[3];
        CHECK(wubu_bonzi_lighting(0.8f, rgb) == 0, "lighting");
        NEAR(rgb[0], 0.8f, 1e-5f);
    }
    NEAR(wubu_bonzi_parasocial(0.9f, 0.1f), 0.81f, 1e-5f);
    {
        float mem[3] = { 1, 2, 3 }, anchor;
        CHECK(wubu_bonzi_anchor(mem, 3, &anchor) == 0, "anchor");
        NEAR(anchor, 2.0f, 1e-5f);
    }
    {
        char script[64];
        CHECK(wubu_bonzi_repair(0.5f, script, 64) >= 0, "repair");
    }
    {
        int action = 0;
        CHECK(wubu_bonzi_mood_action(0.8f, &action) == 0 && action == 1, "mood action");
    }
    CHECK(wubu_bonzi_topic_mem("news", (float[]){1}, 1) == 0, "topic mem");
    {
        float text[3] = { 1, 2, 3 }, emotion[1];
        CHECK(wubu_bonzi_speech_emotion(text, 3, emotion) == 0, "speech emotion");
    }
    {
        float ap[2];
        CHECK(wubu_bonzi_couple(0.7f, ap) == 0, "coupling");
        NEAR(ap[0], 0.7f, 1e-5f);
    }
    {
        float budget = 0;
        CHECK(wubu_bonzi_idle_budget(30.0f, 100.0f, &budget) == 0, "idle budget");
        NEAR(budget, 50.0f, 1e-4f);
    }

    if (failures == 0) printf("ALL BONZI2 TESTS PASSED\n");
    else printf("%d BONZI2 FAILURES\n", failures);
    return failures ? 1 : 0;
}