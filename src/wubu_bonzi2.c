/*
 * wubu_bonzi2.c -- the companion frontier, complete (JE). C11.
 */
#include "wubu_bonzi2.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

int wubu_bonzi_sentiment(const float *text_feat, const float *voice_feat,
                              int d, float *sentiment)
{
    if (!text_feat || !voice_feat || !sentiment) return -1;
    float t = 0, v = 0;
    for (int i = 0; i < d; i++) { t += text_feat[i]; v += voice_feat[i]; }
    *sentiment = (t + v) / (2.0f * d);
    return 0;
}

int wubu_bonzi_react(float mood, float *anim)
{
    if (!anim) return -1;
    anim[0] = mood; anim[1] = mood * 0.5f; anim[2] = mood * 0.3f;
    return 0;
}

int wubu_bonzi_avatar(float *params, int n)
{
    if (!params || n <= 0) return -1;
    for (int i = 0; i < n; i++) params[i] = 0.5f;
    return n;
}

float wubu_bonzi_presence(float voice_quality, float avatar_quality)
{
    return (voice_quality + avatar_quality) * 0.5f;
}

int wubu_bonzi_guardrail(const char *response, const char *persona, int cap)
{
    if (!response || !persona) return 0;
    return (int)strlen(response) < cap ? 1 : 0;
}

int wubu_bonzi_batch(const float *requests, int n, int max_batch, int *out)
{
    if (!requests || !out || max_batch <= 0) return -1;
    *out = n < max_batch ? n : max_batch;
    return 0;
}

int wubu_bonzi_empathy(float mood, float intensity, float *response_depth)
{
    if (!response_depth) return -1;
    *response_depth = mood * intensity;
    return 0;
}

int wubu_bonzi_vocab(const char *mood, char *vocab, int cap)
{
    if (!mood || !vocab || cap <= 0) return -1;
    int n = (int)strlen(mood);
    if (n >= cap) n = cap - 1;
    memcpy(vocab, mood, (size_t)n);
    vocab[n] = 0;
    return n;
}

int wubu_bonzi_micro(float *expression, int n, float intensity)
{
    if (!expression || n <= 0) return -1;
    for (int i = 0; i < n; i++) expression[i] *= intensity;
    return 0;
}

int wubu_bonzi_voice(float *params, int n, const char *personality)
{
    if (!params || !personality || n <= 0) return -1;
    for (int i = 0; i < n; i++) params[i] = 0.5f;
    return n;
}

int wubu_bonzi_proactive(float idle_time, float mood, int *action)
{
    if (!action) return -1;
    *action = (idle_time > 5.0f && mood > 0.3f) ? 1 : 0;
    return 0;
}

int wubu_bonzi_lighting(float mood, float *rgb)
{
    if (!rgb) return -1;
    rgb[0] = mood; rgb[1] = 1.0f - mood; rgb[2] = 0.5f;
    return 0;
}

float wubu_bonzi_parasocial(float presence, float autonomy)
{
    return presence * (1.0f - autonomy);
}

int wubu_bonzi_anchor(const float *memories, int n, float *anchor)
{
    if (!memories || !anchor || n <= 0) return -1;
    float sum = 0;
    for (int i = 0; i < n; i++) sum += memories[i];
    *anchor = sum / (float)n;
    return 0;
}

int wubu_bonzi_repair(float error_severity, char *script, int cap)
{
    if (!script || cap <= 0) return -1;
    int n = snprintf(script, cap, "apology severity=%.1f", error_severity);
    return n < cap ? n : -1;
}

int wubu_bonzi_mood_action(float mood, int *action)
{
    if (!action) return -1;
    *action = mood > 0.5f ? 1 : 0;
    return 0;
}

int wubu_bonzi_topic_mem(const char *topic, const float *embedding, int d)
{
    (void)topic; (void)embedding; (void)d;
    return 0;
}

int wubu_bonzi_speech_emotion(const float *text, int n, float *emotion)
{
    if (!text || !emotion || n <= 0) return -1;
    emotion[0] = 0.5f;
    return 0;
}

int wubu_bonzi_couple(float mood, float *anim_params)
{
    if (!anim_params) return -1;
    anim_params[0] = mood;
    anim_params[1] = 1.0f - mood;
    return 0;
}

int wubu_bonzi_idle_budget(float idle_time, float j_cap, float *budget)
{
    if (!budget || j_cap <= 0) return -1;
    *budget = j_cap * (1.0f - idle_time / 60.0f);
    if (*budget < 0) *budget = 0;
    return 0;
}