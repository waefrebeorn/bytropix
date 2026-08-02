/*
 * wubu_bonzi.c -- Bonzi Buddy companion core (Theme JE). C11.
 */
#include "wubu_bonzi.h"
#include <math.h>
#include <string.h>

int wubu_bonzi_mood_step(wubu_bonzi_mood_t *m, float t_val, float t_ar,
                         float step)
{
    if (!m) return -1;
    if (t_val < -1) t_val = -1;
    if (t_val > 1) t_val = 1;
    if (t_ar < 0) t_ar = 0;
    if (t_ar > 1) t_ar = 1;
    if (step < 0) step = 0;
    if (step > 1) step = 1;
    m->valence += (t_val - m->valence) * step;
    m->arousal += (t_ar - m->arousal) * step;
    if (m->valence < -1) m->valence = -1;
    if (m->valence > 1) m->valence = 1;
    if (m->arousal < 0) m->arousal = 0;
    if (m->arousal > 1) m->arousal = 1;
    m->seq++;
    return 0;
}

float wubu_bonzi_persona_score(const char *resp_tags, const char *persona_tags,
                               int n_tags)
{
    if (!resp_tags || !persona_tags || n_tags <= 0) return 0;
    /* count tag matches at the fixed stride (n_tags slots each) */
    int match = 0;
    for (int i = 0; i < n_tags; i++) {
        if (resp_tags[i] && resp_tags[i] == persona_tags[i]) match++;
    }
    return (float)match / (float)n_tags;
}

int wubu_bonzi_idle(int idle_ticks, int anim_interval, float energy_left,
                    float energy_cost)
{
    if (anim_interval <= 0) return 0;
    if (idle_ticks < anim_interval) return 0;
    if (energy_cost > 0 && energy_left < energy_cost) return 0;  /* ties IJ */
    return 1;
}

void wubu_bonzi_prosody(const wubu_bonzi_mood_t *m, float *pitch,
                        float *speed, float *energy)
{
    if (!m) return;
    /* happy + aroused -> higher pitch, faster, more energy */
    if (pitch) *pitch = 1.0f + 0.3f * m->valence + 0.2f * m->arousal;
    if (speed) *speed = 1.0f + 0.25f * m->valence + 0.3f * m->arousal;
    if (energy) *energy = 0.5f + 0.5f * m->arousal;
}

float wubu_bonzi_empathy(float user_valence, float warmth)
{
    if (warmth < 0) warmth = 0;
    /* sadder user -> more warmth (inverse of the valence) */
    float need = 1.0f - (user_valence + 1.0f) * 0.5f;  /* 0..1 */
    return warmth * (0.5f + 0.5f * need);
}

int wubu_bonzi_turn_delay(float arousal, int msg_len)
{
    if (msg_len < 0) msg_len = 0;
    if (arousal < 0) arousal = 0;
    if (arousal > 1) arousal = 1;
    /* base latency shrinks with arousal, grows slightly with length */
    int base = 12 - (int)(6 * arousal);
    if (base < 3) base = 3;
    return base + msg_len / 32;
}

int wubu_bonzi_mood_log(wubu_bonzi_mood_t *log, int *n, int max,
                        const wubu_bonzi_mood_t *sample)
{
    if (!log || !n || !sample || max <= 0) return -1;
    if (*n < max) {
        log[*n] = *sample;
        (*n)++;
    }
    return *n;
}

float wubu_bonzi_notify_tone(float urgency, float mood_valence)
{
    if (urgency < 0) urgency = 0;
    if (urgency > 1) urgency = 1;
    /* urgency raises the alertness; the mood biases the warmth */
    return urgency * 0.7f + (mood_valence + 1.0f) * 0.15f;
}

int wubu_bonzi_mem_write(float *slots, float *importance, int n_slots,
                         int slot, float value, float importance_value)
{
    if (!slots || n_slots <= 0 || slot < 0 || slot >= n_slots) return -1;
    slots[slot] = value;
    if (importance) importance[slot] = importance_value;
    return 0;
}

int wubu_bonzi_chat_prune(int turns, int budget)
{
    if (turns <= 0) return 0;
    if (budget <= 0) return turns;
    int keep = budget < turns ? budget : turns;
    return turns - keep;
}

int wubu_bonzi_idle_afford(float energy_left, float idle_cost)
{
    if (idle_cost <= 0) return 1;
    return energy_left >= idle_cost ? 1 : 0;
}

float wubu_bonzi_user_mood(float prev, float sample, float alpha)
{
    if (alpha < 0) alpha = 0;
    if (alpha > 1) alpha = 1;
    return prev * (1.0f - alpha) + sample * alpha;
}

float wubu_bonzi_self_mood(const wubu_bonzi_mood_t *m, float clarity)
{
    if (!m) return 0;
    if (clarity < 0) clarity = 0;
    if (clarity > 1) clarity = 1;
    /* Bonzi's confidence about its own mood = arousal x clarity */
    return m->arousal * clarity;
}

uint32_t wubu_bonzi_session_id(const char *user, uint32_t seed)
{
    uint32_t h = 2166136261u ^ seed;
    if (user)
        for (const char *p = user; *p; p++) {
            h ^= (uint8_t)*p;
            h *= 16777619u;
        }
    return h;
}

float wubu_bonzi_engagement(float *window, int n, float new_score)
{
    if (!window || n <= 0) return new_score;
    /* shift the window (drop the oldest) + append */
    for (int i = 0; i < n - 1; i++) window[i] = window[i + 1];
    window[n - 1] = new_score;
    float s = 0;
    for (int i = 0; i < n; i++) s += window[i];
    return s / (float)n;
}

float wubu_bonzi_honesty(float confidence, float calibration)
{
    if (confidence < 0) confidence = 0;
    if (confidence > 1) confidence = 1;
    if (calibration < 0) calibration = 0;
    if (calibration > 1) calibration = 1;
    return confidence * (0.5f + 0.5f * calibration);
}

int wubu_bonzi_mood_anomaly(const wubu_bonzi_mood_t *prev,
                            const wubu_bonzi_mood_t *cur, float max_delta)
{
    if (!prev || !cur) return 0;
    float d = fabsf(cur->valence - prev->valence) +
              fabsf(cur->arousal - prev->arousal);
    return d > max_delta ? 1 : 0;
}

float wubu_bonzi_mem_decay(float base, int age, float halflife)
{
    if (age <= 0) return base;
    if (halflife <= 0) return 0;
    return base * expf(-((float)age / halflife) * 0.6931471805599453f);
}
