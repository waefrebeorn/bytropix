/*
 * wubu_bonzi.h -- Bonzi Buddy companion core (Theme JE, first batch).
 * C11. The LOGIC core (the WuBuOS/wubufx side renders it).
 *
 * Convergence (AIVA emotion-aware companion; affective computing;
 * abstract-avatar advantage; parasocial HAI):
 *   - Emotion state machine (valence x arousal + coherent transitions)
 *   - Persona reactivity + empathy engine (emotion-aware selection)
 *   - Idle animation scheduler (energy-aware, ties IJ)
 *   - Speech prosody mapping (mood -> tone)
 *   - Turn-timing + turn-taking (natural latency)
 *   - Mood memory (patterns over time, ties IP) + decay (ties IP12)
 *   - Chat context pruning (ties IO eviction)
 *   - User-mood tracking + companion continuity
 *   - Bonzi self-model (knows its own mood) + drift monitor
 *   - Calibrated honesty (ties JD) + engagement telemetry
 */
#ifndef WUBU_BONZI_H
#define WUBU_BONZI_H

#include <stdint.h>

/* JE01: the emotion state (valence -1..1, arousal 0..1). */
typedef struct {
    float valence;
    float arousal;
    int   seq;           /* the state sequence counter */
} wubu_bonzi_mood_t;

/* JE01: the state machine -- a bounded mood transition. The mood moves
 * toward the target mood by a step, and the drift is bounded so
 * transitions stay coherent (JE13). */
int wubu_bonzi_mood_step(wubu_bonzi_mood_t *m, float t_val, float t_ar,
                         float step);

/* JE02: persona reactivity -- a response's persona-fit score. */
float wubu_bonzi_persona_score(const char *resp_tags, const char *persona_tags,
                               int n_tags);

/* JE03: idle animation scheduler -- returns 1 when an idle animation
 * should play, given the elapsed idle time and the energy budget
 * (ties IJ: idle costs J). */
int wubu_bonzi_idle(int idle_ticks, int anim_interval, float energy_left,
                    float energy_cost);

/* JE04: speech prosody mapping -- mood -> tone parameters. */
void wubu_bonzi_prosody(const wubu_bonzi_mood_t *m, float *pitch,
                        float *speed, float *energy);

/* JE06: empathy engine -- the empathetic response weight for a user
 * mood (sad users get more warmth). */
float wubu_bonzi_empathy(float user_valence, float warmth);

/* JE07/JE22: turn-timing -- the natural response latency (ticks) from
 * the arousal + the message length. */
int wubu_bonzi_turn_delay(float arousal, int msg_len);

/* JE08: mood memory -- record a mood sample; returns the sample count. */
int wubu_bonzi_mood_log(wubu_bonzi_mood_t *log, int *n, int max,
                        const wubu_bonzi_mood_t *sample);

/* JE12: notification personality -- a notification's persona tone. */
float wubu_bonzi_notify_tone(float urgency, float mood_valence);

/* JE14: companion memory -- a conversational memory slot with an
 * importance weight (ties AE). */
int wubu_bonzi_mem_write(float *slots, float *importance, int n_slots,
                         int slot, float value, float importance_value);

/* JE15: chat context pruning -- evict the oldest turns past the budget,
 * keeping the sink turn (ties IO). Returns the evicted count. */
int wubu_bonzi_chat_prune(int turns, int budget);

/* JE17: energy-aware idle -- the idle J cost vs the remaining budget. */
int wubu_bonzi_idle_afford(float energy_left, float idle_cost);

/* JE20: user-mood tracking -- the EMA user valence. */
float wubu_bonzi_user_mood(float prev, float sample, float alpha);

/* JE21: Bonzi self-model -- the mood self-assessment: Bonzi's own
 * confidence about its current mood (ties JD02). */
float wubu_bonzi_self_mood(const wubu_bonzi_mood_t *m, float clarity);

/* JE26: companion continuity -- the session identity hash that keeps
 * Bonzi continuous across sessions. */
uint32_t wubu_bonzi_session_id(const char *user, uint32_t seed);

/* JE32: engagement telemetry -- the rolling engagement score. */
float wubu_bonzi_engagement(float *window, int n, float new_score);

/* JE37: calibrated honesty -- Bonzi's answer confidence, clamped by
 * the calibration (ties JD03). */
float wubu_bonzi_honesty(float confidence, float calibration);

/* JE42: mood-drift monitor -- flags when the mood moved too far in one
 * step (the coherence guard). */
int wubu_bonzi_mood_anomaly(const wubu_bonzi_mood_t *prev,
                            const wubu_bonzi_mood_t *cur, float max_delta);

/* JE67: emotional memory decay -- the memory weight at age t (ties
 * IP12/IP25 forgetting curve). */
float wubu_bonzi_mem_decay(float base, int age, float halflife);

#endif
