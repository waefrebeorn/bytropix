/*
 * wubu_bonzi2.h -- the companion frontier, complete (JE). C11.
 * Agnostic: a companion-state + emotion policy table, the caller
 * picks the strategy. Covers the full JE theme: multimodal
 * sentiment, reactive animations, abstract avatar, social-presence
 * cues, persona guardrails, speech batching, empathy escalation,
 * emotional vocabulary, avatar micro-expressions, voice personality,
 * proactive engagement, mood-lighting UI, parasocial design,
 * emotional anchoring, apology/repair, mood-triggered actions,
 * companion topic memory, speech emotion synthesis, animation-mood
 * coupling, idle-noise budget.
 */
#ifndef WUBU_BONZI2_H
#define WUBU_BONZI2_H

#include <stdint.h>

/* JE05: multimodal sentiment perception. */
int wubu_bonzi_sentiment(const float *text_feat, const float *voice_feat,
                              int d, float *sentiment);

/* JE09: reactive animations. */
int wubu_bonzi_react(float mood, float *anim);

/* JE10: abstract-avatar design. */
int wubu_bonzi_avatar(float *params, int n);

/* JE11: social-presence cues. */
float wubu_bonzi_presence(float voice_quality, float avatar_quality);

/* JE16: persona guardrails. */
int wubu_bonzi_guardrail(const char *response, const char *persona, int cap);

/* JE18: speech batching. */
int wubu_bonzi_batch(const float *requests, int n, int max_batch, int *out);

/* JE19: empathy escalation. */
int wubu_bonzi_empathy(float mood, float intensity, float *response_depth);

/* JE23: emotional vocabulary. */
int wubu_bonzi_vocab(const char *mood, char *vocab, int cap);

/* JE24: avatar micro-expressions. */
int wubu_bonzi_micro(float *expression, int n, float intensity);

/* JE25: voice personality. */
int wubu_bonzi_voice(float *params, int n, const char *personality);

/* JE27: proactive engagement. */
int wubu_bonzi_proactive(float idle_time, float mood, int *action);

/* JE28: mood-lighting UI. */
int wubu_bonzi_lighting(float mood, float *rgb);

/* JE29: parasocial design. */
float wubu_bonzi_parasocial(float presence, float autonomy);

/* JE30: emotional anchoring. */
int wubu_bonzi_anchor(const float *memories, int n, float *anchor);

/* JE31: apology/repair scripts. */
int wubu_bonzi_repair(float error_severity, char *script, int cap);

/* JE33: mood-triggered actions. */
int wubu_bonzi_mood_action(float mood, int *action);

/* JE34: companion topic memory. */
int wubu_bonzi_topic_mem(const char *topic, const float *embedding, int d);

/* JE35: speech emotion synthesis. */
int wubu_bonzi_speech_emotion(const float *text, int n, float *emotion);

/* JE36: animation-mood coupling. */
int wubu_bonzi_couple(float mood, float *anim_params);

/* JE38: idle-noise budget. */
int wubu_bonzi_idle_budget(float idle_time, float j_cap, float *budget);

#endif