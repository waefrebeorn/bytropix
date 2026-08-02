/*
 * wubu_rsi.h -- recursive self-improvement frontier (Theme IV). C11.
 * Bounded verifiable loops, self-referential improvement, recursive
 * decomposition, prompt evolution, metacognitive transfer, harness
 * synthesis, self-reflection, soft-mellowmax planning, experience
 * loops, synthetic data, weak-to-strong, scaffolding, awareness,
 * bounded self-modification, fine-tune scheduling.
 */
#ifndef WUBU_RSI_H
#define WUBU_RSI_H

#include <stdint.h>

/* IV01: verifiable RSI gate -- improve only if the verifier passes. */
int wubu_rsi_gate(float verifier_score, float th, int *consecutive_fails);

/* IV02: Goedel-style self-reference -- the agent improves its improver. */
int wubu_rsi_improve_improver(float self_score, float meta_score, float th);

/* IV03: LADDER decomposition -- split a hard goal recursively. */
int wubu_rsi_decompose(float difficulty, float budget, int depth,
                       int *n_subgoals);

/* IV04: prompt mutation (Promptbreeder-style) with a fitness gate. */
int wubu_rsi_prompt_mutate(const char *parent, char *child, int cap,
                           float fitness);

/* IV05: metacognitive transfer -- reuse a strategy across domains. */
float wubu_rsi_transfer(float src_perf, float similarity);

/* IV06: harness synthesis -- auto-generate a test harness score. */
float wubu_rsi_harness(float coverage, float asserts);

/* IV07: intrinsic self-reflection -- the policy improves from its
 * own rollouts (the preference self-supervision). */
int wubu_rsi_reflect(const float *win, const float *lose, int n,
                     float *grad);

/* IV08: soft-mellowmax planning -- softmax-planned MCTS value. */
float wubu_rsi_mellowmax(const float *values, int n, float omega);

/* IV09: experience loop -- streaming telemetry -> improvement. */
typedef struct { long evals, wins; float running; } wubu_rsi_exp_t;
int wubu_rsi_experience(wubu_rsi_exp_t *e, int win, float value);

/* IV10: synthetic-data pipeline (self-generated, filtered). */
int wubu_rsi_synth(float quality, float diversity, float th);

/* IV11: weak-to-strong -- the small teacher supervises the big student. */
float wubu_rsi_weak2strong(float teacher_acc, float agreement);

/* IV12: scaffolding improvement score. */
float wubu_rsi_scaffold(float steps_saved, float reliability);

/* IV14: self-awareness audit -- calibrated capability estimate. */
float wubu_rsi_awareness(float predicted, float actual);

/* IV15: bounded self-modification -- the safe-pace weight delta. */
float wubu_rsi_bounded_delta(float grad, float max_step, float budget_left);

/* IV16: fine-tune scheduler -- when to schedule a fine-tune. */
int wubu_rsi_ft_schedule(long evals, long every, float drift);

#endif
