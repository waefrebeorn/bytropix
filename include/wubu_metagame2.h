/*
 * wubu_metagame2.h -- the metacognition frontier, complete (JD). C11.
 * Agnostic: a metacog-state + regulation policy table, the caller
 * picks the strategy. Covers the full JD theme: meta-level regulation,
 * strategy selection, confidence-conditioned compute, reflection prompts,
 * capability asymmetry, learning-progress prediction, self-assessment
 * audit, skill library, regulation policy, energy budget, confidence
 * stability, feedback loop, capability transfer, self-prediction,
 * early stopping, regulation under budget, competence-weighted delegation,
 * monitor independence, benchmarks, calibration-cost model.
 */
#ifndef WUBU_METAGAME2_H
#define WUBU_METAGAME2_H

#include <stdint.h>

/* JD06: meta-level regulation (Meta-R1 style). */
int wubu_meta_regulate(const float *policy_conf, int n, float th, int *action);

/* JD07: strategy selection by competence. */
int wubu_meta_strategy(const float *competence, int n, int *chosen);

/* JD15: confidence-conditioned compute allocation. */
int wubu_meta_compute(float confidence, float budget, float *alloc);

/* JD20: metacognitive reflection prompts. */
int wubu_meta_reflect(const char *prompt, char *reflection, int cap);

/* JD22: capability asymmetry detection. */
int wubu_meta_asymmetry(const float *caps_a, const float *caps_b, int n, float *diff);

/* JD25: learning-progress prediction. */
float wubu_meta_progress(const float *history, int n, float lr);

/* JD26: self-assessment audit. */
int wubu_meta_audit(float self_score, float ground_truth, float th);

/* JD27: metacognitive skill library. */
int wubu_meta_skill_lib(const char *skill, int n_skills, int *found);

/* JD28: regulation policy (retry/stop/delegate). */
int wubu_meta_reg_policy(float error_rate, float confidence, int *action);

/* JD29: metacog energy budget. */
float wubu_meta_energy(long self_monitored_ops, float j_per_op);

/* JD31: confidence-stability check. */
int wubu_meta_stability(const float *confidence, int n, float th);

/* JD32: metacog feedback into the loop-ledger. */
int wubu_meta_feedback(float error, float *ledger, int n);

/* JD33: capability-transfer prediction. */
float wubu_meta_transfer(const float *src_cap, const float *dst_cap, int n);

/* JD34: self-predicted pass@1. */
float wubu_meta_pass1(float confidence, int n_tasks);

/* JD35: metacog-driven early stopping. */
int wubu_meta_early_stop(float confidence, float best_loss, float th);

/* JD36: regulation under budget. */
int wubu_meta_reg_budget(float reg_cost, float budget);

/* JD37: competence-weighted delegation. */
int wubu_meta_delegate(const float *competence, int n, int *delegated);

/* JD38: self-monitoring independence check. */
int wubu_meta_independence(float actor_score, float monitor_score, float th);

/* JD39: metacog benchmarks per dimension. */
float wubu_meta_bench(const float *scores, int n);

/* JD40: calibration-cost model. */
float wubu_meta_calib_cost(float effort, float benefit);

#endif