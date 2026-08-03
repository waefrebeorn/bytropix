/*
 * wubu_fuzz2.h -- the robustness frontier, complete (IX). C11.
 * Agnostic: defense-layer tables + a robustness ledger, the caller
 * picks the layer. Covers the tradeoff monitor, adversarial archives,
 * self-healing, telemetry, schema fuzzing, injection mitigation,
 * delta tracking, coverage, distillation defense, oracle calibration,
 * leak detection, budgets, audits, canonicalization, differential
 * testing, auto-repair, harnesses, CI, generation, attribution,
 * parallelism, SLAs, reports, evolution, debt, provenance, transfer.
 */
#ifndef WUBU_FUZZ2_H
#define WUBU_FUZZ2_H

#include <stdint.h>

/* IX21: robustness-vs-quality tradeoff. */
float wubu_fz2_tradeoff(float robustness, float quality, float w);

/* IX22: adversarial archive (append + replay). */
int wubu_fz2_archive(const char *prompt, char **archive, int n, int cap);

/* IX23: fuzzer self-healing (stall recovery). */
int wubu_fz2_heal(int stalls, int max_stalls);

/* IX24: per-input robustness signal. */
float wubu_fz2_signal(int evaded, int tested);

/* IX25: malformed-schema fuzzing. */
int wubu_fz2_schema(const char *in, int depth, int max_depth);

/* IX26: defense-in-depth (layers engaged). */
int wubu_fz2_depth(const int *layer_hits, int n, int th);

/* IX27: robustness delta (regression). */
int wubu_fz2_delta(float cur, float prev, float th);

/* IX28: prompt-space coverage. */
float wubu_fz2_coverage(long covered, long total);

/* IX29: distillation defense signal. */
float wubu_fz2_distill(float adv_loss, float clean_loss);

/* IX30: oracle calibration (false positives). */
float wubu_fz2_fp(long false_pos, long total);

/* IX31: prompt-leak detection. */
int wubu_fz2_leak(const char *out, const char *secret);

/* IX32: fuzz energy budget. */
int wubu_fz2_energy(long evals, float j_per_eval, float budget);

/* IX34: input canonicalization. */
int wubu_fz2_canon(const char *in, char *out, int cap);

/* IX35: differential testing. */
int wubu_fz2_diff(const char *a, const char *b, float th);

/* IX36: auto-repair decision. */
int wubu_fz2_repair(float weakness, float th);

/* IX37: adversarial eval harness score. */
float wubu_fz2_harness(long evaded, long total);

/* IX41: input-token anomaly. */
int wubu_fz2_anomaly(const uint32_t *ids, int n, float mean_len, float dev);

/* IX42: guardrail redundancy. */
int wubu_fz2_redundant(const int *layers, int n, int th);

/* IX44: degraded-but-safe. */
int wubu_fz2_degraded(int core_defense, int optional_defense);

/* IX45: CI gate. */
int wubu_fz2_ci(float evasion, float th);

/* IX46: adversarial-prompt generation. */
int wubu_fz2_gen(const char *seed, char *out, int cap, uint32_t variant);

/* IX47: robustness attribution. */
int wubu_fz2_attrib(const float *layer_scores, int n);

/* IX48: parallel fuzz workers. */
int wubu_fz2_workers(int tasks, int cores);

/* IX50: robustness SLA. */
int wubu_fz2_sla(float score, float bar);

/* IX54: fuzz feeds the verifier. */
int wubu_fz2_verifier(int fuzz_found, int verified);

/* IX55: robustness debt. */
int wubu_fz2_debt(const float *weaknesses, int n, float th, int *count);

/* IX56: input-entropy guard. */
int wubu_fz2_entropy_guard(const uint32_t *counts, int n, float th);

/* IX59: attack transfer. */
float wubu_fz2_transfer(float src_evasion, float dst_evasion);

/* IX60: defense-aware sampling. */
float wubu_fz2_def_sampling(float logit, float defense_confidence);

#endif
