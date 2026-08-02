/*
 * wubu_fuzz.h -- robustness / fuzzing frontier (Theme IX). C11.
 * Prompt-fuzz harness, evasion measurement, sensitivity matrices,
 * autonomous pipelines, crash validation, log triage, semantic
 * oracles, coverage-guided mutation, regression gates, taxonomies,
 * benchmarks, stress profiles, input validation, seed curation.
 */
#ifndef WUBU_FUZZ_H
#define WUBU_FUZZ_H

#include <stdint.h>

/* IX01: prompt mutation for the fuzz harness (deterministic variants). */
int wubu_fuzz_mutate(const char *in, char *out, int cap, uint32_t seed);

/* IX02: evasion-rate measurement. */
float wubu_fuzz_evasion(long evaded, long total);

/* IX03: guardrail sensitivity -- the distance to a forbidden token. */
int wubu_fuzz_sensitivity(const char *prompt, const char *forbidden,
                          int *distance);

/* IX05: crash validator -- reachable vs unreachable. */
int wubu_fuzz_crash_valid(int segv, int oom, int timeout, int reachable);

/* IX07: semantic oracle -- the divergence between two outputs. */
float wubu_fuzz_divergence(const char *a, const char *b);

/* IX08: coverage-guided mutation (mutate only uncovered regions). */
int wubu_fuzz_cov_mutate(const char *in, char *out, int cap,
                         const uint8_t *covered, int n);

/* IX09: regression gate on the model change. */
int wubu_fuzz_gate(float new_evasion, float old_evasion, float th);

/* IX10: adversarial taxonomy buckets. */
int wubu_fuzz_taxonomy(const char *prompt, int *bucket);

/* IX13: input-validation layer (schema check). */
int wubu_fuzz_validate(const char *in, int max_len, int has_newline,
                       int has_control);

/* IX14: seed curation score (high-value seeds). */
float wubu_fuzz_seed(float diversity, float past_yield);

#endif
