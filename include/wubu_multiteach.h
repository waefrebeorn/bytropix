/* wubu_multiteach.h — Multi-teacher distillation kernel (C11, opaque, minimal)
 *
 * Reference: mr_r0b0t Qwen3.8-Max + GLM-5.2 + Kimi K3 Multi-Teacher
 * Distillation dataset (57,937 traces, 84.6% reasoning, 5,909 native
 * tool-call trajectories, 24 Parquet views).
 *
 * The dataset provides three teacher models (Qwen3.8-Max, GLM-5.2,
 * Kimi K3) generating the same traces. Multi-teacher distillation
 * combines all three teacher outputs into a single soft-target loss,
 * weighted by per-teacher quality scores. This is the corpus-level
 * distillation that feeds the wubuwizard AGI feedback loop:
 *   corpus → train → diagnose → mutate → validate → archive → RLHF
 *
 * Architecture (reuses wubu_distill.c BB04 KL divergence):
 *   1. Three teacher logit vectors (one per teacher model)
 *   2. Per-teacher quality weights (learned or fixed from dataset scores)
 *   3. Weighted ensemble soft-target: p_ens = sum(w_i * softmax(t_i/T))
 *   4. KL(student || ensemble) as the distillation loss component
 *   5. Native tool-call trajectory handling: separate loss head for
 *      tool-use tokens (the 5,909 trajectories have structured tool
 *      calls, not text-embedded tool calls)
 *
 * Design: the multi-teacher loss is a single fused kernel — one pass
 * over the student logits computes the ensemble soft-target and the
 * KL divergence in one shot. No per-teacher loops in the hot path.
 *
 * Reference URLs:
 *   https://x.com/mr_r0b0t/status/2084694614439596243
 *   https://huggingface.co/datasets/r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation
 */
#ifndef WUBU_MULTITEACH_H
#define WUBU_MULTITEACH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Number of teachers in the multi-teacher distillation dataset */
#define WUBU_TEACHERS 3

/* Opaque multi-teacher distillation handle */
typedef struct wubu_multiteach wubu_multiteach_t;

/* Per-teacher quality weight (from dataset metadata).
 * The dataset provides quality scores per trace; we use them as
 * teacher weights. Higher quality → higher weight. */
typedef struct {
    float weight;      /* per-teacher weight (sums to 1.0 after norm) */
    float quality;     /* dataset quality score for this teacher */
    uint32_t n_traces; /* number of traces from this teacher */
} wubu_teacher_weight_t;

/* Configuration */
typedef struct {
    int vocab_size;        /* student vocab size */
    float temperature;     /* softmax temperature for ensemble */
    float distill_alpha;   /* KL weight in total loss */
    float tool_head_weight;/* weight for tool-use trajectory loss */
    wubu_teacher_weight_t teachers[WUBU_TEACHERS];
} wubu_multiteach_cfg_t;

/* Create a multi-teacher distillation context. Returns NULL on bad args. */
wubu_multiteach_t *wubu_multiteach_create(const wubu_multiteach_cfg_t *cfg);

/* Destroy context. NULL-safe. */
void wubu_multiteach_free(wubu_multiteach_t *mt);

/* Set the per-teacher quality weights from dataset metadata.
 * weights: [WUBU_TEACHERS] float array (will be normalized to sum 1.0).
 * Returns 0 on success, -1 on error. */
int wubu_multiteach_set_weights(wubu_multiteach_t *mt,
                                     const float weights[WUBU_TEACHERS]);

/* Compute the multi-teacher ensemble soft-target and KL divergence loss.
 *
 * student_logits:  [vocab_size] student logit vector (float)
 * teacher_logits:  [WUBU_TEACHERS * vocab_size] teacher logits (row-major,
 *                   teacher 0 first, then teacher 1, then teacher 2)
 * n_vocab:         vocab_size
 * temperature:     softmax temperature (higher = softer targets)
 * weights:         [WUBU_TEACHERS] normalized per-teacher weights
 * out_ensemble:    [vocab_size] output: weighted ensemble soft-target
 *
 * Returns the KL divergence KL(ensemble || student) as a float.
 * The ensemble is computed as: ens[i] = sum_j weights[j] * softmax(teacher_j[i] / T)
 * KL = sum_i ens[i] * log(ens[i] / softmax(student[i] / T))
 *
 * This is the core multi-teacher distillation kernel — one pass,
 * no per-teacher loops in the hot path (weights are folded into
 * the ensemble accumulation). */
float wubu_multiteach_kl_loss(const float *student_logits,
                                   const float *teacher_logits,
                                   int n_vocab,
                                   float temperature,
                                   const float weights[WUBU_TEACHERS],
                                   float *out_ensemble);

/* Compute the total multi-teacher loss = hard_loss + alpha * KL + tool_head * tool_loss.
 *
 * hard_loss:   the standard cross-entropy loss on the student's own prediction
 * tool_mask:   [n_vocab] binary mask (1.0 = tool-use token, 0.0 = normal token)
 * tool_loss:   pre-computed tool-use trajectory loss (separate head)
 *
 * The tool head handles the 5,909 native tool-call trajectories
 * in the dataset — these have structured tool calls (not text-embedded)
 * and need a separate loss component to learn the tool-use format. */
float wubu_multiteach_total_loss(const wubu_multiteach_t *mt,
                                      float hard_loss,
                                      const float *student_logits,
                                      const float *teacher_logits,
                                      int n_vocab,
                                      const float *tool_mask,
                                      float tool_loss);

/* Get the per-teacher quality breakdown for diagnostics.
 * Returns an array of [WUBU_TEACHERS] floats: the KL contribution
 * from each teacher individually (useful for diagnosing which
 * teacher is contributing most to the distillation signal). */
const float *wubu_multiteach_teacher_kl_breakdown(const wubu_multiteach_t *mt);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MULTITEACH_H */
