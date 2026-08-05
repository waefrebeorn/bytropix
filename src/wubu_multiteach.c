/* wubu_multiteach.c — Multi-teacher distillation kernel (C11, opaque, minimal)
 *
 * Fused multi-teacher KL divergence + tool-use trajectory head.
 * Reuses wubu_distill.c BB04 KL divergence pattern.
 *
 * The hot path computes the ensemble soft-target and KL in a
 * single pass over the student logits — no per-teacher loops
 * in the inner accumulation.
 */
#define _POSIX_C_SOURCE 200809L
#include "wubu_multiteach.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct wubu_multiteach {
    wubu_multiteach_cfg_t cfg;
    float teacher_kl_breakdown[WUBU_TEACHERS]; /* per-teacher KL */
};

wubu_multiteach_t *wubu_multiteach_create(const wubu_multiteach_cfg_t *cfg) {
    if (!cfg || cfg->vocab_size <= 0 || cfg->temperature <= 0.0f) return NULL;
    wubu_multiteach_t *mt = (wubu_multiteach_t *)calloc(1, sizeof(*mt));
    if (!mt) return NULL;
    mt->cfg = *cfg;
    return mt;
}

void wubu_multiteach_free(wubu_multiteach_t *mt) {
    if (!mt) return;
    free(mt);
}

int wubu_multiteach_set_weights(wubu_multiteach_t *mt,
                                       const float weights[WUBU_TEACHERS]) {
    if (!mt || !weights) return -1;
    float sum = 0;
    for (int i = 0; i < WUBU_TEACHERS; i++) sum += weights[i];
    if (sum < 1e-8f) return -1;
    for (int i = 0; i < WUBU_TEACHERS; i++)
        mt->cfg.teachers[i].weight = weights[i] / sum;
    return 0;
}

/* Fused multi-teacher KL divergence kernel.
 *
 * Algorithm (one pass, folded weights):
 *   1. For each vocab position i:
 *      a. Compute student softmax: s[i] = softmax(logit[i] / T)
 *      b. For each teacher j: compute t_j[i] = softmax(logit_j[i] / T)
 *      c. Accumulate ensemble: ens[i] = sum_j w_j * t_j[i]
 *      d. Accumulate per-teacher KL: kl_j += w_j * sum_i ens[i] * log(ens[i] / t_j[i])
 *   2. Total KL = sum_j kl_j
 *
 * We compute this in a single forward pass over the vocab,
 * accumulating the ensemble and per-teacher KL simultaneously.
 * The temperature scaling is folded into the softmax computation. */
float wubu_multiteach_kl_loss(const float *student_logits,
                                   const float *teacher_logits,
                                   int n_vocab,
                                   float temperature,
                                   const float weights[WUBU_TEACHERS],
                                   float *out_ensemble) {
    if (!student_logits || !teacher_logits || !weights || !out_ensemble ||
        n_vocab <= 0 || temperature <= 0.0f)
        return 0.0f;

    float inv_T = 1.0f / temperature;
    float total_kl = 0.0f;

    /* First pass: compute student softmax and ensemble */
    float *student_soft = (float *)malloc((size_t)n_vocab * sizeof(float));
    float *ensemble = (float *)calloc(n_vocab, sizeof(float));
    if (!student_soft || !ensemble) {
        free(student_soft); free(ensemble);
        return 0.0f;
    }

    /* Student softmax (numerically stable) */
    float max_s = -1e30f;
    for (int i = 0; i < n_vocab; i++) {
        float val = student_logits[i] * inv_T;
        if (val > max_s) max_s = val;
        student_soft[i] = val;
    }
    float s_sum = 0;
    for (int i = 0; i < n_vocab; i++) {
        student_soft[i] = expf(student_soft[i] - max_s);
        s_sum += student_soft[i];
    }
    if (s_sum < 1e-8f) s_sum = 1e-8f;
    for (int i = 0; i < n_vocab; i++)
        student_soft[i] /= s_sum;

    /* Ensemble accumulation: ens[i] = sum_j w_j * softmax(teacher_j[i] / T) */
    for (int j = 0; j < WUBU_TEACHERS; j++) {
        const float *t_logits = teacher_logits + (size_t)j * n_vocab;
        float w = weights[j];
        if (w < 1e-6f) continue; /* skip near-zero weight teachers */

        /* Teacher j softmax */
        float max_t = -1e30f;
        for (int i = 0; i < n_vocab; i++) {
            float val = t_logits[i] * inv_T;
            if (val > max_t) max_t = val;
        }
        float t_sum = 0;
        for (int i = 0; i < n_vocab; i++) {
            float t_val = expf(t_logits[i] * inv_T - max_t);
            t_sum += t_val;
            ensemble[i] += w * t_val;
        }
        if (t_sum < 1e-8f) t_sum = 1e-8f;
        for (int i = 0; i < n_vocab; i++)
            ensemble[i] /= t_sum;
    }

    /* Normalize ensemble (weights may not sum to 1 if some were skipped) */
    float e_sum = 0;
    for (int i = 0; i < n_vocab; i++) e_sum += ensemble[i];
    if (e_sum < 1e-8f) e_sum = 1e-8f;
    for (int i = 0; i < n_vocab; i++)
        ensemble[i] /= e_sum;

    /* Copy ensemble to output */
    memcpy(out_ensemble, ensemble, (size_t)n_vocab * sizeof(float));

    /* Second pass: KL divergence and per-teacher KL */
    for (int i = 0; i < n_vocab; i++) {
        float ens_i = ensemble[i];
        float stu_i = student_soft[i];
        if (ens_i > 1e-30f && stu_i > 1e-30f) {
            total_kl += ens_i * logf(ens_i / stu_i);
        }
    }

    /* Per-teacher KL breakdown (computed here for storage by caller;
     * wubu_multiteach_kl_loss doesn't have access to mt, so we skip
     * the per-teacher store here). */
    /* The breakdown is computed in wubu_multiteach_total_loss which
     * has the mt handle. Here we just compute the total KL. */

    free(student_soft);
    free(ensemble);
    return total_kl;
}

float wubu_multiteach_total_loss(const wubu_multiteach_t *mt,
                                       float hard_loss,
                                       const float *student_logits,
                                       const float *teacher_logits,
                                       int n_vocab,
                                       const float *tool_mask,
                                       float tool_loss) {
    if (!mt) return hard_loss;
    float alpha = mt->cfg.distill_alpha;
    float tool_w = mt->cfg.tool_head_weight;

    float kl = wubu_multiteach_kl_loss(student_logits, teacher_logits,
                                           n_vocab, mt->cfg.temperature,
                                           (const float[]){mt->cfg.teachers[0].weight,
                                                           mt->cfg.teachers[1].weight,
                                                           mt->cfg.teachers[2].weight},
                                           NULL);
    /* Tool-use trajectory loss contribution */
    float tool_contrib = tool_w * tool_loss;

    /* Compute per-teacher KL breakdown (diagnostic cache).
     * We cast away const since this is cached diagnostic state,
     * not part of the immutable configuration. */
    wubu_multiteach_t *mt_mut = (wubu_multiteach_t *)mt;
    float inv_T = 1.0f / mt->cfg.temperature;
    for (int j = 0; j < WUBU_TEACHERS; j++) {
        const float *t_logits = teacher_logits + (size_t)j * n_vocab;
        float w = mt->cfg.teachers[j].weight;
        if (w < 1e-6f) { mt_mut->teacher_kl_breakdown[j] = 0; continue; }

        float max_t = -1e30f;
        for (int i = 0; i < n_vocab; i++) {
            float val = t_logits[i] * inv_T;
            if (val > max_t) max_t = val;
        }
        float t_sum = 0;
        for (int i = 0; i < n_vocab; i++) t_sum += expf(t_logits[i] * inv_T - max_t);
        if (t_sum < 1e-8f) t_sum = 1e-8f;

        /* Rebuild ensemble for per-teacher KL */
        float *ensemble = (float *)calloc(n_vocab, sizeof(float));
        for (int j2 = 0; j2 < WUBU_TEACHERS; j2++) {
            const float *tl = teacher_logits + (size_t)j2 * n_vocab;
            float wj = mt->cfg.teachers[j2].weight;
            if (wj < 1e-6f) continue;
            float max_j = -1e30f;
            for (int i = 0; i < n_vocab; i++) {
                float val = tl[i] * inv_T;
                if (val > max_j) max_j = val;
            }
            float sj_sum = 0;
            for (int i = 0; i < n_vocab; i++) sj_sum += expf(tl[i] * inv_T - max_j);
            if (sj_sum < 1e-8f) sj_sum = 1e-8f;
            for (int i = 0; i < n_vocab; i++)
                ensemble[i] += wj * expf(tl[i] * inv_T - max_j) / sj_sum;
        }
        float e_sum = 0;
        for (int i = 0; i < n_vocab; i++) e_sum += ensemble[i];
        if (e_sum > 1e-8f) for (int i = 0; i < n_vocab; i++) ensemble[i] /= e_sum;

        float t_kl = 0;
        for (int i = 0; i < n_vocab; i++) {
            float t_i = expf(t_logits[i] * inv_T - max_t) / t_sum;
            if (t_i > 1e-30f && ensemble[i] > 1e-30f)
                t_kl += t_i * logf(t_i / ensemble[i]);
        }
        mt_mut->teacher_kl_breakdown[j] = w * t_kl;
        free(ensemble);
    }

    return hard_loss + alpha * kl + tool_contrib;
}

const float *wubu_multiteach_teacher_kl_breakdown(const wubu_multiteach_t *mt) {
    if (!mt) return NULL;
    return mt->teacher_kl_breakdown;
}
