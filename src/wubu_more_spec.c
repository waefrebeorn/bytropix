/*
 * wubu_more_spec.c -- Remaining speculative-decoding policy variants
 * (M07/M08/M09/M10/M15/M17/M18/M19/M20). C11.
 *
 * Convergence (EAGLE/Medusa/REST/cascade 7-hop): the remaining M-family gaps are
 * *policy* decisions that compose already-wired machinery (spec_tuner K, kv_evict
 * importance, wubu_layer_skip early-exit, continuous_batching). This module gives
 * each a small, testable pure function so the operator can select/apply it:
 *   - M07 REST: residual-estimating draft token confidence -> accept if |resid|<thr.
 *   - M08 tree restructure: given accepted branch depth and a re-verify budget, say
 *        whether to restructure the draft tree this step.
 *   - M09 contrastive: accept only if draft prob exceeds reference prob (lossless).
 *   - M10 distil: gate a draft-model swap on a distillation quality estimate.
 *   - M15 spec MoE: skip experts in the draft whose routing score < floor.
 *   - M17 cascade: accept the big-model verification if small-model draft agreed.
 *   - M18 swap: switch draft model when acceptance < low for `patience` steps.
 *   - M19 layer-stream resume: 1 if a partially-streamed layer can resume decode.
 *   - M20 cascade+early-exit: combine M17 + layer_skip early-exit decision.
 *
 * Triple-DA: probabilities clamped to [0,1]; thresholds in (0,1); deterministic.
 */
#include "wubu_more_spec.h"
#include <stdlib.h>
#include <math.h>

/* M07 REST: accept draft if residual magnitude below thr (lossless-ish). */
int wubu_rest_accept(float residual, float thr) {
    if (residual < 0.0f) residual = -residual;
    if (thr <= 0.0f) thr = 1e-3f;
    return (residual < thr) ? 1 : 0;
}

/* M08 tree restructure: restructure when the accepted depth < budget (tree too
 * shallow -> expand) OR depth == budget (full -> verify). Returns 1 to restructure. */
int wubu_tree_restructure(int accepted_depth, int budget) {
    if (budget <= 0) return 0;
    if (accepted_depth < 0) accepted_depth = 0;
    if (accepted_depth >= budget) return 0;   /* full tree, verify */
    return (accepted_depth < budget / 2) ? 1 : 0; /* too shallow -> grow */
}

/* M09 contrastive: accept only if draft_prob >= ref_prob (lossless). */
int wubu_contrastive_accept(float draft_p, float ref_p) {
    if (draft_p < 0.0f) draft_p = 0.0f; if (draft_p > 1.0f) draft_p = 1.0f;
    if (ref_p   < 0.0f) ref_p   = 0.0f; if (ref_p   > 1.0f) ref_p   = 1.0f;
    return (draft_p >= ref_p) ? 1 : 0;
}

/* M10 distil gate: swap to distilled draft when quality estimate >= qmin. */
int wubu_distil_gate(float quality_est, float qmin) {
    if (quality_est < 0.0f) quality_est = 0.0f;
    if (qmin < 0.0f) qmin = 0.0f;
    return (quality_est >= qmin) ? 1 : 0;
}

/* M15 spec MoE: skip an expert whose routing score < floor (returns 1=skip). */
int wubu_spec_moe_skip(float route_score, float floor) {
    if (route_score < 0.0f) route_score = 0.0f;
    if (floor < 0.0f) floor = 0.0f;
    return (route_score < floor) ? 1 : 0;
}

/* M17 cascade: accept big-model output if small-model draft agreed (match==1). */
int wubu_cascade_accept(int draft_match) { return draft_match ? 1 : 0; }

/* M18 swap: swap draft model when acceptance < low for `patience` consecutive
 * steps. Stateful counter; returns 1 when a swap should fire. */
int wubu_swap_check(int *streak, float acceptance, float low, int patience) {
    if (!streak || patience <= 0) return 0;
    if (low < 0.0f) low = 0.0f;
    if (acceptance < low) (*streak)++;
    else *streak = 0;
    return (*streak >= patience) ? 1 : 0;
}

/* M19 layer-stream resume: resume decode if a layer is partially streamed
 * (streamed > 0 and < total). */
int wubu_layer_resume(int streamed, int total) {
    if (total <= 0) return 0;
    if (streamed <= 0) return 0;
    return (streamed < total) ? 1 : 0;
}

/* M20 cascade + early-exit: accept if cascade matched AND the current layer is
 * allowed to early-exit (layer < floor_layer). */
int wubu_cascade_earlyexit(int draft_match, int layer, int floor_layer) {
    if (!draft_match) return 0;
    if (floor_layer < 0) floor_layer = 0;
    return (layer < floor_layer) ? 1 : 0;
}
