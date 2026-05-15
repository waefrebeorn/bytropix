/**
 * Poincaré router backward pass (single-level + nested 2-level).
 *
 * Single-level backward: see wubu_poincare_router_backward below.
 * Nested 2-level backward: see wubu_nested_moe_router_backward.
 *
 * Forward (nested):
 *   Level 1: Poincaré distance to 16 coarse centroids (R=1.5) → softmax → top-2 groups
 *   Level 2: Poincaré distance to 16 fine centroids in each selected group (R=0.5) → scores
 *   Blend:   final_score = 0.3 * group_score + 0.7 * expert_score
 *   Softmax over 32 candidates → top-8 experts with normalized weights
 *
 * Backward (straight-through estimation):
 *   - top-k selections treated as non-differentiable
 *   - Backprop through: weight normalization → final softmax → blend coefficients
 *     → fine-level distance gradients → fine centroids + input
 *     → coarse-level softmax → coarse distance gradients → coarse centroids + input
 */

#include "wubu_moe_hyperbolic.h"
#include "wubu_mobius.h"
#include "gguf_reader.h"  // wubu_log_map, wubu_exp_map
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ============================================================
// Internal helper: backprop through one Poincaré distance
// ============================================================
// Computes gradient of d = poincare_dist(x_ball, centroid, R) w.r.t.
// x_ball and centroid, given d_dist = d(loss)/d(distance).
//
// Uses the same Möbius add gradient logic as the single-level backward.
//
// Parameters:
//   x_ball: [d] — point in Poincaré ball
//   centroid: [d] — centroid in Poincaré ball
//   d: dimension
//   R: Poincaré ball radius
//   d_dist: d(loss)/d(distance) — scalar gradient
//   d_x_ball: [d] — output gradient w.r.t. x_ball (add to existing)
//   d_centroid: [d] — output gradient w.r.t. centroid (add to existing, or NULL)
//   neg_x, z, d_z: [d] scratch buffers (pre-allocated)
static inline void poincare_dist_backward_one(
    const float *x_ball, const float *centroid, int d, float R,
    float d_dist,
    float *d_x_ball, float *d_centroid,
    float *neg_x, float *z, float *d_z)
{
    // z = (-x_ball) ⊕ centroid  (Möbius addition)
    for (int i = 0; i < d; i++) neg_x[i] = -x_ball[i];
    wubu_mobius_add(neg_x, centroid, d, R, z);

    // Norm of z
    float nz2 = 0;
    for (int i = 0; i < d; i++) nz2 += z[i] * z[i];
    float nz = sqrtf(nz2);
    if (nz < 1e-12f) return;  // zero distance → no gradient

    // d(dist)/dz_i = R² * z_i / (nz * (R² - nz²))
    float R2 = R * R;
    float denom = nz * (R2 - nz * nz);
    if (denom < 1e-15f) denom = 1e-15f;
    for (int i = 0; i < d; i++)
        d_z[i] = d_dist * R2 * z[i] / denom;

    // Backprop through mobius_add(neg_x, centroid):
    // z = mobius_add(neg_x, centroid)
    // Need d_neg_x and d_centroid from d_z
    float c = 1.0f / R2;
    float n_negx2 = 0, n_cente2 = 0, d_negx_cente = 0;
    for (int i = 0; i < d; i++) {
        n_negx2 += neg_x[i] * neg_x[i];
        n_cente2 += centroid[i] * centroid[i];
        d_negx_cente += neg_x[i] * centroid[i];
    }

    if (n_negx2 < 1e-30f) {
        // neg_x ≈ 0: z ≈ centroid, d_centroid = d_z
        if (d_centroid) {
            for (int i = 0; i < d; i++) d_centroid[i] += d_z[i];
        }
        // d_neg_x = 0 (no gradient to input from this)
    } else if (n_cente2 < 1e-30f) {
        // centroid ≈ 0: z ≈ neg_x, d_neg_x = d_z
        if (d_x_ball) {
            for (int i = 0; i < d; i++) d_x_ball[i] -= d_z[i]; // neg_x = -x_ball
        }
        // d_centroid = 0
    } else {
        // General case: O(d) gradient
        float S_xz = 0, S_yz = 0, S_zz = 0;
        for (int i = 0; i < d; i++) {
            S_xz += d_z[i] * neg_x[i];
            S_yz += d_z[i] * centroid[i];
            S_zz += d_z[i] * z[i];
        }

        float cny2 = c * n_cente2, cnx2 = c * n_negx2;
        float two_c_dot = 2.0f * c * d_negx_cente;
        float alpha = 1.0f + two_c_dot + cnx2 * c * n_cente2;
        if (fabsf(alpha) < 1e-30f) alpha = 1e-30f;
        float beta = 1.0f + two_c_dot + cny2;
        float gamma = 1.0f - cnx2;
        float inv_alpha = 1.0f / alpha;
        float two_c_inv_a = 2.0f * c * inv_alpha;
        float two_c2_ny2_ia = 2.0f * c * c * n_cente2 * inv_alpha;
        float two_c2_nx2_ia = 2.0f * c * c * n_negx2 * inv_alpha;

        // d_neg_x (d_x for first argument of mobius_add)
        if (d_x_ball) {
            for (int i = 0; i < d; i++) {
                float dx = beta * d_z[i] * inv_alpha
                         + two_c_inv_a * centroid[i] * S_xz
                         - two_c_inv_a * neg_x[i] * S_yz
                         - (two_c_inv_a * centroid[i] + two_c2_ny2_ia * neg_x[i]) * S_zz;
                d_x_ball[i] -= dx;  // neg_x = -x_ball, so d_x_ball = -d_neg_x
            }
        }

        // d_centroid (d_y for second argument)
        if (d_centroid) {
            for (int i = 0; i < d; i++) {
                float dy = gamma * d_z[i] * inv_alpha
                         + two_c_inv_a * (neg_x[i] + centroid[i]) * S_xz
                         - two_c_inv_a * neg_x[i] * S_zz
                         - two_c2_nx2_ia * centroid[i] * S_zz;
                d_centroid[i] += dy;
            }
        }
    }
}

// ============================================================
// Internal helper: backprop through exp_map
// ============================================================
// Given x (Euclidean input), x_ball = exp_map(x, R), and d_x_ball (gradient
// w.r.t. x_ball), computes d_x = d(loss)/d(x) and adds it to d_x_out.
static inline void exp_map_backward_one(
    const float *x, int d, float R,
    const float *d_x_ball,
    float *d_x_out)
{
    if (!d_x_out || !d_x_ball) return;

    float nv2 = 0;
    for (int i = 0; i < d; i++) nv2 += x[i] * x[i];
    float nv = sqrtf(nv2);

    if (nv >= 1e-12f) {
        float ratio = nv / R;
        if (ratio >= 0.9999f) ratio = 0.9999f;
        float tanh_r = tanhf(ratio);
        float sech2 = 1.0f - tanh_r * tanh_r;
        float g = tanh_r * R / nv;
        float gp = (sech2 * nv - tanh_r * R) / (nv * nv);
        float dot = 0;
        for (int i = 0; i < d; i++) dot += d_x_ball[i] * x[i];
        float factor = gp / nv;
        for (int i = 0; i < d; i++)
            d_x_out[i] += d_x_ball[i] * g + factor * x[i] * dot;
    } else {
        for (int i = 0; i < d; i++)
            d_x_out[i] += d_x_ball[i];
    }
}

// ============================================================
// Single-level Poincaré router backward
// ============================================================
void wubu_poincare_router_backward(const float *x, int B, int T,
                                   const float *scores,
                                   const float *d_scores,
                                   const poincare_router_t *router,
                                   float *d_x,
                                   float *d_centroids) {
    if (!router || !router->loaded) return;
    int N = B * T, d = D_MODEL;
    float R = R_POINCARE_INPUT;
    float temp = router->temperature;
    float inv_temp = -1.0f / temp;  // d(score)/d(dist) = -1/temp
    const float *centroids = router->centroids;

    // Zero centroids gradient if provided
    if (d_centroids)
        memset(d_centroids, 0, (int64_t)N_EXPERTS * d * sizeof(float));

    // Per-token scratch: x_ball, plus distance gradient temps
    float *x_ball = (float *)malloc(d * sizeof(float));
    float *neg_x = (float *)malloc(d * sizeof(float));
    float *z = (float *)malloc(d * sizeof(float));      // mobius_add(-x_ball, centroid)
    float *d_z = (float *)malloc(d * sizeof(float));    // distance gradient through z
    
    if (!x_ball || !neg_x || !z || !d_z) {
        free(x_ball); free(neg_x); free(z); free(d_z);
        return;
    }

    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * d;
        const float *ds_s = d_scores + s * N_EXPERTS;

        // Map input to Poincaré ball (same as forward)
        float nx2 = 0;
        for (int i = 0; i < d; i++) nx2 += x_s[i] * x_s[i];
        float nx = sqrtf(nx2);
        if (nx < 1e-30f) {
            // x_ball = 0 = origin. Point at origin has zero distance gradient.
            continue;
        }
        float ratio = nx / R;
        if (ratio >= 0.9999f) ratio = 0.9999f;
        float tanh_r = tanhf(ratio);
        float scale = tanh_r * R / nx;
        for (int i = 0; i < d; i++) x_ball[i] = x_s[i] * scale;

        // Accumulate d_x_ball
        float *d_x_ball_acc = NULL;
        if (d_x) {
            d_x_ball_acc = (float *)calloc(d, sizeof(float));
        }

        for (int e = 0; e < N_EXPERTS; e++) {
            float ds = ds_s[e];
            if (fabsf(ds) < 1e-15f) continue;
            const float *cent_e = centroids + e * d;

            float d_dist = inv_temp * ds;  // d_loss/d(dist)

            float *d_cent_e = d_centroids ? d_centroids + e * d : NULL;
            poincare_dist_backward_one(x_ball, cent_e, d, R,
                                       d_dist,
                                       d_x_ball_acc, d_cent_e,
                                       neg_x, z, d_z);
        }

        // Backprop d_x_ball_acc → d_x through exp_map backward
        if (d_x && d_x_ball_acc) {
            exp_map_backward_one(x_s, d, R, d_x_ball_acc, d_x + s * d);
            free(d_x_ball_acc);
        } else if (d_x_ball_acc) {
            free(d_x_ball_acc);
        }
    }

    free(x_ball); free(neg_x); free(z); free(d_z);
}

// ============================================================
// Nested MoE router backward (2-level hierarchy)
// ============================================================
// x:        [B*T, D_MODEL] — forward input (Euclidean)
// out_indices:  [B*T, N_ACTIVE_EXPTS] — forward selected expert indices
// out_weights:  [B*T, N_ACTIVE_EXPTS] — forward final normalized weights (not used for backward,
//                kept for API consistency; we recompute internally)
// d_out_weights: [B*T, N_ACTIVE_EXPTS] — upstream gradient w.r.t. final normalized weights
// router:   coarse + fine centroids
// d_x:      [B*T, D_MODEL] — gradient w.r.t. input (add to existing, or NULL)
// d_coarse_centroids: [N_HYPERBOLIC_GROUPS * D_MODEL] — or NULL
// d_fine_centroids: [N_EXPERTS * D_MODEL] — or NULL
//
// Straight-through: top-k at group level (coarse) and expert level (final) are
// treated as non-differentiable. Only selected groups and experts receive gradients.
void wubu_nested_moe_router_backward(
    const float *x, int B, int T,
    const int *out_indices, const float *out_weights,
    const float *d_out_weights,
    const nested_moe_router_t *router,
    float *d_x,
    float *d_coarse_centroids,
    float *d_fine_centroids)
{
    if (!router || !router->loaded) return;
    int N = B * T, d = D_MODEL;
    float temp = router->temperature;
    float inv_temp = -1.0f / temp;
    float R_coarse = R_POINCARE_COARSE;
    float R_fine = R_POINCARE_FINE;

    const int n_groups = N_HYPERBOLIC_GROUPS;
    const int n_fine_per_group = N_EXPERTS_PER_GROUP;
    const int n_candidates = N_ACTIVE_EXPTS == 8 ? 32 : (2 * N_EXPERTS_PER_GROUP);
    // n_candidates = top-2 groups × N_EXPERTS_PER_GROUP = 32

    // Zero centroid gradients if provided
    if (d_coarse_centroids)
        memset(d_coarse_centroids, 0, (int64_t)n_groups * d * sizeof(float));
    if (d_fine_centroids)
        memset(d_fine_centroids, 0, (int64_t)N_EXPERTS * d * sizeof(float));

    // Pre-allocate per-token scratch buffers
    int scratch_size = d                           // x_ball_coarse
                     + d                           // x_ball_fine
                     + n_groups                    // coarse_scores (pre-softmax, reused as softmax)
                     + n_groups                    // d_coarse_grad (softmax gradient per group)
                     + n_candidates                // candidate_blended
                     + n_candidates                // candidate_softmax
                     + n_candidates                // d_candidate_grad
                     + n_candidates                // fine_scores (-dist/temp)
                     + n_candidates                // candidate_group_idx
                     + n_candidates                // candidate_expert_ids
                     + d                           // d_x_ball_coarse_acc
                     + d                           // d_x_ball_fine_acc
                     + d                           // neg_x scratch
                     + d                           // z scratch
                     + d;                          // d_z scratch

    float *scratch = (float *)malloc(scratch_size * sizeof(float));
    if (!scratch) return;

    float *x_ball_coarse    = scratch;
    float *x_ball_fine      = x_ball_coarse + d;
    float *coarse_scores    = x_ball_fine + d;
    float *d_coarse_grad    = coarse_scores + n_groups;
    float *candidate_blended= d_coarse_grad + n_groups;
    float *candidate_softmax= candidate_blended + n_candidates;
    float *d_candidate_grad = candidate_softmax + n_candidates;
    float *fine_scores      = d_candidate_grad + n_candidates;
    float *candidate_group_idx_f = fine_scores + n_candidates;
    // candidate_expert_ids: store as ints, reuse a float array reinterpreted
    int *candidate_expert_ids = (int *)(candidate_group_idx_f + n_candidates);

    float *d_x_ball_coarse_acc = (float *)(candidate_expert_ids + n_candidates);
    float *d_x_ball_fine_acc   = d_x_ball_coarse_acc + d;
    float *neg_x               = d_x_ball_fine_acc + d;
    float *z_tmp               = neg_x + d;
    float *d_z_tmp             = z_tmp + d;

    // Working arrays for top-k
    int sel_groups[2];
    float sel_group_scores[2];

    // Per-token loop
    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * d;
        const int *inds_s = out_indices + s * N_ACTIVE_EXPTS;
        const float *d_wts_s = d_out_weights + s * N_ACTIVE_EXPTS;

        // ============================================================
        // RECOMPUTE FORWARD (same as wubu_nested_moe_router_forward)
        // ============================================================

        // Map input to both Poincaré balls
        euclidean_to_poincare_ball(x_s, d, R_coarse, x_ball_coarse);
        euclidean_to_poincare_ball(x_s, d, R_fine, x_ball_fine);

        // --- Level 1: Coarse routing ---
        for (int g = 0; g < n_groups; g++) {
            const float *cent_g = router->coarse_centroids + g * d;
            float dist = wubu_poincare_dist(x_ball_coarse, cent_g, d, R_coarse);
            coarse_scores[g] = -dist / temp;
        }
        // coarse_scores now holds pre-softmax scores
        // We need the softmax for backward, compute it
        float coarse_max = coarse_scores[0];
        for (int g = 1; g < n_groups; g++)
            if (coarse_scores[g] > coarse_max) coarse_max = coarse_scores[g];
        float coarse_sum = 0.0f;
        for (int g = 0; g < n_groups; g++) {
            coarse_scores[g] = expf(coarse_scores[g] - coarse_max);
            coarse_sum += coarse_scores[g];
        }
        float inv_coarse_sum = 1.0f / (coarse_sum + 1e-30f);
        for (int g = 0; g < n_groups; g++)
            coarse_scores[g] *= inv_coarse_sum;
        // coarse_scores now holds softmax values

        // Select top-2 groups
        topk_from_array(coarse_scores, n_groups, 2, sel_groups, sel_group_scores);

        // --- Level 2: Fine routing + blend ---
        int nc = 0;  // number of candidates (always 32 if top-2 found)

        for (int g_idx = 0; g_idx < 2; g_idx++) {
            int group = sel_groups[g_idx];
            float group_softmax = sel_group_scores[g_idx];

            for (int e = 0; e < n_fine_per_group; e++) {
                int expert_id = group * n_fine_per_group + e;
                const float *cent_e = router->fine_centroids + expert_id * d;

                float dist = wubu_poincare_dist(x_ball_fine, cent_e, d, R_fine);
                float score = -dist / temp;

                candidate_blended[nc] = 0.3f * group_softmax + 0.7f * score;
                candidate_expert_ids[nc] = expert_id;
                candidate_group_idx_f[nc] = (float)g_idx;  // store as float for convenience
                fine_scores[nc] = score;
                nc++;
            }
        }

        // Softmax over candidates
        float cand_max = candidate_blended[0];
        for (int i = 1; i < nc; i++)
            if (candidate_blended[i] > cand_max) cand_max = candidate_blended[i];
        float cand_sum = 0.0f;
        for (int i = 0; i < nc; i++) {
            candidate_softmax[i] = expf(candidate_blended[i] - cand_max);
            cand_sum += candidate_softmax[i];
        }
        float inv_cand_sum = 1.0f / (cand_sum + 1e-30f);
        for (int i = 0; i < nc; i++)
            candidate_softmax[i] *= inv_cand_sum;

        // ============================================================
        // BACKWARD PASS
        // ============================================================

        // Find which selected experts (from forward) correspond to which candidates
        int selected_candidate_idx[N_ACTIVE_EXPTS];
        int found_count = 0;
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
            int expert_id = inds_s[k];
            selected_candidate_idx[k] = -1;
            for (int i = 0; i < nc; i++) {
                if (candidate_expert_ids[i] == expert_id) {
                    selected_candidate_idx[k] = i;
                    found_count++;
                    break;
                }
            }
        }

        // If we couldn't find all selected experts, skip this token's backward
        // (This shouldn't happen with straight-through estimation, but guard against it.)
        if (found_count < N_ACTIVE_EXPTS) continue;

        // --- Step 1: Backprop through weight normalization ---
        // out_weights[k] = candidate_softmax[selected_idx] / sum(candidate_softmax[selected])
        // where sum is over the 8 selected experts
        float selected_softmax[N_ACTIVE_EXPTS];
        float S = 0.0f;  // sum of selected softmax values
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
            int idx = selected_candidate_idx[k];
            selected_softmax[k] = candidate_softmax[idx];
            S += selected_softmax[k];
        }

        // d_selected_softmax[k] = d_w_k / S - (1/S²) * sum_j(d_w_j * selected_softmax[j])
        //      = d_w_k / S - C / S²   where C is constant for all k
        float C = 0.0f;
        for (int k = 0; k < N_ACTIVE_EXPTS; k++)
            C += d_wts_s[k] * selected_softmax[k];

        float inv_S = 1.0f / (S + 1e-30f);
        float inv_S2 = inv_S * inv_S;

        // Initialize d_candidate_grad to zero (gradient through softmax pre-activation)
        memset(d_candidate_grad, 0, nc * sizeof(float));

        float d_sel_softmax[N_ACTIVE_EXPTS];
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
            d_sel_softmax[k] = d_wts_s[k] * inv_S - C * inv_S2;
            int idx = selected_candidate_idx[k];
            d_candidate_grad[idx] = d_sel_softmax[k];
        }

        // --- Step 2: Backprop through softmax over candidates ---
        // candidate_softmax[i] = softmax(candidate_blended[i])
        // d_candidate_blended[i] = candidate_softmax[i] * (d_candidate_grad[i] - dot)
        // where dot = sum_j(candidate_softmax[j] * d_candidate_grad[j])
        float dot_candidate = 0.0f;
        for (int i = 0; i < nc; i++)
            dot_candidate += candidate_softmax[i] * d_candidate_grad[i];

        for (int i = 0; i < nc; i++)
            d_candidate_grad[i] = candidate_softmax[i] * (d_candidate_grad[i] - dot_candidate);

        // Now d_candidate_grad[i] = d(loss)/d(candidate_blended[i])

        // --- Step 3: Backprop through blend coefficients ---
        // candidate_blended[i] = 0.3 * group_softmax[g_idx] + 0.7 * fine_scores[i]
        // d_group_softmax += 0.3 * sum(d_candidate_grad for candidates in this group)
        // d_fine_scores[i] += 0.7 * d_candidate_grad[i]

        float d_group_softmax[2] = {0.0f, 0.0f};
        // d_fine_score[i] = 0.7 * d_candidate_grad[i]
        // We can reuse d_candidate_grad as d_fine_score (just scale by 0.7)
        for (int i = 0; i < nc; i++) {
            int g_idx = (int)candidate_group_idx_f[i];
            d_group_softmax[g_idx] += 0.3f * d_candidate_grad[i];
            d_candidate_grad[i] *= 0.7f;  // now = d(loss)/d(fine_score[i])
        }

        // --- Step 4: Backprop through fine scores → distance gradients ---
        // fine_scores[i] = -dist / temp
        // d_loss/d(dist_i) = -1/temp * d_loss/d(fine_score_i)
        // Then backprop through Poincaré distance
        memset(d_x_ball_fine_acc, 0, d * sizeof(float));

        for (int i = 0; i < nc; i++) {
            float d_fs = d_candidate_grad[i];  // d(loss)/d(fine_score)
            if (fabsf(d_fs) < 1e-15f) continue;

            int expert_id = candidate_expert_ids[i];
            const float *cent_e = router->fine_centroids + expert_id * d;

            float d_dist = inv_temp * d_fs;  // d(loss)/d(dist)

            float *d_cent_e = d_fine_centroids ? d_fine_centroids + expert_id * d : NULL;
            poincare_dist_backward_one(x_ball_fine, cent_e, d, R_fine,
                                       d_dist,
                                       d_x_ball_fine_acc, d_cent_e,
                                       neg_x, z_tmp, d_z_tmp);
        }

        // Backprop d_x_ball_fine_acc → d_x through exp_map (fine radius)
        if (d_x) {
            exp_map_backward_one(x_s, d, R_fine, d_x_ball_fine_acc, d_x + s * d);
        }

        // --- Step 5: Backprop through coarse softmax ---
        // coarse_scores[g] = softmax pre-activation (now holds softmax values after recompute)
        // Wait — we overwrote coarse_scores with the softmax values.
        // We need the softmax values (which we have) and the softmax gradient.
        //
        // d_coarse_grad[g] = d(loss)/d(coarse_softmax[g])
        // Initialize to zero, then add gradients for the 2 selected groups
        memset(d_coarse_grad, 0, n_groups * sizeof(float));
        for (int g_idx = 0; g_idx < 2; g_idx++) {
            int group = sel_groups[g_idx];
            d_coarse_grad[group] += d_group_softmax[g_idx];
        }

        // Softmax backward:
        // d_coarse_pre[g] = coarse_softmax[g] * (d_coarse_grad[g] - dot_coarse)
        // where dot_coarse = sum_h(coarse_softmax[h] * d_coarse_grad[h])
        float dot_coarse = 0.0f;
        for (int g = 0; g < n_groups; g++)
            dot_coarse += coarse_scores[g] * d_coarse_grad[g];

        // Backprop through coarse scores → distance gradients
        memset(d_x_ball_coarse_acc, 0, d * sizeof(float));

        for (int g = 0; g < n_groups; g++) {
            float d_cs = coarse_scores[g] * (d_coarse_grad[g] - dot_coarse);
            if (fabsf(d_cs) < 1e-15f) continue;

            const float *cent_g = router->coarse_centroids + g * d;

            // coarse_score = -dist / temp
            float d_dist = inv_temp * d_cs;

            float *d_cent_g = d_coarse_centroids ? d_coarse_centroids + g * d : NULL;
            poincare_dist_backward_one(x_ball_coarse, cent_g, d, R_coarse,
                                       d_dist,
                                       d_x_ball_coarse_acc, d_cent_g,
                                       neg_x, z_tmp, d_z_tmp);
        }

        // Backprop d_x_ball_coarse_acc → d_x through exp_map (coarse radius)
        if (d_x) {
            exp_map_backward_one(x_s, d, R_coarse, d_x_ball_coarse_acc, d_x + s * d);
        }
    }

    free(scratch);
}
