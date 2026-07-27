/* wubu_ssm_chunked.c — Chunked Gated DeltaNet recurrence
 *
 * This is the EXACT same recurrence as the sequential path in wubu_ssm.c,
 * written in matrix (outer-product / rank-1) form:
 *
 *     a_t = exp(clamp(g_t))
 *     h_t = a_t * ( h_{t-1} - beta_t * k_t k_t^T h_{t-1} ) + beta_t * v_t k_t^T
 *         = a_t * h_{t-1} + k_t ( beta_t * v_t - beta_t * k_t^T (a_t h_{t-1}) )^T
 *
 * i.e. the per-token sequential DeltaNet update.  Processing the tokens inside
 * a chunk in the same left-to-right order as the sequential path makes this
 * provably IDENTICAL to scalar (every token sees exactly the same h history),
 * so it cannot diverge — while still being grouped into chunks for the
 * inter-chunk state carry and future large-CS parallel matmul speedups.
 *
 * State h is row-major [d*d], matching the sequential path's ssm_state layout
 * exactly (no transpose on entry/exit).
 */
#include "wubu_ssm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>

// Recurrent-state integrity guard: the Gated DeltaNet state s_t is carried
// across calls in model->ssm_states. For untrained/random SSM weights (or a
// transient gate spike in any model) the decay exp(g_last) can exceed 1 and
// drive s_t to Inf/NaN within a few chunks, permanently poisoning the
// persistent state so every subsequent forward (e.g. decode after a prefill)
// is corrupted. A trained model keeps its state bounded well below this
// threshold, so the clamp is a no-op for real weights but a hard floor
// against corruption. 1e3 is ~6 orders of magnitude above typical SSM state.
#define SSM_STATE_CLAMP 1e3f

static inline void ssm_state_clamp(float *h, int n) {
    for (int i = 0; i < n; i++) {
        float v = h[i];
        if (v > SSM_STATE_CLAMP) h[i] = SSM_STATE_CLAMP;
        else if (v < -SSM_STATE_CLAMP) h[i] = -SSM_STATE_CLAMP;
        else if (!(v == v)) h[i] = 0.0f;          // NaN
        else if (v != 0.0f && v * 0.5f == v) h[i] = 0.0f;  // Inf
    }
}

#define CS 2

void wubu_ssm_chunked_recurrence(
    int B, int T,
    const float *q_norm, const float *k_norm, const float *v_conv,
    const float *beta_flat, const float *gate_flat,
    float *ssm_state, float *delta_out)
{
    if (B != 1) { fprintf(stderr, "chunked: only B=1 supported\n"); return; }
    const int d  = SSM_D_STATE;
    const int hk = SSM_K_HEADS;
    const int hv = SSM_V_HEADS;
    int pad = (CS - T % CS) % CS;
    int nt  = T + pad;
    int nc  = nt / CS;

    size_t sz_t = (size_t)nt * d * sizeof(float);
    float *qp = (float *)calloc(hk, sz_t);
    float *kp = (float *)calloc(hk, sz_t);
    float *vp = (float *)calloc(hv, sz_t);
    float *bp = (float *)calloc(hv, nt * sizeof(float));
    float *gp = (float *)calloc(hv, nt * sizeof(float));
    if (!qp || !kp || !vp || !bp || !gp) goto cleanup;

    for (int h = 0; h < hk; h++)
        for (int t = 0; t < T; t++)
            memcpy(qp + (size_t)h * nt * d + (size_t)t * d,
                   q_norm + (size_t)(t * hk + h) * d,
                   d * sizeof(float));
    for (int h = 0; h < hk; h++)
        for (int t = 0; t < T; t++)
            memcpy(kp + (size_t)h * nt * d + (size_t)t * d,
                   k_norm + (size_t)(t * hk + h) * d,
                   d * sizeof(float));
    for (int h = 0; h < hv; h++) {
        for (int t = 0; t < T; t++)
            memcpy(vp + (size_t)h * nt * d + (size_t)t * d,
                   v_conv + (size_t)(t * hv + h) * d,
                   d * sizeof(float));
        for (int t = 0; t < T; t++) {
            // beta_flat / gate_flat are stored with stride DT_RANK (NOT SSM_V_HEADS)
            bp[(size_t)h * nt + t] = beta_flat[(size_t)(t * DT_RANK + h)];
            gp[(size_t)h * nt + t] = gate_flat[(size_t)(t * DT_RANK + h)];
        }
    }
    memset(delta_out, 0, (size_t)hv * T * d * sizeof(float));

    #pragma omp parallel for if(hv > 1)
    for (int vh = 0; vh < hv; vh++) {
        int kh = vh % SSM_K_HEADS;  // cyclic repeat (matches inline forward)
        float *h = ssm_state + (size_t)vh * d * d;
        const float qsc = 1.0f / sqrtf((float)d);

        float *q_s = qp + (size_t)kh * nt * d;
        float *k_s = kp + (size_t)kh * nt * d;
        float *v_s = vp + (size_t)vh * nt * d;
        float *b_s = bp + (size_t)vh * nt;
        float *g_s = gp + (size_t)vh * nt;

        for (int c = 0; c < nc; c++) {
            int off = c * CS;
            int cur_nt = T - off;
            if (cur_nt > CS) cur_nt = CS;
            if (cur_nt < 0) cur_nt = 0;

            // Process tokens inside the chunk in the EXACT sequential order
            // (same as the scalar path), so the result is identical to scalar.
            for (int i = 0; i < cur_nt; i++) {
                float bi = b_s[off + i];
                float gi = g_s[off + i];
                if (gi > 80.0f) gi = 80.0f;
                float ai = (gi < -80.0f) ? 0.0f : expf(gi);  // matches tgt_safe_expf
                const float *kt = k_s + (size_t)(off + i) * d;
                const float *vt = v_s + (size_t)(off + i) * d;
                const float *qt = q_s + (size_t)(off + i) * d;

                // 1. decay: h = a_t * h
                for (int r = 0; r < d; r++) {
                    float *hrow = h + (size_t)r * d;
                    for (int cc = 0; cc < d; cc++) hrow[cc] *= ai;
                }
                // 2. vo = h @ k   (d-vector)
                float vo[SSM_D_STATE];
                for (int r = 0; r < d; r++) {
                    const float *hrow = h + (size_t)r * d;
                    double s = 0;
                    for (int cc = 0; cc < d; cc++) s += (double)hrow[cc] * (double)kt[cc];
                    vo[r] = (float)s;
                }
                // 3+4. h += k ( beta*v - beta*vo )^T   ==  h + k (vn - vo)^T
                for (int r = 0; r < d; r++) {
                    float *hrow = h + (size_t)r * d;
                    double diff = (double)bi * ((double)vt[r] - (double)vo[r]);
                    for (int cc = 0; cc < d; cc++)
                        hrow[cc] = (float)((double)hrow[cc] + (double)kt[cc] * diff);
                }
                ssm_state_clamp(h, d * d);

                // 5. output o = h @ (q * scale)
                float *out = delta_out + (size_t)(off + i) * hv * d + (size_t)vh * d;
                for (int r = 0; r < d; r++) {
                    const float *hrow = h + (size_t)r * d;
                    double s = 0;
                    for (int cc = 0; cc < d; cc++)
                        s += (double)hrow[cc] * (double)(qt[cc] * qsc);
                    out[r] = (float)s;
                }
            }
        }
    }

cleanup:
    free(qp); free(kp); free(vp); free(bp); free(gp);
}
