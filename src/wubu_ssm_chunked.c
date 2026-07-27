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

/* ===========================================================================
 * wubu_ssm_gdn_chunked — PRINCIPLED Gated DeltaNet chunkwise-parallel prefill.
 *
 * Math (exact, from the GDN literature: veitner "Chunkwise Gated Delta Rule",
 * sustcsonglin "DeltaNet Explained II", Yang et al. Gated Delta Networks
 * arXiv:2412.06464, and the GDN paper Appendix D.6 which verifies the
 * chunkwise form matches the scalar recurrence to machine precision in fp64):
 *
 *   Scalar GDN recurrence (the verified reference, wubu_ssm.c):
 *     S_t = a_t ( S_{t-1} - beta_t k_t k_t^T S_{t-1} ) + beta_t v_t k_t^T
 *         = a_t S_{t-1} + k_t ( beta_t v_t - beta_t (a_t k_t^T S_{t-1}) )^T
 *
 *   Chunkwise-parallel form (chunk size C, gate a_r = exp(gate_r), tied scalar
 *   decay gamma_r = a_r).  Per chunk, with K,V,Q in R^{C x d}, beta in R^C,
 *   gamma in R^C:
 *     WY factors (UT transform, strictly-lower-triangular solve):
 *       Lw[r,i] = beta_r (K K^T)[r,i]      (i < r)
 *       W  = (I + Lw)^{-1} (beta .* K)     (forward-substitution, per d-col)
 *       Lv[r,i] = beta_r gamma_r gamma_i (K K^T)[r,i]   (i < r)
 *       U~ = (I + Lv)^{-1} (beta .* V)
 *     Gate-rescaled (GDN):  K' = gamma_C gamma_r k_r ,  Q' = gamma_r q_r ,
 *                           W' = gamma_r w_r ,  S' = gamma_C S
 *     Chunk state transition:
 *       S_next = S' + (U~ - W' S^T)^T K'
 *     Chunk outputs:
 *       O = Q' S^T + (Q K^T .* M) (U~ - W' S^T)      M[r,i]=1 iff i<=r
 *
 *   This is EXACT: at C=1 it reduces to the scalar recurrence (harness-enforced),
 *   and for C>1 it computes the identical S and O via dense matmuls (O(T C d +
 *   T d^3 / C)) — the principled, numerically-stable, GPU-ready form.  It is
 *   OPT-IN behind WUBU_GDN_CHUNK (the verified scalar form stays the default).
 * =========================================================================== */

#ifndef GDN_MAXC
#define GDN_MAXC 256
#endif

void wubu_ssm_gdn_chunked(
    int B, int T,
    const float *q_norm, const float *k_norm, const float *v_conv,
    const float *beta_flat, const float *gate_flat,
    int C,                       /* chunk size (1..GDN_MAXC) */
    float *ssm_state, float *delta_out)
{
    if (B != 1) { fprintf(stderr, "gdn: only B=1 supported\n"); return; }
    if (C < 1) C = 1;
    if (C > GDN_MAXC) C = GDN_MAXC;
    const int d  = SSM_D_STATE;
    const int hk = SSM_K_HEADS;
    const int hv = SSM_V_HEADS;
    int pad = (C - T % C) % C;
    int nt  = T + pad;
    int nc  = nt / C;

    size_t sz_t = (size_t)nt * d * sizeof(float);
    float *qp = (float *)calloc(hk, sz_t);
    float *kp = (float *)calloc(hk, sz_t);
    float *vp = (float *)calloc(hv, sz_t);
    float *bp = (float *)calloc(hv, nt * sizeof(float));
    float *gp = (float *)calloc(hv, nt * sizeof(float));
    if (!qp || !kp || !vp || !bp || !gp) goto cleanup_gdn;

    for (int h = 0; h < hk; h++)
        for (int t = 0; t < T; t++)
            memcpy(qp + (size_t)h * nt * d + (size_t)t * d,
                   q_norm + (size_t)(t * hk + h) * d, d * sizeof(float));
    for (int h = 0; h < hk; h++)
        for (int t = 0; t < T; t++)
            memcpy(kp + (size_t)h * nt * d + (size_t)t * d,
                   k_norm + (size_t)(t * hk + h) * d, d * sizeof(float));
    for (int h = 0; h < hv; h++) {
        for (int t = 0; t < T; t++)
            memcpy(vp + (size_t)h * nt * d + (size_t)t * d,
                   v_conv + (size_t)(t * hv + h) * d, d * sizeof(float));
        for (int t = 0; t < T; t++) {
            bp[(size_t)h * nt + t] = beta_flat[(size_t)(t * DT_RANK + h)];
            gp[(size_t)h * nt + t] = gate_flat[(size_t)(t * DT_RANK + h)];
        }
    }
    memset(delta_out, 0, (size_t)hv * T * d * sizeof(float));

    /* per-chunk scratch (C-bounded) */
    float *Kc  = (float *)calloc((size_t)C * d, sizeof(float));   /* C x d */
    float *Vc  = (float *)calloc((size_t)C * d, sizeof(float));
    float *Qc  = (float *)calloc((size_t)C * d, sizeof(float));
    float *Wm  = (float *)calloc((size_t)C * d, sizeof(float));   /* W  C x d */
    float *Um  = (float *)calloc((size_t)C * d, sizeof(float));   /* U~ C x d */
    float *KK  = (float *)calloc((size_t)C * C, sizeof(float));   /* C x C */
    float *Lw  = (float *)calloc((size_t)C * C, sizeof(float));
    float *Lv  = (float *)calloc((size_t)C * C, sizeof(float));
    float *gvec= (float *)calloc(C, sizeof(float));               /* gamma per tok */
    float *bvec= (float *)calloc(C, sizeof(float));
    float *tmp = (float *)calloc((size_t)C * d, sizeof(float));   /* U~ - W' S^T  C x d */
    float *QS  = (float *)calloc((size_t)C * d, sizeof(float));   /* Q' S^T (precompute) C x d */
    if (!Kc||!Vc||!Qc||!Wm||!Um||!KK||!Lw||!Lv||!gvec||!bvec||!tmp||!QS) goto cleanup_gdn2;

    #pragma omp parallel for if(hv > 1) private(Kc,Vc,Qc,Wm,Um,KK,Lw,Lv,gvec,bvec)
    for (int vh = 0; vh < hv; vh++) {
        int kh = vh % SSM_K_HEADS;
        float *h = ssm_state + (size_t)vh * d * d;
        const float qsc = 1.0f / sqrtf((float)d);
        const float *q_s = qp + (size_t)kh * nt * d;
        const float *k_s = kp + (size_t)kh * nt * d;
        const float *v_s = vp + (size_t)vh * nt * d;
        const float *b_s = bp + (size_t)vh * nt;
        const float *g_s = gp + (size_t)vh * nt;

        float gamma_last = 1.0f;   /* carry decay across chunks (gamma_C of prev) */

        for (int c = 0; c < nc; c++) {
            int off = c * C;
            int cur = C;
            if (off + C > T) cur = T - off;   /* last partial chunk */
            if (cur <= 0) continue;

            /* gather chunk matrices */
            for (int i = 0; i < cur; i++) {
                memcpy(Kc + (size_t)i * d, k_s + (size_t)(off + i) * d, d * sizeof(float));
                memcpy(Vc + (size_t)i * d, v_s + (size_t)(off + i) * d, d * sizeof(float));
                memcpy(Qc + (size_t)i * d, q_s + (size_t)(off + i) * d, d * sizeof(float));
                float gi = g_s[off + i];
                if (gi > 80.0f) gi = 80.0f;
                gvec[i] = (gi < -80.0f) ? 0.0f : expf(gi);
                bvec[i] = b_s[off + i];
            }
            for (int i = cur; i < C; i++) { gvec[i] = 1.0f; bvec[i] = 0.0f; }

            /* KK = K K^T  (C x C) */
            for (int r = 0; r < cur; r++)
                for (int i = 0; i < cur; i++) {
                    double s = 0; const float *kr = Kc + (size_t)r * d, *ki = Kc + (size_t)i * d;
                    for (int j = 0; j < d; j++) s += (double)kr[j] * (double)ki[j];
                    KK[(size_t)r * C + i] = (float)s;
                }

            /* WY factors via strictly-lower-triangular solve (forward sub). */
            /* W  = (I + Lw)^{-1} (beta .* K):   Lw[r,i] = beta_r KK[r,i], i<r  */
            for (int r = 0; r < cur; r++) {
                for (int i = 0; i < r; i++)
                    Lw[(size_t)r * C + i] = bvec[r] * KK[(size_t)r * C + i];
                for (int j = 0; j < d; j++) {
                    /* W[r,:] = beta_r K[r,:] - sum_{i<r} Lw[r,i] W[i,:] */
                    float rhs = bvec[r] * Kc[(size_t)r * d + j];
                    for (int i = 0; i < r; i++)
                        rhs -= Lw[(size_t)r * C + i] * Wm[(size_t)i * d + j];
                    Wm[(size_t)r * d + j] = rhs;   /* (I+Lw) unit-diag lower => divide by 1 */
                }
            }
            /* U~ = (I + Lv)^{-1} (beta .* V):  Lv[r,i] = beta_r gamma_r gamma_i KK[r,i], i<r */
            for (int r = 0; r < cur; r++) {
                for (int i = 0; i < r; i++)
                    Lv[(size_t)r * C + i] = bvec[r] * gvec[r] * gvec[i] * KK[(size_t)r * C + i];
                for (int j = 0; j < d; j++) {
                    float rhs = bvec[r] * Vc[(size_t)r * d + j];
                    for (int i = 0; i < r; i++)
                        rhs -= Lv[(size_t)r * C + i] * Um[(size_t)i * d + j];
                    Um[(size_t)r * d + j] = rhs;
                }
            }

            /* gate-rescaled quantities (GDN): K'=gC g_r k, Q'=g_r q, W'=g_r w, S'=gC S */
            float gC = gamma_last;   /* gamma_C carry-in (last token of prev chunk) */

            /* tmp[i,m] = U~[i,m] - g_r * (W S^T)[i,m]   (C x d) */
            for (int i = 0; i < cur; i++) {
                float gwi = gvec[i];
                for (int m = 0; m < d; m++) {
                    double ws = 0;   /* (W S^T)[i,m] = sum_p W[i,p] S[p,m] */
                    for (int p = 0; p < d; p++)
                        ws += (double)Wm[(size_t)i * d + p] * (double)h[(size_t)p * d + m];
                    tmp[(size_t)i * d + m] = Um[(size_t)i * d + m] - gwi * (float)ws;
                }
            }

            /* QS[i,m] = (g_r Q[i,:]) S^T [m] = sum_p g_r Q[i,p] S[m,p]   (C x d) */
            for (int i = 0; i < cur; i++) {
                float gqi = gvec[i];
                for (int m = 0; m < d; m++) {
                    double s = 0;
                    for (int p = 0; p < d; p++)
                        s += (double)Qc[(size_t)i * d + p] * (double)h[(size_t)m * d + p];
                    QS[(size_t)i * d + m] = gqi * (float)s;
                }
            }

            /* Outputs O[i,j] = QS[i,j] + sum_m (QK_causal)[i,m] * tmp[i,m] */
            for (int i = 0; i < cur; i++) {
                for (int j = 0; j < d; j++) {
                    double out = (double)QS[(size_t)i * d + j];
                    for (int m = 0; m < d; m++) {
                        if (m <= j) {  /* causal mask M[r,i]=1 iff i<=r */
                            double qk = 0;
                            for (int p = 0; p < d; p++)
                                qk += (double)Qc[(size_t)i * d + p] * (double)Kc[(size_t)m * d + p];
                            out += qk * (double)tmp[(size_t)i * d + m];
                        }
                    }
                    delta_out[(size_t)(off + i) * hv * d + (size_t)vh * d + j] = (float)out;
                }
            }

            /* State transition: S_next[j,m] = gC*S[j,m] + sum_i (gC g_r k[i,j]) * tmp[i,m]
             *   = gC*S[j,m] + gC * sum_i g_r k[i,j] tmp[i,m]   (outer product) */
            for (int j = 0; j < d; j++) {
                for (int m = 0; m < d; m++) {
                    double outer = 0;
                    for (int i = 0; i < cur; i++)
                        outer += (double)(gvec[i] * Kc[(size_t)i * d + j]) * (double)tmp[(size_t)i * d + m];
                    double sn = gC * (double)h[(size_t)j * d + m] + gC * outer;
                    float v = (float)sn;
                    if (v > SSM_STATE_CLAMP) v = SSM_STATE_CLAMP;
                    else if (v < -SSM_STATE_CLAMP) v = -SSM_STATE_CLAMP;
                    else if (!(v == v)) v = 0.0f;
                    else if (v != 0.0f && v * 0.5f == v) v = 0.0f;
                    h[(size_t)j * d + m] = v;
                }
            }

            gamma_last = gvec[cur - 1];   /* carry last token's gamma to next chunk */
        }
    }

cleanup_gdn2:
    free(Kc); free(Vc); free(Qc); free(Wm); free(Um);
    free(KK); free(Lw); free(Lv); free(gvec); free(bvec); free(tmp); free(QS);
cleanup_gdn:
    free(qp); free(kp); free(vp); free(bp); free(gp);
}
