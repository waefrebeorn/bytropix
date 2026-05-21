/* wubu_ssm_chunked.c — Chunked Gated DeltaNet recurrence
 *
 * Matches llama.cpp delta-net-base.cpp build_delta_net_chunking() exactly.
 *
 * Per chunk [CS=64 tokens]:
 *   1. Build decay mask, KB, KQ matrices
 *   2. Solve (I+L)^T X = -L to get attention matrix A = I+X
 *   3. intra = A^T @ v_b        [d, CS]
 *   4. kbd = (k_b*exp(G))^T @ A [d, CS]
 *   5. v_prime = k_cd^T @ s_t   [CS, d]
 *   6. v_new = v_t - v_prime
 *   7. v_attn = v_new^T @ kq + s_t^T @ q_g [d, CS]
 *   8. state update: s_t *= exp(g_last) + kg^T @ v_new
 */
#include "wubu_ssm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>

#define CS 64

static void transpose_mat(int n, const float *src, float *dst)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            dst[j * n + i] = src[i * n + j];
}

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
    const int rf = hv / hk;
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
            bp[(size_t)h * nt + t] = beta_flat[(size_t)(t * hv + h)];
            gp[(size_t)h * nt + t] = gate_flat[(size_t)(t * hv + h)];
        }
    }
    memset(delta_out, 0, (size_t)hv * T * d * sizeof(float));

    #pragma omp parallel for if(hv > 1)
    for (int vh = 0; vh < hv; vh++) {
        int kh = vh / rf;
        float *h = ssm_state + (size_t)vh * d * d;
        size_t sz_cs = (size_t)CS * d;
        size_t sz_cs2 = (size_t)CS * CS;
        size_t sz_dd = (size_t)d * d;

        size_t alloc_sz = 13 * sz_cs + 5 * sz_cs2 + 3 * sz_dd + (size_t)CS;
        float *scr = (float *)malloc(alloc_sz * sizeof(float));
        if (!scr) continue;
        float *qc = scr, *kc = qc + sz_cs, *vc = kc + sz_cs;
        float *v_b = vc + sz_cs, *k_b = v_b + sz_cs;
        float *kbd = k_b + sz_cs;
        float *qg = kbd + sz_cs;
        float *kg = qg + sz_cs;
        float *intra = kg + sz_cs;
        float *v_t = intra + sz_cs;
        float *v_prime = v_t + sz_cs;
        float *v_new = v_prime + sz_cs;
        float *v_attn = v_new + sz_cs;
        float *kb = v_attn + sz_cs;
        float *kq = kb + sz_cs2;
        float *mask = kq + sz_cs2;
        float *lhs = mask + sz_cs2;
        float *attn = lhs + sz_cs2;
        float *s_t = attn + sz_cs2;
        float *kcd = s_t + sz_dd;
        float *kgv = kcd + sz_dd;
        float *g_cs = kgv + sz_dd;

        transpose_mat(d, h, s_t);

        for (int c = 0; c < nc; c++) {
            int off = c * CS;
            int cur_nt = nt - off;
            if (cur_nt > CS) cur_nt = CS;

            float *q_s = qp + (size_t)kh * nt * d + (size_t)off * d;
            float *k_s = kp + (size_t)kh * nt * d + (size_t)off * d;
            float *v_s = vp + (size_t)vh * nt * d + (size_t)off * d;
            float *b_s = bp + (size_t)vh * nt + off;
            float *g_s = gp + (size_t)vh * nt + off;

            // Gather chunk, q pre-scaled
            float qsc = 1.0f / sqrtf((float)d);
            for (int i = 0; i < cur_nt; i++) {
                for (int j = 0; j < d; j++) {
                    qc[(size_t)i * d + j] = q_s[(size_t)i * d + j] * qsc;
                    kc[(size_t)i * d + j] = k_s[(size_t)i * d + j];
                    vc[(size_t)i * d + j] = v_s[(size_t)i * d + j];
                }
            }
            for (int i = cur_nt; i < CS; i++)
                memset(qc + (size_t)i * d, 0, d * sizeof(float));
            for (int i = cur_nt; i < CS; i++)
                memset(kc + (size_t)i * d, 0, d * sizeof(float));
            for (int i = cur_nt; i < CS; i++)
                memset(vc + (size_t)i * d, 0, d * sizeof(float));

            // v_b = v * beta, k_b = k * beta
            for (int i = 0; i < CS; i++) {
                float bi = (i < cur_nt) ? b_s[i] : 0.0f;
                for (int j = 0; j < d; j++) {
                    v_b[(size_t)i * d + j] = vc[(size_t)i * d + j] * bi;
                    k_b[(size_t)i * d + j] = kc[(size_t)i * d + j] * bi;
                }
            }

            // Cumsum gate
            g_cs[0] = (off < T) ? g_s[0] : 0.0f;
            for (int i = 1; i < CS; i++)
                g_cs[i] = g_cs[i-1] + ((off + i < T) ? g_s[i] : 0.0f);
            float g_last = g_cs[cur_nt - 1];
            for (int i = cur_nt; i < CS; i++) g_cs[i] = g_last;

            // Decay mask M[i][j] = exp(G[j]-G[i]) for i >= j (lower tri, causal), else 0
            for (int i = 0; i < CS; i++)
                for (int j = 0; j < CS; j++)
                    mask[(size_t)i * CS + j] = (i >= j)
                        ? expf(fminf(g_cs[j] - g_cs[i], 80.0f)) : 0.0f;

            // KB = mask ⊙ (k^T @ k_b)
            for (int i = 0; i < CS; i++)
                for (int j = 0; j < CS; j++) {
                    float s = 0;
                    for (int z = 0; z < d; z++)
                        s += kc[(size_t)i * d + z] * k_b[(size_t)j * d + z];
                    kb[(size_t)i * CS + j] = s * mask[(size_t)i * CS + j];
                }

            // KQ = mask ⊙ (k^T @ q) — lower including diagonal (LOWER_DIAG)
            for (int i = 0; i < CS; i++)
                for (int j = 0; j < CS; j++) {
                    float s = 0;
                    for (int z = 0; z < d; z++)
                        s += kc[(size_t)i * d + z] * qc[(size_t)j * d + z];
                    kq[(size_t)i * CS + j] = s * mask[(size_t)i * CS + j];
                }
            // Zero strictly upper (j > i), keeping i >= j
            for (int i = 0; i < CS; i++)
                for (int j = i + 1; j < CS; j++)
                    kq[(size_t)i * CS + j] = 0.0f;

            // L = tri(KB, strict_lower); lhs = I + L
            for (int i = 0; i < CS; i++)
                for (int j = i; j < CS; j++)
                    kb[(size_t)i * CS + j] = 0.0f;
            for (int i = 0; i < CS; i++) {
                memcpy(lhs + (size_t)i * CS, kb + (size_t)i * CS, CS * sizeof(float));
                lhs[(size_t)i * CS + i] = 1.0f;
            }

            // Solve L^T @ X = -L  (L unit lower, L^T unit upper)
            // Bottom-up forward sub for L^T upper triangular
            for (int j = 0; j < CS; j++) {
                for (int i = CS - 1; i >= 0; i--) {
                    float b = (i > j) ? -kb[(size_t)i * CS + j] : 0.0f;
                    float s = b;
                    for (int k = i + 1; k < CS; k++)
                        s -= lhs[(size_t)k * CS + i] * attn[(size_t)k * CS + j];
                    attn[(size_t)i * CS + j] = s;
                }
            }
            // A = I + X
            for (int i = 0; i < CS; i++)
                attn[(size_t)i * CS + i] += 1.0f;

            // --- intra = A^T @ v_b [d, CS] ---
            for (int dim = 0; dim < d; dim++)
                for (int t = 0; t < CS; t++) {
                    float s = 0;
                    for (int z = 0; z < CS; z++)
                        s += attn[(size_t)z * CS + t] * v_b[(size_t)z * d + dim];
                    intra[(size_t)dim * CS + t] = s;
                }

            // v_t = intra^T [CS, d]
            for (int t = 0; t < CS; t++)
                for (int dim = 0; dim < d; dim++)
                    v_t[(size_t)t * d + dim] = intra[(size_t)dim * CS + t];

            // --- kbd = (k_b * exp(G))^T @ A [d, CS] ---
            for (int dim = 0; dim < d; dim++)
                for (int t = 0; t < CS; t++) {
                    float s = 0;
                    for (int z = 0; z < CS; z++) {
                        float kbg_z = k_b[(size_t)z * d + dim]
                            * expf(fminf(g_cs[z], 80.0f));
                        s += kbg_z * attn[(size_t)z * CS + t];
                    }
                    kbd[(size_t)dim * CS + t] = s;
                }

            // --- q_g = q * exp(G) [CS, d] ---
            for (int t = 0; t < CS; t++) {
                float ge = expf(fminf(g_cs[t], 80.0f));
                for (int dim = 0; dim < d; dim++)
                    qg[(size_t)t * d + dim] = qc[(size_t)t * d + dim] * ge;
            }

            // --- kg = k * exp(g_last - G) [CS, d] ---
            for (int t = 0; t < CS; t++) {
                float gd = expf(fminf(g_last - g_cs[t], 80.0f));
                for (int dim = 0; dim < d; dim++)
                    kg[(size_t)t * d + dim] = kc[(size_t)t * d + dim] * gd;
            }

            // --- v_prime = k_cd^T @ s_t [CS, d] ---
            for (int t = 0; t < CS; t++)
                for (int dim = 0; dim < d; dim++) {
                    float s = 0;
                    for (int z = 0; z < d; z++)
                        s += kbd[(size_t)z * CS + t] * s_t[(size_t)z * d + dim];
                    v_prime[(size_t)t * d + dim] = s;
                }

            // --- v_new = v_t - v_prime [CS, d] ---
            for (int t = 0; t < CS; t++)
                for (int dim = 0; dim < d; dim++)
                    v_new[(size_t)t * d + dim] = v_t[(size_t)t * d + dim] - v_prime[(size_t)t * d + dim];

            // --- v_attn = v_new^T @ kq + s_t^T @ q_g [d, CS] ---
            for (int dim = 0; dim < d; dim++)
                for (int t = 0; t < CS; t++) {
                    float s = 0;
                    for (int z = 0; z < CS; z++)
                        s += v_new[(size_t)z * d + dim] * kq[(size_t)z * CS + t];
                    for (int z = 0; z < d; z++)
                        s += s_t[(size_t)z * d + dim] * qg[(size_t)t * d + z];
                    v_attn[(size_t)dim * CS + t] = s;
                }

            // Write output for real tokens
            for (int i = 0; i < cur_nt; i++) {
                size_t out_off = (size_t)(off + i) * hv * d + (size_t)vh * d;
                for (int j = 0; j < d; j++)
                    delta_out[out_off + j] = v_attn[(size_t)j * CS + i];
            }

            // --- State update: s_t = s_t * exp(g_last) + kg^T @ v_new ---
            for (int dr = 0; dr < d; dr++)
                for (int dc = 0; dc < d; dc++) {
                    float s = 0;
                    for (int t = 0; t < CS; t++)
                        s += kg[(size_t)t * d + dr] * v_new[(size_t)t * d + dc];
                    kgv[(size_t)dr * d + dc] = s;
                }

            float gl_exp = expf(fminf(g_last, 80.0f));
            for (int r = 0; r < d; r++)
                for (int c = 0; c < d; c++)
                    s_t[(size_t)r * d + c] = s_t[(size_t)r * d + c] * gl_exp + kgv[(size_t)r * d + c];
        }

        transpose_mat(d, s_t, h);
        free(scr);
    }

cleanup:
    free(qp); free(kp); free(vp); free(bp); free(gp);
}
