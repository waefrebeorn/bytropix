/*
 * wubu_flash_prefill.c -- Fused tiled prefill attention (doc H01).
 *
 * Source: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact
 * Attention with IO-Awareness", NeurIPS 2022.
 *
 * Core idea: During prefill (processing the prompt), the attention
 * computation is compute-bound (not memory-bound like decode). Standard
 * softmax attention materializes the full S×S attention matrix, causing
 * cache thrashing for long prompts. FlashAttention tiles the computation
 * into blocks that fit in L1/L2 cache, computing partial softmax + output
 * contributions per tile and combining them with the online-softmax
 * (running max + log-sum-exp) reduction.
 *
 * Result: O(S) memory instead of O(S²), 2-4× faster prefill for S≥512.
 *
 * This is a CPU implementation using the same tiling strategy:
 *   1. Tile Q, K, V into blocks of size B_tc
 *   2. For each Q tile, iterate over K/V tiles
 *   3. Compute local QK^T, local softmax (with running max/sumexp),
 *      accumulate weighted V output
 *   4. Final reduction combines per-tile outputs with the LSE correction
 *
 * Self-contained C11, no third-party deps.
 */

#include "wubu_flash_prefill.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Fused prefill attention with online softmax (FlashAttention algorithm).
 *
 * Q: [n_heads, seq_len, head_dim] row-major
 * K: [n_heads, seq_len, head_dim] row-major
 * V: [n_heads, seq_len, head_dim] row-major
 * out: [n_heads, seq_len, head_dim] row-major
 *
 * Computes: out[h, i, :] = softmax_j(Q[h,i,:] · K[h,j,:]^T / sqrt(d)) · V[h,j,:]
 *
 * Tile size B_tc controls the memory/compute tradeoff.
 */
void wubu_flash_prefill_attn(const float *Q, const float *K, const float *V,
                               float *out, int n_heads, int seq_len, int head_dim,
                               int B_tc) {
    if (!Q || !K || !V || !out || n_heads <= 0 || seq_len <= 0 || head_dim <= 0)
        return;
    if (B_tc <= 0) B_tc = 64;  /* default tile size */

    float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    /* Per-Q-tile working buffers (allocated once) */
    float *s_ij = (float *)malloc(B_tc * B_tc * sizeof(float));   /* local scores */
    float *o_acc = (float *)malloc(B_tc * head_dim * sizeof(float)); /* output accumulator */
    float *o_tmp = (float *)malloc(B_tc * head_dim * sizeof(float));  /* temp output */
    float *m_prev = (float *)malloc(B_tc * sizeof(float));  /* running max */
    float *l_prev = (float *)malloc(B_tc * sizeof(float));  /* running sumexp */

    if (!s_ij || !o_acc || !o_tmp || !m_prev || !l_prev) {
        free(s_ij); free(o_acc); free(o_tmp); free(m_prev); free(l_prev);
        return;
    }

    for (int h = 0; h < n_heads; h++) {
        const float *Qh = Q + (size_t)h * seq_len * head_dim;
        const float *Kh = K + (size_t)h * seq_len * head_dim;
        const float *Vh = V + (size_t)h * seq_len * head_dim;
        float *Outh = out + (size_t)h * seq_len * head_dim;

        /* Iterate over Q tiles */
        for (int i0 = 0; i0 < seq_len; i0 += B_tc) {
            int Bi = (i0 + B_tc <= seq_len) ? B_tc : (seq_len - i0);

            /* Initialize accumulators for this Q tile */
            memset(o_acc, 0, Bi * head_dim * sizeof(float));
            for (int r = 0; r < Bi; r++) {
                m_prev[r] = -INFINITY;
                l_prev[r] = 0.0f;
            }

            /* Iterate over K/V tiles */
            for (int j0 = 0; j0 < seq_len; j0 += B_tc) {
                int Bj = (j0 + B_tc <= seq_len) ? B_tc : (seq_len - j0);

                /* 1. Compute local scores s_ij[q, k] = Q[i0+q] · K[j0+k]^T / sqrt(d) */
                for (int q = 0; q < Bi; q++) {
                    const float *Qq = Qh + (size_t)(i0 + q) * head_dim;
                    for (int k = 0; k < Bj; k++) {
                        const float *Kk = Kh + (size_t)(j0 + k) * head_dim;
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            dot += Qq[d] * Kk[d];
                        }
                        s_ij[q * Bj + k] = dot * inv_sqrt_d;
                    }
                }

                /* 2. Online softmax: update running max and sumexp */
                float m_new[64], l_new[64], exp_diff[64];
                for (int q = 0; q < Bi; q++) {
                    /* Find max of current tile */
                    float m_tile = -INFINITY;
                    for (int k = 0; k < Bj; k++) {
                        if (s_ij[q * Bj + k] > m_tile)
                            m_tile = s_ij[q * Bj + k];
                    }
                    m_new[q] = fmaxf(m_prev[q], m_tile);
                    exp_diff[q] = expf(m_prev[q] - m_new[q]);
                    l_new[q] = l_prev[q] * exp_diff[q];

                    /* Add exp(s - m_new) to running sumexp */
                    float l_tile = 0.0f;
                    for (int k = 0; k < Bj; k++) {
                        float e = expf(s_ij[q * Bj + k] - m_new[q]);
                        s_ij[q * Bj + k] = e;  /* reuse s_ij for exp values */
                        l_tile += e;
                    }
                    l_new[q] += l_tile;
                }

                /* 3. Accumulate weighted V output */
                for (int q = 0; q < Bi; q++) {
                    float *oq = o_acc + q * head_dim;
                    /* Rescale existing accumulator */
                    float scale = exp_diff[q];
                    for (int d = 0; d < head_dim; d++) {
                        oq[d] *= scale;
                    }
                    /* Add new tile's weighted V contribution */
                    for (int k = 0; k < Bj; k++) {
                        float w = s_ij[q * Bj + k];  /* already exp'd */
                        const float *Vk = Vh + (size_t)(j0 + k) * head_dim;
                        for (int d = 0; d < head_dim; d++) {
                            oq[d] += w * Vk[d];
                        }
                    }
                }

                /* Update running state */
                for (int q = 0; q < Bi; q++) {
                    m_prev[q] = m_new[q];
                    l_prev[q] = l_new[q];
                }
            }

            /* 4. Normalize by sumexp and write output */
            for (int q = 0; q < Bi; q++) {
                float *oq = o_acc + (size_t)q * head_dim;
                float *out_q = Outh + (size_t)(i0 + q) * head_dim;
                float inv_l = (l_prev[q] > 0.0f) ? 1.0f / l_prev[q] : 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    out_q[d] = oq[d] * inv_l;
                }
            }
        }
    }

    free(s_ij); free(o_acc); free(o_tmp); free(m_prev); free(l_prev);
}
