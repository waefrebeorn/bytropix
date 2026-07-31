/*
 * wubu_expert_choice.c -- Fine-grained MoE expert-choice routing (doc E05).
 *
 * Source: Zhou et al., "Brainformers: Hybrid Sparse Attention and
 * Mixture-of-Experts", Google, 2024; Switch Transformer top-1 routing;
 * Expert Choice (Zhou et al., 2022).
 *
 * Core idea: Standard top-k routing picks the k best experts per token.
 * Expert Choice *inverts* the routing: each expert picks the top-k tokens
 * it wants to process, balancing load across experts automatically without
 * auxiliary loss. This eliminates expert collapse and achieves better
 * specialization with the same compute budget.
 *
 * For our CPU engine: we have n_experts (from the model config) and
 * n_tokens. The router produces a [n_tokens, n_experts] score matrix.
 * Standard routing: for each token, pick top-k experts.
 * Expert choice: for each expert, pick top-k tokens.
 *
 * Self-contained C11, no third-party deps.
 */

#include "wubu_expert_choice.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Standard top-k routing: for each token, pick the k best experts.
 * scores: [n_tokens, n_experts] row-major
 * out_assignments: [n_tokens * k] — top-k expert IDs per token
 * out_weights: [n_tokens * k] — softmax weights for selected experts */
void wubu_topk_route(const float *scores, int n_tokens, int n_experts, int k,
                     int *out_assignments, float *out_weights) {
    if (!scores || !out_assignments || !out_weights || n_tokens <= 0 || n_experts <= 0 || k <= 0)
        return;

    for (int t = 0; t < n_tokens; t++) {
        const float *row = scores + (size_t)t * n_experts;

        /* Find top-k experts by score (simple selection for small k) */
        int *selected = out_assignments + t * k;
        float *weights = out_weights + t * k;

        /* Copy scores and find top-k indices */
        float *tmp_scores = (float *)malloc(n_experts * sizeof(float));
        int *tmp_idx = (int *)malloc(n_experts * sizeof(int));
        for (int i = 0; i < n_experts; i++) {
            tmp_scores[i] = row[i];
            tmp_idx[i] = i;
        }
        /* Partial sort: find top-k */
        for (int i = 0; i < k; i++) {
            int best = i;
            for (int j = i + 1; j < n_experts; j++) {
                if (tmp_scores[j] > tmp_scores[best]) best = j;
            }
            /* Swap */
            float ts = tmp_scores[i]; tmp_scores[i] = tmp_scores[best]; tmp_scores[best] = ts;
            int ti = tmp_idx[i]; tmp_idx[i] = tmp_idx[best]; tmp_idx[best] = ti;
        }
        /* Copy top-k */
        for (int i = 0; i < k; i++) {
            selected[i] = tmp_idx[i];
        }

        /* Softmax over selected experts */
        float max_s = tmp_scores[0];
        float sum_exp = 0.0f;
        for (int i = 0; i < k; i++) {
            float e = expf(tmp_scores[i] - max_s);
            weights[i] = e;
            sum_exp += e;
        }
        float inv = (sum_exp > 1e-10f) ? 1.0f / sum_exp : 0.0f;
        for (int i = 0; i < k; i++) weights[i] *= inv;

        free(tmp_scores);
        free(tmp_idx);
    }
}

/* Expert Choice routing: for each expert, pick the top-k tokens.
 * scores: [n_tokens, n_experts] row-major
 * out_assignments: [n_experts * k] — top-k token IDs per expert
 * out_weights: [n_experts * k] — routing weights
 *
 * This produces balanced load (each expert handles exactly k tokens)
 * and better specialization than standard routing. */
void wubu_expert_choice_route(const float *scores, int n_tokens, int n_experts, int k,
                                int *out_assignments, float *out_weights) {
    if (!scores || !out_assignments || !out_weights || n_tokens <= 0 || n_experts <= 0 || k <= 0)
        return;

    /* tokens_per_expert = ceil(n_tokens * k / n_experts) */
    int k_per_expert = k;
    /* Ensure each expert doesn't take more than n_tokens */
    if (k_per_expert > n_tokens) k_per_expert = n_tokens;

    for (int e = 0; e < n_experts; e++) {
        /* Extract this expert's scores across all tokens: scores[t, e] */
        float *expert_scores = (float *)malloc(n_tokens * sizeof(float));
        int *token_idx = (int *)malloc(n_tokens * sizeof(int));
        for (int t = 0; t < n_tokens; t++) {
            expert_scores[t] = scores[(size_t)t * n_experts + e];
            token_idx[t] = t;
        }

        /* Find top-k tokens for this expert */
        for (int i = 0; i < k_per_expert; i++) {
            int best = i;
            for (int j = i + 1; j < n_tokens; j++) {
                if (expert_scores[j] > expert_scores[best]) best = j;
            }
            float ts = expert_scores[i]; expert_scores[i] = expert_scores[best]; expert_scores[best] = ts;
            int ti = token_idx[i]; token_idx[i] = token_idx[best]; token_idx[best] = ti;

            out_assignments[e * k + i] = token_idx[i];
        }

        /* Compute softmax weights for selected tokens */
        float max_s = expert_scores[0];
        float sum_exp = 0.0f;
        for (int i = 0; i < k_per_expert; i++) {
            float ev = expf(expert_scores[i] - max_s);
            out_weights[e * k + i] = ev;
            sum_exp += ev;
        }
        float inv = (sum_exp > 1e-10f) ? 1.0f / sum_exp : 0.0f;
        for (int i = 0; i < k_per_expert; i++)
            out_weights[e * k + i] *= inv;

        free(expert_scores);
        free(token_idx);
    }
}

/* Compute load balance across experts.
 * Returns the coefficient of variation (std/mean) of per-expert token counts.
 * Lower = more balanced. */
float wubu_route_load_balance(const int *assignments, int n_experts, int k, int n_tokens) {
    if (!assignments || n_experts <= 0 || k <= 0 || n_tokens <= 0) return -1.0f;

    /* Count how many tokens each expert processes */
    int *counts = (int *)calloc(n_experts, sizeof(int));
    for (int e = 0; e < n_experts; e++) {
        for (int i = 0; i < k; i++) {
            int tok = assignments[e * k + i];
            if (tok >= 0 && tok < n_tokens) counts[tok]++;
        }
    }

    /* Compute mean and std of token-to-expert assignments */
    float mean = (float)n_tokens / (float)n_experts;
    float var = 0.0f;
    for (int i = 0; i < n_experts; i++) {
        float d = (float)counts[i] - mean;
        var += d * d;
    }
    var /= (float)n_experts;
    float std = sqrtf(var);
    float cv = (mean > 1e-10f) ? std / mean : 0.0f;

    free(counts);
    return cv;
}
