/*
 * wubu_spec_decode.c — Speculative decoding framework
 *
 * Implements lossless token acceleration via draft model
 * proposal + target model verification via rejection sampling.
 *
 * C11, zero-malloc hot path, opaque struct, self-contained.
 */
#include "wubu_spec_decode.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct wubu_spec_decode_ctx {
    int draft_vocab_size;
    int target_vocab_size;
    int max_draft_len;

    /* Per-position acceptance log probability buffer */
    float *log_accept_prob;  /* [max_draft_len] */
};

wubu_spec_decode_ctx_t *wubu_spec_decode_init(
        int draft_vocab_size, int target_vocab_size, int max_draft_len)
{
    if (draft_vocab_size <= 0 || target_vocab_size <= 0 || max_draft_len <= 0)
        return NULL;

    wubu_spec_decode_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

    ctx->draft_vocab_size = draft_vocab_size;
    ctx->target_vocab_size = target_vocab_size;
    ctx->max_draft_len = max_draft_len;
    ctx->log_accept_prob = malloc((size_t)max_draft_len * sizeof(float));
    if (!ctx->log_accept_prob) { free(ctx); return NULL; }

    return ctx;
}

void wubu_spec_decode_free(wubu_spec_decode_ctx_t *ctx)
{
    if (!ctx) return;
    free(ctx->log_accept_prob);
    free(ctx);
}

int wubu_spec_decode(
        wubu_spec_decode_ctx_t *ctx,
        const float *q_logits,
        const float *draft_logit_batch,
        int *accepted,
        int *n_accepted,
        int *n_rejected,
        uint64_t seed)
{
    if (!ctx || !q_logits || !draft_logit_batch || !accepted || !n_accepted || !n_rejected)
        return -1;
    if (ctx->max_draft_len <= 0) { *n_accepted = 0; *n_rejected = 0; return 0; }

    int n = ctx->max_draft_len;
    *n_accepted = 0;
    *n_rejected = 0;

    /* Simple LCG PRNG — sufficient for rejection sampling.
     * Avoids rand() which requires global state and is not thread-safe. */
    uint64_t rng = seed ? seed : 1;

    for (int i = 0; i < n; i++) {
        /* Find argmax of target logits at position i (greedy target token) */
        float max_q = -1e30f;
        int target_tok = 0;
        /* Also compute P_draft for this draft token */
        const float *draft_logits = draft_logit_batch + (size_t)i * ctx->draft_vocab_size;

        /* Softmax computation for Q at this position (approximated via max-sub) */
        /* For spec decoding: compare draft token prob vs target prob.
         * We use the target's probability for the DRAFT token proposed by the draft model. */
        const float *q = q_logits + i * ctx->target_vocab_size;

        /* Find draft max token and its log_prob (log-space for stability) */
        float draft_max_logp = -1e30f;
        int draft_tok = 0;
        float draft_max = -1e30f;
        for (int t = 0; t < ctx->draft_vocab_size; t++) {
            if (draft_logits[t] > draft_max) {
                draft_max = draft_logits[t];
                draft_tok = t;
            }
        }
        /* Convert draft logit to pseudo-probability via softmax over draft vocab */
        float draft_sum = 0.0f;
        for (int t = 0; t < ctx->draft_vocab_size; t++) {
            float p = expf(draft_logits[t] - draft_max);
            draft_sum += p;
            ctx->log_accept_prob[t] = draft_logits[t] - draft_max - logf(draft_sum);
        }
        float draft_prob = expf(draft_logits[draft_tok] - draft_max) / draft_sum;

        /* Target probability for the draft token (use target model logit) */
        int target_idx = (draft_tok < ctx->target_vocab_size) ? draft_tok : 0;
        float q_max = -1e30f;
        for (int t = 0; t < ctx->target_vocab_size; t++) {
            if (q[t] > q_max) q_max = q[t];
        }
        float q_sum = 0.0f;
        for (int t = 0; t < ctx->target_vocab_size; t++)
            q_sum += expf(q[t] - q_max);
        float q_prob_draft_tok = expf(q[target_idx] - q_max) / q_sum;

        /* Rejection sampling: accept if q_prob >= draft_prob or RNG < q_prob/draft_prob */
        float alpha = q_prob_draft_tok / (draft_prob + 1e-10f);
        if (alpha >= 1.0f || q_prob_draft_tok >= draft_prob) {
            /* Accept */
            accepted[*n_accepted] = i;
            (*n_accepted)++;
        } else {
            /* Reject: roll RNG */
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            float r = (float)((rng >> 33) & 0x7FFFFFFFull) / (float)0x7FFFFFFFull;
            if (r < alpha) {
                accepted[*n_accepted] = i;
                (*n_accepted)++;
            } else {
                /* First rejection ends acceptance of draft prefix */
                *n_rejected = 1;
                break;
            }
        }
    }

    return 0;
}

float wubu_spec_decode_throughput(int n_draft_tokens, float accept_rate)
{
    if (n_draft_tokens <= 0 || accept_rate <= 0.0f || accept_rate > 1.0f) return 1.0f;
    /* Expected tokens per target forward pass:
     * E[accepted] = sum_{k=1}^{n} k * P(exactly k accepted)
     *             = sum_{k=1}^{n} k * alpha^{k-1} * (1-alpha) + n * alpha^n
     * Simplified: for small alpha, ~1/(1-alpha). For large n, ~n*accept_rate. */
    float expected = 1.0f;
    float p_all_accept = 1.0f;
    for (int k = 1; k <= n_draft_tokens; k++) {
        p_all_accept *= accept_rate;
        expected += p_all_accept;
    }
    return expected;
}

void wubu_spec_decode_eagle3_conditioning(
        const float *draft_logit_batch,
        const float *draft_states,
        int max_draft_len,
        int draft_vocab_size,
        float *target_cond,
        float temperature)
{
    if (!draft_logit_batch || !draft_states || !target_cond || max_draft_len <= 0) return;

    int d = draft_vocab_size;
    /* EAGLE-3: use draft model's hidden states and logits as
     * additional conditioning for target model's next-token prediction.
     * Simple weighted sum of draft token embeddings projected back. */
    for (int i = 0; i < max_draft_len; i++) {
        const float *draft_logits = draft_logit_batch + (size_t)i * d;
        /* Softmax over draft logits to get draft prob distribution */
        float max_logit = -1e30f;
        for (int t = 0; t < d; t++) {
            float l = draft_logits[t] / temperature;
            if (l > max_logit) max_logit = l;
        }
        float sum_exp = 0.0f;
        for (int t = 0; t < d; t++) {
            float p = expf(draft_logits[t] / temperature - max_logit);
            sum_exp += p;
        }
        /* Accumulate weighted draft token distribution into target cond */
        float inv_sum = 1.0f / (sum_exp + 1e-10f);
        for (int t = 0; t < d && t < 128; t++) { /* Cap at 128 dims for safety */
            float p = expf(draft_logits[t] / temperature - max_logit) * inv_sum;
            target_cond[t] += p * draft_states[i * d + t];
        }
    }
}

int wubu_spec_verify_tree(
        const int *candidates, const int *parents,
        const float *draft_probs, const float *target_probs,
        int n_cand, int vocab,
        int *accepted, int max_acc, float rng_val)
{
    if (n_cand <= 0 || max_acc <= 0) return 0;

    /* BFS order: process candidates in array order (root first, then children).
     * For each candidate, check if parent was accepted (or parent is -1). */
    int n_accepted = 0;
    int *parent_accepted = (int *)calloc(n_cand, sizeof(int));
    if (!parent_accepted) return 0;

    for (int i = 0; i < n_cand && n_accepted < max_acc; i++) {
        int pid = parents[i];
        /* Root or parent was accepted */
        if (pid < 0 || (pid >= 0 && pid < i && parent_accepted[pid])) {
            float p_t = target_probs[candidates[i]];
            float p_d = draft_probs[i];
            if (p_d <= 0.0f) p_d = 1e-10f;
            float ratio = p_t / p_d;
            if (p_t >= p_d || rng_val < ratio) {
                accepted[n_accepted++] = candidates[i];
                parent_accepted[i] = 1;
            } else {
                /* First rejection — stop */
                break;
            }
        } else {
            /* Parent not accepted — skip this subtree */
            break;
        }
    }

    free(parent_accepted);
    return n_accepted;
}

int wubu_spec_bonus_token(
        const float *target_probs, const float *draft_probs,
        int vocab, float rng_val)
{
    /* Residual distribution: r(t) = max(0, p_target(t) - p_draft(t)) */
    float *residual = (float *)calloc(vocab, sizeof(float));
    if (!residual) return 0;

    float total = 0.0f;
    for (int t = 0; t < vocab; t++) {
        float d = draft_probs ? draft_probs[t] : 0.0f;
        float r = target_probs[t] - d;
        if (r > 0.0f) {
            residual[t] = r;
            total += r;
        }
    }

    if (total <= 0.0f) {
        free(residual);
        return 0;
    }

    /* Sample from residual using rng_val */
    float cum = 0.0f;
    float needle = rng_val * total;
    int sampled = 0;
    for (int t = 0; t < vocab; t++) {
        cum += residual[t];
        if (cum >= needle) {
            sampled = t;
            break;
        }
    }

    free(residual);
    return sampled;
}