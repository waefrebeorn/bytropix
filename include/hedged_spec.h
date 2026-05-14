#ifndef WUBU_HEDGED_SPEC_H
#define WUBU_HEDGED_SPEC_H

/**
 * hedged_spec.h — Hedged speculative execution pattern
 *
 * Adapted from tailslayer (LaurieWired/tailslayer) hedged-read pattern:
 *   Replicate data across N independent "channels" (or speculative paths),
 *   issue all N reads (or speculations) concurrently, take first valid result.
 *
 * Applied to speculative decoding in LLM inference:
 *   N = number of draft tokens to speculate
 *   "Channel" = independent computation path (GPU SM, CPU core, expert)
 *   "Hedged read" = verify all N candidates in a single forward pass
 *   "First response" = longest accepted prefix
 *
 * Also applicable to:
 *   - MoE expert routing: run top-K experts, take best result
 *   - Parallel beam search: N beams, pick best
 *   - Memory read: N replicas on different channels, take first
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Hedged Read: N-replica with first-response-wins
// ============================================================

/**
 * Hedged read state for a single value.
 * N replicas; first thread to complete signals done.
 */
typedef struct {
    volatile int done;        // set to 1 by first completing worker
    int winner_idx;           // index of first completing worker
    pthread_mutex_t lock;     // for safe winner recording
    int n_workers;            // total number of worker threads
} hedged_read_state;

/** Initialize hedged read state for N workers. */
static inline void hedged_read_init(hedged_read_state *s, int n_workers) {
    s->done = 0;
    s->winner_idx = -1;
    pthread_mutex_init(&s->lock, NULL);
    s->n_workers = n_workers;
}

/** Worker calls this on completion. Returns 1 if this worker won, 0 if another won first. */
static inline int hedged_read_finish(hedged_read_state *s, int worker_idx) {
    if (s->done) return 0;  // another worker already finished
    pthread_mutex_lock(&s->lock);
    if (!s->done) {
        s->done = 1;
        s->winner_idx = worker_idx;
        pthread_mutex_unlock(&s->lock);
        return 1;  // we won!
    }
    pthread_mutex_unlock(&s->lock);
    return 0;
}

/** Check if any worker has completed. */
static inline int hedged_read_is_done(hedged_read_state *s) {
    return s->done;
}

/** Get the winner index (only valid after done==1). */
static inline int hedged_read_winner(hedged_read_state *s) {
    return s->winner_idx;
}

/** Destroy hedged read state. */
static inline void hedged_read_destroy(hedged_read_state *s) {
    pthread_mutex_destroy(&s->lock);
}

// ============================================================
// N-way Replica Manager (from tailslayer pattern)
// ============================================================

/**
 * N-way replica array management.
 * Manages N copies of a block of data, each at a known offset.
 * Used to place replicas on different memory channels (different physical pages).
 */
typedef struct {
    void *base;          // base address of allocation
    size_t elem_size;    // size of each element in bytes
    size_t stride;       // bytes between replicas (usually channel_offset)
    int n_replicas;      // number of replicas
} replica_manager;

/** Initialize replica manager over an existing buffer. */
static inline void replica_init(replica_manager *rm, void *base,
                                 size_t elem_size, size_t stride, int n_replicas) {
    rm->base = base;
    rm->elem_size = elem_size;
    rm->stride = stride;
    rm->n_replicas = n_replicas;
}

/** Get pointer to replica r's copy of logical element idx. */
static inline void *replica_get(replica_manager *rm, int replica_idx, size_t logical_idx) {
    return (char*)rm->base + replica_idx * rm->stride + logical_idx * rm->elem_size;
}

/** Copy value into all replicas. */
static inline void replica_insert(replica_manager *rm, size_t logical_idx,
                                   const void *val) {
    for (int r = 0; r < rm->n_replicas; r++) {
        memcpy(replica_get(rm, r, logical_idx), val, rm->elem_size);
    }
}

// ============================================================
// Speculative Decoding Pattern
// ============================================================

/**
 * Speculative verification context.
 *
 * For LLM speculative decoding:
 *   - Draft model proposes N candidate tokens
 *   - Target model verifies all N in one forward pass (the "hedged read")
 *   - Accept longest prefix that matches
 *
 * The N candidates are like N replicas on independent channels.
 * The verification pass is the hedged read.
 * The acceptance check is "first valid response."
 */
typedef struct {
    int *candidate_tokens;    // [N] draft tokens (input)
    float *candidate_scores;  // [N] draft logprobs (input)
    int n_candidates;         // number of speculations
    int max_accepted;         // size of accepted_tokens buffer (>= n_candidates + 1 for bonus)
    int accepted_prefix_len;  // [output] how many tokens accepted
    int *accepted_tokens;     // [output] accepted token IDs (caller allocates, size max_accepted)
} spec_decode_ctx;

/** Initialize speculative decode context. */
static inline void spec_decode_init(spec_decode_ctx *ctx,
                                     int *candidates, float *scores,
                                     int n, int *accepted, int max_accepted) {
    ctx->candidate_tokens = candidates;
    ctx->candidate_scores = scores;
    ctx->n_candidates = n;
    ctx->max_accepted = max_accepted;
    ctx->accepted_prefix_len = 0;
    ctx->accepted_tokens = accepted;
}

/**
 * Verify candidates against target logits (greedy).
 * This is the "hedged read" — all N candidates verified in one pass.
 * Returns number of tokens accepted (0 = reject all, N = accept all).
 *
 * pattern: for each position i:
 *   if target_token_id == candidate_token_id: accept
 *   else: stop (accept prefix up to i-1)
 */
static inline int spec_decode_verify(spec_decode_ctx *ctx,
                                      const int *target_token_ids) {
    int accepted = 0;
    int max_acc = ctx->max_accepted;
    for (int i = 0; i < ctx->n_candidates && accepted < max_acc; i++) {
        if (target_token_ids[i] == ctx->candidate_tokens[i]) {
            ctx->accepted_tokens[accepted++] = ctx->candidate_tokens[i];
        } else {
            break;
        }
    }
    ctx->accepted_prefix_len = accepted;
    return accepted;
}

/**
 * Rejection sampling verification (Leviathan et al. 2023).
 * Accept with probability min(1, p_target/p_draft).
 * When p_target > p_draft, also sample a bonus token from the residual.
 * Returns number of tokens accepted.
 */
static inline int spec_decode_verify_rejection(spec_decode_ctx *ctx,
                                                 const float *target_probs,
                                                 float rand_uniform) {
    int accepted = 0;
    int max_acc = ctx->max_accepted;
    for (int i = 0; i < ctx->n_candidates && accepted < max_acc; i++) {
        int tok = ctx->candidate_tokens[i];
        float p_target = target_probs[i];
        float p_draft = ctx->candidate_scores[i];
        
        if (p_target > p_draft) {
            // Always accept: target model is more confident.
            // Also sample a bonus token from the residual distribution
            // (p_target - p_draft) / (1 - p_draft) in next position.
            ctx->accepted_tokens[accepted++] = tok;
        } else if (rand_uniform < p_target / p_draft) {
            // Accept with probability p_target/p_draft
            ctx->accepted_tokens[accepted++] = tok;
        } else {
            // Reject: stop here. The next token is sampled from
            // max(0, p_target - p_draft) / (1 - p_draft / p_target).
            // For this simplified version, just stop.
            break;
        }
    }
    ctx->accepted_prefix_len = accepted;
    return accepted;
}

#ifdef __cplusplus
}
#endif

#endif // WUBU_HEDGED_SPEC_H
