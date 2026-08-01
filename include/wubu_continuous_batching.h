#ifndef WUBU_CONTINUOUS_BATCHING_H
#define WUBU_CONTINUOUS_BATCHING_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
typedef struct wubu_paged_kv wubu_paged_kv_t;
typedef struct wubu_prefix_cache wubu_prefix_cache_t;

#define WUBU_CONT_MAX_SEQ 256
#define WUBU_CONT_MAX_TOKENS 4096

typedef struct {
    int alive;
    int seq_id;
    int n_tokens;
    int *tokens;
    int prefill_done;
    int tokens_generated;
    int *block_ids;
    int n_blocks;
} wubu_seq_state_t;

typedef struct {
    wubu_paged_kv_t *paged_kv;
    wubu_prefix_cache_t *prefix_cache;
    wubu_seq_state_t seqs[WUBU_CONT_MAX_SEQ];
    int max_seq;
    int max_tokens_per_seq;
    int block_size;
    int head_dim;
    int n_kv_heads;
    int n_active;
    uint64_t iteration;
} wubu_cont_batch_t;

typedef struct {
    int seq_idx;
    int is_prefill;
    int n_new_tokens;
    int prefix_matched;
} wubu_sched_item_t;

/* Batch lifecycle */
wubu_cont_batch_t *wubu_cont_batch_create(int block_size, int n_blocks,
                                          int head_dim, int n_kv_heads,
                                          int max_seq, int max_tokens);
void wubu_cont_batch_free(wubu_cont_batch_t *cb);

/* Sequence admission */
int wubu_cont_batch_add_seq(wubu_cont_batch_t *cb, const int *tokens, int n_tokens);
void wubu_cont_batch_remove_seq(wubu_cont_batch_t *cb, int seq_idx);

/* Per-iteration scheduling */
int wubu_cont_batch_schedule(wubu_cont_batch_t *cb, wubu_sched_item_t *out, int max_items);
void wubu_cont_batch_prefill_done(wubu_cont_batch_t *cb, int seq_idx);
void wubu_cont_batch_record_token(wubu_cont_batch_t *cb, int seq_idx, int token_id);

/* D01+D04: overlap prefill with decode — run up to max_prefill_tokens
 * of prefill work per iteration while also decoding 1 token for each
 * active decode sequence. Returns the number of prefill tokens consumed
 * this round (0 if no prefill scheduled). */
int wubu_cont_batch_overlap(wubu_cont_batch_t *cb, wubu_sched_item_t *out,
                            int max_items, int max_prefill_tokens);

/* D03: disaggregated prefill/decode — two separate passes sharing one KV store.
 * Pass 1 (prefill engine): consume up to max_prefill_tokens of prefill work for
 *   new sequences. Pass 2 (decode engine): decode 1 token for every active
 *   decode sequence. This is the PD-disaggregation pattern (separate prefill
 *   and decode "instances") applied on a single host. Returns total items. */
int wubu_cont_batch_disagg(wubu_cont_batch_t *cb, wubu_sched_item_t *out,
                           int max_items, int max_prefill_tokens, int *n_prefill_out);

/* Stats */
void wubu_cont_batch_stats(const wubu_cont_batch_t *cb,
                           int *active, int *total_tokens, int *kv_blocks_used,
                           int *kv_blocks_free, size_t *prefix_hits, size_t *prefix_misses);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_CONTINUOUS_BATCHING_H */