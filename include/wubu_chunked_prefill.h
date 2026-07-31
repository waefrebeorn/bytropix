/*
 * wubu_chunked_prefill.h -- Chunked prefill + disaggregated PD (doc D03/D04).
 *
 * Splits long prompts into chunks interleaved with decode steps,
 * preventing a long prompt from blocking ongoing decodes.
 *
 * Self-contained C11, no third-party deps.
 */

#ifndef WUBU_CHUNKED_PREFILL_H
#define WUBU_CHUNKED_PREFILL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_MAX_PREFILL_JOBS 64

typedef enum {
    WUBU_PREFILL_PENDING = 0,
    WUBU_PREFILL_DONE    = 1
} wubu_prefill_state_t;

typedef struct {
    int total_tokens;
    int prefilled;
    wubu_prefill_state_t state;
} wubu_prefill_job_t;

typedef struct {
    int max_chunk_size;
    wubu_prefill_job_t jobs[WUBU_MAX_PREFILL_JOBS];
    int n_pending;
} wubu_chunked_prefill_t;

/* Create/destroy. */
wubu_chunked_prefill_t *wubu_chunked_prefill_create(int max_chunk_size);
void wubu_chunked_prefill_free(wubu_chunked_prefill_t *c);

/* Submit a prompt for chunked prefill. Returns job ID or -1. */
int wubu_chunked_prefill_submit(wubu_chunked_prefill_t *c, int n_tokens);

/* Get next chunk size for a job. Returns chunk size, 0 if done, -1 on error. */
int wubu_chunked_prefill_next_chunk(wubu_chunked_prefill_t *c, int job_id);

/* Check if a job is complete. */
bool wubu_chunked_prefill_is_done(wubu_chunked_prefill_t *c, int job_id);

/* Get progress fraction for a job. */
float wubu_chunked_prefill_progress(wubu_chunked_prefill_t *c, int job_id);

/* Schedule prefill chunks + decode tokens for this step. */
int wubu_chunked_prefill_schedule(wubu_chunked_prefill_t *c,
                                    int n_decode_budget,
                                    int *out_chunks, int *out_decode);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_CHUNKED_PREFILL_H */
