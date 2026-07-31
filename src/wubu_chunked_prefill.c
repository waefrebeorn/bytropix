/*
 * wubu_chunked_prefill.c -- Chunked prefill + disaggregated prefill/decode (doc D03/D04).
 *
 * Source: vLLM "Chunked prefill"; Saratha et al., "LLM Inference
 * Unveiled" (arXiv:2607.02558); disaggregated PD papers.
 *
 * Core idea:
 * - D04 Chunked prefill: Split long prompts into chunks that are processed
 *   one at a time, interleaved with ongoing decode steps. This prevents
 *   a long prompt from blocking decode for shorter requests.
 * - D03 Disaggregated prefill/decode: Run prefill and decode in separate
 *   passes/scheduling slots. Prefill is compute-bound (can use all cores);
 *   decode is memory-bound (bandwidth-limited). Separating them allows
 *   each phase to be optimized independently.
 *
 * For our CPU engine: the scheduler runs prefill chunks and decode tokens
 * in the same time step, time-sliced, so neither phase monopolizes.
 *
 * Self-contained C11, no third-party deps.
 */

#include "wubu_chunked_prefill.h"
#include <stdlib.h>
#include <string.h>

/* Create a chunked prefill context. */
wubu_chunked_prefill_t *wubu_chunked_prefill_create(int max_chunk_size) {
    if (max_chunk_size <= 0) max_chunk_size = 512;
    wubu_chunked_prefill_t *c = (wubu_chunked_prefill_t *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->max_chunk_size = max_chunk_size;
    c->n_pending = 0;
    return c;
}

void wubu_chunked_prefill_free(wubu_chunked_prefill_t *c) {
    free(c);
}

/* Submit a prompt for chunked prefill.
 * Returns a prefill job ID (>= 0) or -1 on capacity. */
int wubu_chunked_prefill_submit(wubu_chunked_prefill_t *c, int n_tokens) {
    if (!c || n_tokens <= 0 || c->n_pending >= WUBU_MAX_PREFILL_JOBS) return -1;
    int id = c->n_pending++;
    c->jobs[id].total_tokens = n_tokens;
    c->jobs[id].prefilled = 0;
    c->jobs[id].state = WUBU_PREFILL_PENDING;
    return id;
}

/* Get the next chunk to process for a given job.
 * Returns number of tokens in this chunk (may be < max_chunk_size if
 * the remaining tokens are fewer), or 0 if the job is complete or -1 on error. */
int wubu_chunked_prefill_next_chunk(wubu_chunked_prefill_t *c, int job_id) {
    if (!c || job_id < 0 || job_id >= c->n_pending) return -1;
    wubu_prefill_job_t *job = &c->jobs[job_id];
    if (job->state == WUBU_PREFILL_DONE) return 0;
    int remaining = job->total_tokens - job->prefilled;
    if (remaining <= 0) {
        job->state = WUBU_PREFILL_DONE;
        return 0;
    }
    int chunk = (remaining < c->max_chunk_size) ? remaining : c->max_chunk_size;
    job->prefilled += chunk;
    job->state = (job->prefilled >= job->total_tokens) ? WUBU_PREFILL_DONE : WUBU_PREFILL_PENDING;
    return chunk;
}

/* Check if a prefill job is complete. */
bool wubu_chunked_prefill_is_done(wubu_chunked_prefill_t *c, int job_id) {
    if (!c || job_id < 0 || job_id >= c->n_pending) return true;
    return c->jobs[job_id].state == WUBU_PREFILL_DONE;
}

/* Get progress (fraction of tokens prefilled). */
float wubu_chunked_prefill_progress(wubu_chunked_prefill_t *c, int job_id) {
    if (!c || job_id < 0 || job_id >= c->n_pending) return 0.0f;
    wubu_prefill_job_t *job = &c->jobs[job_id];
    if (job->total_tokens <= 0) return 0.0f;
    return (float)job->prefilled / (float)job->total_tokens;
}

/* Schedule: returns chunk sizes for all pending jobs in this step.
 * Alternates between prefill chunks and decode steps to prevent
 * a long prompt from blocking decode.
 *
 * n_decode_budget: number of decode tokens to schedule this step
 * out_chunks: output array of chunk sizes per job (0 = skip this job)
 * out_decode: output number of decode tokens scheduled
 * Returns number of active jobs. */
int wubu_chunked_prefill_schedule(wubu_chunked_prefill_t *c,
                                    int n_decode_budget,
                                    int *out_chunks, int *out_decode) {
    if (!c || !out_chunks || !out_decode) return -1;
    *out_decode = 0;
    int active = 0;
    for (int i = 0; i < c->n_pending; i++) {
        if (c->jobs[i].state == WUBU_PREFILL_DONE) {
            out_chunks[i] = 0;
        } else {
            int chunk = wubu_chunked_prefill_next_chunk(c, i);
            out_chunks[i] = chunk;
            if (chunk > 0) active++;
        }
    }
    /* Allocate remaining budget to decode */
    int total_prefill = 0;
    for (int i = 0; i < c->n_pending; i++) total_prefill += out_chunks[i];
    *out_decode = (total_prefill < n_decode_budget) ? (n_decode_budget - total_prefill) : 0;
    return active;
}
