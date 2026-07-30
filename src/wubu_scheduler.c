/*
 * wubu_scheduler.c -- Stage 2 deterministic token scheduler.
 *
 * Research basis (Kevin-Bacon hop 6-7): Anyscale 2025 continuous
 * batching + iteration-level KV-cache merge. Model-agnostic: operates
 * on int token ids only, never touches model weights or KV internals.
 *
 * Two policies:
 *   1. FIFO: oldest scheduled sequence wins next decode step.
 *   2. ROUND_ROBIN: cycles active sequences, bounded by max_batch.
 *
 * Invariants:
 *   - max_seq < SCHED_MAX_SEQ
 *   - n_active <= max_batch at all times
 *   - completed sequences get their kv_cache_len set to zero for
 *     caller-side reuse/reclaim.
 */

#include "wubu_scheduler.h"
#include <string.h>
#include <stdio.h>

#ifndef SCHED_MAX_SEQ
#define SCHED_MAX_SEQ 1024
#endif
#ifndef SCHED_MAX_BATCH
#define SCHED_MAX_BATCH 16
#endif

struct seq {
    int id;
    int n_tokens;
    int completed;
};
struct sched {
    struct seq seqs[SCHED_MAX_SEQ];
    int head, tail, active, max_batch;
    int policy; /* 0=FIFO, 1=ROUND_ROBIN */
    int rr_index;
};

static struct sched G;

void wubu_sched_init(int max_batch, int policy) {
    memset(&G, 0, sizeof(G));
    G.max_batch = max_batch > 0 ? (max_batch < SCHED_MAX_BATCH ? max_batch : SCHED_MAX_BATCH) : SCHED_MAX_BATCH;
    G.policy = policy ? 1 : 0;
}

int wubu_sched_submit(int id, int n_tokens) {
    if (G.active >= SCHED_MAX_SEQ) return -1;
    int i = (G.tail + G.active) % SCHED_MAX_SEQ;
    G.seqs[i].id = id;
    G.seqs[i].n_tokens = n_tokens;
    G.seqs[i].completed = 0;
    G.active++;
    return 0;
}

int wubu_sched_next(int ids_out[SCHED_MAX_BATCH], int max_ids) {
    int n = 0;
    if (G.active == 0) return 0;
    int m = max_ids < G.active ? max_ids : G.active;
    if (G.policy == 0) {
        /* FIFO: pop from head */
        for (int i = 0; i < m; i++) {
            ids_out[i] = G.seqs[G.head].id;
            G.head = (G.head + 1) % SCHED_MAX_SEQ;
            G.active--;
            n++;
        }
    } else {
        /* ROUND_ROBIN: scan from rr_index in a single pass, up to m */
        int scanned = 0, picked = 0;
        while (picked < m && scanned < G.active) {
            int idx = (G.head + (G.rr_index % G.active)) % SCHED_MAX_SEQ;
            ids_out[picked++] = G.seqs[idx].id;
            G.rr_index++;
            scanned++;
            /* remove picked slot by tail-swap */
            int last = (G.head + G.active - 1) % SCHED_MAX_SEQ;
            if (idx != last) {
                G.seqs[idx] = G.seqs[last];
            }
            G.active--;
        }
        n = picked;
        while (G.head < G.active && G.seqs[G.head].completed) {
            G.head = (G.head + 1) % SCHED_MAX_SEQ;
        }
    }
    return n;
}

void wubu_sched_complete(int id) {
    for (int i = 0; i < G.active; i++) {
        int idx = (G.head + i) % SCHED_MAX_SEQ;
        if (G.seqs[idx].id == id) {
            G.seqs[idx].completed = 1;
            return;
        }
    }
}

int wubu_sched_active(void) { return G.active; }
