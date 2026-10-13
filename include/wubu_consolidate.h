#ifndef WUBU_CONSOLIDATE_H
#define WUBU_CONSOLIDATE_H
#include <stdint.h>

/* wubu_consolidate.h -- PATHS #320 memory consolidation (sleep replay).
 *
 * Sleep replay: during an idle "sleep" phase, the system replays recent
 * experiences to consolidate them -- strengthening important ones (replayed
 * more, weighted by salience) and pruning weak/noisy ones. This mirrors
 * hippocampal replay in biological memory consolidation. Deterministic,
 * fixed-size, real work (no stubs). */

#define WUBU_CONSOL_MAX 128

typedef struct {
    int      id;
    double   salience;    /* importance 0..1 */
    int      replay_count;/* times replayed (strengthened) */
    double   strength;    /* consolidated strength (grows with replay) */
    int      pruned;      /* 1 = dropped (weak) */
} wubu_consol_mem_t;

typedef struct {
    wubu_consol_mem_t mem[WUBU_CONSOL_MAX];
    int n;
    double salience_threshold;  /* below this + not replayed -> prune */
} wubu_consolidate_t;

/* Initialise with a salience threshold for pruning. */
int wubu_consol_init(wubu_consolidate_t *c, double salience_threshold);

/* Add an experience with a salience. Returns index or -1. */
int wubu_consol_add(wubu_consolidate_t *c, int id, double salience);

/* Run one sleep-replay pass: replay the top-K most salient memories
 * (strengthening them), then prune memories below threshold that weren't
 * replayed. Returns the number pruned. */
int wubu_consol_sleep(wubu_consolidate_t *c, int replay_topk);

/* Strength of a memory (0 if pruned/unknown). */
double wubu_consol_strength(const wubu_consolidate_t *c, int id);

/* Number of live (non-pruned) memories. */
int wubu_consol_live_count(const wubu_consolidate_t *c);

#endif
