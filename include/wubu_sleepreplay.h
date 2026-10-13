#ifndef WUBU_SLEEPREPLAY_H
#define WUBU_SLEEPREPLAY_H

/* wubu_sleepreplay.h -- BRAIN #125 memory consolidation sleep-replay.
 *
 * Sleep-replay memory consolidation: during a "sleep" phase, recent
 * experiences (episodes) are replayed to consolidate them into long-term
 * memory. Replay strength determines how much of an episode is retained;
 * frequently-replayed episodes consolidate strongly (resist forgetting), while
 * unreplayed ones decay. Models hippocampal replay -> cortical consolidation.
 * Deterministic, real work (no stubs). */

#define WUBU_SR_MAX 64

typedef struct {
    int id;
    double strength;       /* current consolidation strength (0..1) */
    int replay_count;      /* times replayed */
} wubu_sr_ep_t;

typedef struct {
    wubu_sr_ep_t eps[WUBU_SR_MAX];
    int n;
    double decay;          /* per-sleep decay of unstrengthened memory */
    double replay_gain;    /* strength gained per replay */
} wubu_sleepreplay_t;

/* Initialise with decay + replay-gain rates. */
int wubu_sr_init(wubu_sleepreplay_t *s, double decay, double replay_gain);

/* Record a new episode (initial strength 0.5). Returns 0/-1. */
int wubu_sr_record(wubu_sleepreplay_t *s, int id);

/* Run one sleep cycle: every episode decays by 'decay', then the given
 * episodes (ids in replay_ids, count n) are replayed (+replay_gain, capped at
 * 1). Returns the number consolidated (strength >= 0.8) after the cycle. */
int wubu_sr_sleep(wubu_sleepreplay_t *s, const int *replay_ids, int n);

/* Strength of an episode. */
double wubu_sr_strength(const wubu_sleepreplay_t *s, int id);

/* Is an episode consolidated (strength >= 0.8, i.e. in long-term memory)? */
int wubu_sr_consolidated(const wubu_sleepreplay_t *s, int id);

/* Total consolidated episodes. */
int wubu_sr_n_consolidated(const wubu_sleepreplay_t *s);

#endif
