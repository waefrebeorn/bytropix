#ifndef WUBU_HUMANGATE_H
#define WUBU_HUMANGATE_H
#include <stdint.h>

/* wubu_humangate.h -- AGENT #146 DA-3 human gate + freeze (death condition).
 *
 * The human-in-the-loop veto: a human operator can FREEZE the self-improvement
 * loop at any time. Once frozen, the loop is halted -- no further mutations are
 * proposed, verified, or promoted -- until (and only until) the human unfreezes
 * with a valid token. This is the "death condition": the human gate is the
 * final authority over the AGI's self-modification. The freeze is kernel-owned
 * (the loop cannot clear it itself) and one-way to set. Fail-closed: a frozen
 * gate denies every mutation. Deterministic, no allocation, real work. */

typedef struct {
    int  frozen;          /* 1 = loop frozen (halted) */
    int  veto_count;      /* total human vetoes (observability) */
    uint64_t freeze_token;/* token required to unfreeze (human-held) */
} wubu_humangate_t;

/* Initialise the gate (unfrozen, no vetoes). The freeze_token is the secret
 * the human must present to unfreeze. */
int wubu_humangate_init(wubu_humangate_t *g, uint64_t freeze_token);

/* Human veto: freeze the loop immediately. One-way (cannot be undone without
 * the token). Increments veto_count. Returns 0. */
int wubu_humangate_veto(wubu_humangate_t *g);

/* May the loop proceed with a mutation right now? Fail-closed: 0 if frozen. */
int wubu_humangate_may_mutate(const wubu_humangate_t *g);

/* Human unfreeze: requires the correct token. Returns 1 if unfrozen, 0 if
 * wrong token (stays frozen). */
int wubu_humangate_unfreeze(wubu_humangate_t *g, uint64_t token);

/* Is the loop currently frozen? */
int wubu_humangate_frozen(const wubu_humangate_t *g);

/* Total vetoes. */
int wubu_humangate_vetoes(const wubu_humangate_t *g);

#endif
