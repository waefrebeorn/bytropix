/*
 * wubu_loopguard.h -- Missing-need guards for the AGI-OS (AG01/AG05/AG06/AG08).
 */
#ifndef WUBU_LOOPGUARD_H
#define WUBU_LOOPGUARD_H

#include <stddef.h>

/* AG01 */
typedef struct {
    long max_steps;       /* hard step ceiling (recursive-loop termination) */
    long deadline_ns;     /* wall-clock deadline (0 = none) */
} wubu_loopguard_t;

/* AG05 */
typedef struct {
    unsigned long long *nonce;  /* append-only attribution nonces */
    int count;
    int cap;
} wubu_traj_t;

/* AG06 */
typedef struct {
    const char *agent;
    long window;
    int  calls;
    int  max_per_window;
} wubu_toolcap_t;

/* AG08 */
typedef struct {
    float sensitivity;    /* severity threshold for HITL gating */
    int   expected_token; /* external approval token (from operator/human) */
} wubu_hitl_t;

int  wubu_loop_may_continue(const wubu_loopguard_t *g, long step, long now_ns);
unsigned long long wubu_traj_append(wubu_traj_t *t, const char *agent, const char *action);
int  wubu_tool_allowed(wubu_toolcap_t *c, const char *agent, long window_now);
int  wubu_hitl_approve(const wubu_hitl_t *h, float severity, int approval_token);

#endif
