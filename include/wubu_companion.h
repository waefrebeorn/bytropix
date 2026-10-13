#ifndef WUBU_COMPANION_H
#define WUBU_COMPANION_H

/* wubu_companion.h -- UX #246 Bonzi companion AGI-UX persona.
 *
 * The Bonzi-style companion persona: the AGI's on-screen presence with a
 * consistent personality expressed through MOOD states that transition based
 * on system state (idle, working, alert, celebrating, concerned). The persona
 * gives the AGI a legible, continuous presence rather than a bare terminal --
 * a single state machine drives its expression. Deterministic, real work
 * (no stubs). */

typedef enum {
    WUBU_COMP_IDLE=0,     /* waiting for input */
    WUBU_COMP_WORKING,    /* actively processing */
    WUBU_COMP_ALERT,      /* something needs attention */
    WUBU_COMP_CELEBRATE,  /* a goal was completed */
    WUBU_COMP_CONCERN,    /* a failure / risk detected */
} wubu_comp_mood_t;

typedef struct {
    wubu_comp_mood_t mood;
    int n_transitions;
    /* time-in-mood counters for expression cadence */
    int ticks_in_mood;
} wubu_companion_t;

/* Initialise (idle). */
int wubu_comp_init(wubu_companion_t *c);

/* Set the mood directly from a system event. Returns 0/-1. */
int wubu_comp_set(wubu_companion_t *c, wubu_comp_mood_t mood);

/* Advance one expression tick (updates time-in-mood). Returns 0/-1. */
int wubu_comp_tick(wubu_companion_t *c);

/* Current mood. */
wubu_comp_mood_t wubu_comp_mood(const wubu_companion_t *c);

/* A short expression string for the current mood (what the companion
 * "says"/shows). Returns a static string. */
const char *wubu_comp_expression(const wubu_companion_t *c);

/* Number of mood transitions so far. */
int wubu_comp_transitions(const wubu_companion_t *c);

#endif
