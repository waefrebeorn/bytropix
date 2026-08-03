/* wubu_user_sim.h -- the tau-bench-style user-simulator harness: a
 * scripted user with a goal constraint + a persona. The user reacts to
 * the agent's evolving state (satisfied / not-yet / policy pushback),
 * the state is verified against the goal (the tau-bench DB-state check),
 * and the persona's template generates the next user utterance -- the
 * data-generator side of the "usable agentic" doctrine. */
#ifndef WUBU_USER_SIM_H
#define WUBU_USER_SIM_H

typedef struct {
    const char *slot;    /* the state slot, e.g. "price" */
    const char *value;   /* its current string value */
} wubu_us_slot_t;

typedef struct {
    const char *name;      /* the persona name */
    const char *goal_slot; /* the slot the user cares about */
    const char *goal_op;   /* "<" ">" "<=" ">=" "==" "!=" */
    double goal_value;     /* the numeric bound */
    int verbose;           /* the persona verbosity (0 = terse, 1 = verbose) */
    const char *goal_str;  /* the exact-value goal (non-numeric slot); */
                           /* NULL = the numeric goal_value constraint */
    const char *goal_note; /* the human goal text (for the utterance) */
} wubu_us_user_t;

/* The user's reaction to the current state:
 *   1  = the goal is satisfied (the agent can finish)
 *   0  = the goal is not yet met (keep going)
 *  -1  = the policy/goal is VIOLATED (the user pushes back)
 * A missing goal slot counts as not-yet; a numeric comparison against
 * the parsed value; a non-numeric slot uses "==" on the string. */
int wubu_us_react(const wubu_us_user_t *u, const wubu_us_slot_t *state,
                  int nslots);

/* The tau-bench DB-state verification: 1 when the goal is met. */
int wubu_us_verify(const wubu_us_user_t *u, const wubu_us_slot_t *state,
                   int nslots);

/* The persona's next utterance for the current reaction (written to out,
 * NUL-terminated, at most outsz bytes). Deterministic per persona. */
int wubu_us_utter(const wubu_us_user_t *u, const wubu_us_slot_t *state,
                  int nslots, char *out, int outsz);

#endif
