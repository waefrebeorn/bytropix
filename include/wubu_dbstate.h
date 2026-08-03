/* wubu_dbstate.h -- the tau-bench DB-state reward verifier: the objective,
 * stateful reward that replaces preference ratings. The agent's final
 * state (a slot table) is compared against the annotated GOAL state; the
 * reward = 1 when every goal constraint holds, partial when some do. */
#ifndef WUBU_DBSTATE_H
#define WUBU_DBSTATE_H

typedef struct {
    const char *slot;   /* the slot name */
    const char *op;     /* "==" "<" ">" "<=" ">=" "!=" */
    const char *value;  /* the goal value (string) */
} wubu_db_goal_t;

typedef struct {
    const char *slot;
    const char *value;
} wubu_db_slot_t;

/* The state-vs-goal comparison:
 *   returns 1 if ALL the goals are met, 0 if any is unmet,
 *   -1 if a goal slot is missing from the state.
 * Numeric ops parse the values; "==" compares strings when either side
 * does not parse as a number. */
int wubu_db_verify(const wubu_db_goal_t *goals, int ngoals,
                   const wubu_db_slot_t *state, int nslots);

/* The partial reward: the fraction of the goals met (missing = unmet).
 * The tau-bench "stateful evaluation" analogue. */
float wubu_db_reward(const wubu_db_goal_t *goals, int ngoals,
                     const wubu_db_slot_t *state, int nslots);

#endif
