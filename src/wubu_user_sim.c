/* wubu_user_sim.c -- the tau-bench-style user-simulator harness. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_user_sim.h"

static const char *find_slot(const wubu_us_user_t *u,
                             const wubu_us_slot_t *state, int nslots)
{
    for (int i = 0; i < nslots; i++)
        if (strcmp(state[i].slot, u->goal_slot) == 0)
            return state[i].value;
    return NULL;
}

static int cmp_num(double a, const char *op, double b)
{
    if (strcmp(op, "<") == 0) return a < b;
    if (strcmp(op, ">") == 0) return a > b;
    if (strcmp(op, "<=") == 0) return a <= b;
    if (strcmp(op, ">=") == 0) return a >= b;
    if (strcmp(op, "==") == 0) return a == b;
    return a != b;
}

int wubu_us_react(const wubu_us_user_t *u, const wubu_us_slot_t *state,
                  int nslots)
{
    if (!u || !state || nslots < 1) return 0;
    const char *v = find_slot(u, state, nslots);
    if (!v) return 0;                 /* the slot is missing: keep going */
    if (u->goal_str)                     /* the exact-value goal */
        return strcmp(v, u->goal_str) == 0 ? 1 : -1;
    char *end = NULL;
    double dv = strtod(v, &end);
    if (end != v && *end == '\0') {   /* numeric: the numeric constraint */
        return cmp_num(dv, u->goal_op, u->goal_value) ? 1 : -1;
    }
    return -1;                          /* unparsable value: push back */
}

int wubu_us_verify(const wubu_us_user_t *u, const wubu_us_slot_t *state,
                   int nslots)
{
    return wubu_us_react(u, state, nslots) == 1;
}

int wubu_us_utter(const wubu_us_user_t *u, const wubu_us_slot_t *state,
                  int nslots, char *out, int outsz)
{
    if (!u || !out || outsz < 1) return 0;
    int r = wubu_us_react(u, state, nslots);
    if (r == 1) {
        if (u->verbose)
            snprintf(out, outsz, "Perfect -- %s meets my requirement. Thank you!",
                     u->goal_slot);
        else
            snprintf(out, outsz, "ok, done");
    } else if (r == -1) {
        if (u->verbose)
            snprintf(out, outsz, "Wait, that does not meet my goal (%s %s %.0f).",
                     u->goal_slot, u->goal_op, u->goal_value);
        else
            snprintf(out, outsz, "that's not what I asked for");
    } else {
        if (u->verbose)
            snprintf(out, outsz, "What about %s? I need %s %s %.0f.",
                     u->goal_slot, u->goal_slot, u->goal_op, u->goal_value);
        else
            snprintf(out, outsz, "and %s?", u->goal_slot);
    }
    return 1;
}
