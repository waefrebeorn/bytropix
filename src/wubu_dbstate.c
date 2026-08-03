/* wubu_dbstate.c -- the DB-state reward verifier. */
#include <stdlib.h>
#include <string.h>
#include "wubu_dbstate.h"

static const char *find(const wubu_db_slot_t *state, int nslots,
                        const char *slot)
{
    for (int i = 0; i < nslots; i++)
        if (strcmp(state[i].slot, slot) == 0) return state[i].value;
    return NULL;
}

static int cmp(double a, const char *op, double b)
{
    if (strcmp(op, "<") == 0) return a < b;
    if (strcmp(op, ">") == 0) return a > b;
    if (strcmp(op, "<=") == 0) return a <= b;
    if (strcmp(op, ">=") == 0) return a >= b;
    if (strcmp(op, "!=") == 0) return a != b;
    return a == b;
}

static int goal_met(const wubu_db_goal_t *g, const char *v)
{
    char *e1 = NULL, *e2 = NULL;
    double ga = strtod(g->value, &e1);
    double va = strtod(v, &e2);
    if (e1 != g->value && *e1 == '\0' && e2 != v && *e2 == '\0')
        return cmp(va, g->op, ga);
    if (strcmp(g->op, "==") == 0) return strcmp(v, g->value) == 0;
    if (strcmp(g->op, "!=") == 0) return strcmp(v, g->value) != 0;
    return 0;
}

int wubu_db_verify(const wubu_db_goal_t *goals, int ngoals,
                   const wubu_db_slot_t *state, int nslots)
{
    if (!goals || ngoals < 1 || !state || nslots < 1) return -1;
    int missing = 0;
    for (int i = 0; i < ngoals; i++) {
        const char *v = find(state, nslots, goals[i].slot);
        if (!v) { missing = 1; continue; }
        if (!goal_met(&goals[i], v)) return 0;
    }
    return missing ? -1 : 1;
}

float wubu_db_reward(const wubu_db_goal_t *goals, int ngoals,
                     const wubu_db_slot_t *state, int nslots)
{
    if (!goals || ngoals < 1) return 0;
    int met = 0;
    for (int i = 0; i < ngoals; i++) {
        const char *v = find(state, nslots, goals[i].slot);
        if (v && goal_met(&goals[i], v)) met++;
    }
    return (float)met / (float)ngoals;
}
