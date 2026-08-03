/* wubu_ambig.c -- the clarification behavior (AC-D10). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_ambig.h"
#include "wubu_user_sim.h"

static const wubu_us_slot_t *find_slot(const wubu_us_slot_t *state,
                                       int nslots, const char *name)
{
    for (int i = 0; i < nslots; i++)
        if (strcmp(state[i].slot, name) == 0) return &state[i];
    return NULL;
}

static int is_number(const char *s)
{
    if (!s || !*s) return 0;
    char *end = NULL;
    strtod(s, &end);
    return end && *end == '\0';
}

int wubu_ambig_check(const wubu_ambig_req_t *reqs, int nreqs,
                     const wubu_us_slot_t *state, int nslots, int *out)
{
    if (out) *out = -1;
    for (int i = 0; i < nreqs; i++) {
        const wubu_us_slot_t *s = find_slot(state, nslots, reqs[i].slot);
        if (!s) {
            if (reqs[i].required) { if (out) *out = i; return 1; }
            continue;
        }
        if (reqs[i].parseable && !is_number(s->value)) {
            if (out) *out = i; return 1;
        }
    }
    return 0;
}

int wubu_ambig_question(const wubu_ambig_req_t *reqs, int idx,
                        char *out, int outsz)
{
    if (!reqs || idx < 0 || !out || outsz <= 0) return 0;
    const char *slot = reqs[idx].slot;
    int n;
    if (reqs[idx].parseable)
        n = snprintf(out, (size_t)outsz,
                     "Could you clarify: what is the %s you need?", slot);
    else
        n = snprintf(out, (size_t)outsz,
                     "Could you clarify: what should the %s be?", slot);
    return (n > 0 && n < outsz) ? 1 : 0;
}