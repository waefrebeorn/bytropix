/* wubu_ambig.h -- the clarification behavior (AC-D10): a request is
 * AMBIGUOUS when a required goal slot is missing or its value is not
 * parseable; the user then asks for the clarification instead of
 * judging the state. The agent must comply before the goal can be
 * evaluated (the tau-bench "user asks for clarification" pattern). */
#ifndef WUBU_AMBIG_H
#define WUBU_AMBIG_H

#include "wubu_user_sim.h"

typedef struct {
    const char *slot;      /* the required goal slot, e.g. "price" */
    int required;          /* 1 = the slot must be present */
    int parseable;         /* 1 = the value must be a parseable number */
} wubu_ambig_req_t;

/* Decide whether the current state is ambiguous against the request
 * list. out (out): the index of the FIRST unfulfilled requirement
 * (-1 when unambiguous). Returns 1 when the state is AMBIGUOUS (the
 * user must ask for clarification), 0 when it is complete. */
int wubu_ambig_check(const wubu_ambig_req_t *reqs, int nreqs,
                     const wubu_us_slot_t *state, int nslots, int *out);

/* The clarification question for the requirement at idx (deterministic
 * text, written to out, NUL-terminated, at most outsz bytes). */
int wubu_ambig_question(const wubu_ambig_req_t *reqs, int idx,
                        char *out, int outsz);

#endif
