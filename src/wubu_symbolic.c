/*
 * wubu_symbolic.c -- Symbolic rule engine + constraint checker (AW05, AW07). C11.
 *
 * Convergence (Prolog/ASP/DL neuro-symbolic 7-hop):
 *   - AW07: a Prolog-ish engine -- facts (predicate(args)) + rules
 *     (head :- body1, body2, ...). Forward-chaining resolution derives new
 *     facts. Pure C, CPU-only. Enables deductive inference over the agent's
 *     knowledge (the "symbolic" half of neuro-symbolic).
 *   - AW05: constraint checker -- safety invariants expressed as logical
 *     rules; the engine verifies a proposed action satisfies all constraints
 *     before it is permitted (symbolic verifier in the decode path, feeding
 *     wubu_safekern). Default-deny: unverified actions rejected.
 *
 * Pure C11, deterministic, testable.
 */
#include "wubu_symbolic.h"
#include <stdlib.h>
#include <string.h>

/* AW07: add a fact (predicate id + up to 3 integer args). */
int wubu_fact_add(wubu_kb_t *kb, int pred, int a0, int a1, int a2) {
    if (!kb || kb->n_facts >= WUBU_KB_MAX_FACTS) return -1;
    int i = kb->n_facts++;
    kb->facts[i].pred = pred;
    kb->facts[i].args[0] = a0; kb->facts[i].args[1] = a1; kb->facts[i].args[2] = a2;
    return 0;
}

static int fact_matches(const wubu_fact_t *f, int pred, int a0, int a1, int a2) {
    if (f->pred != pred) return 0;
    if (a0 >= 0 && f->args[0] != a0) return 0;
    if (a1 >= 0 && f->args[1] != a1) return 0;
    if (a2 >= 0 && f->args[2] != a2) return 0;
    return 1;
}

/* AW07: query a fact (args <0 act as wildcards). Returns 1 if present. */
int wubu_fact_query(const wubu_kb_t *kb, int pred, int a0, int a1, int a2) {
    if (!kb) return 0;
    for (int i = 0; i < kb->n_facts; i++)
        if (fact_matches(&kb->facts[i], pred, a0, a1, a2)) return 1;
    return 0;
}

/* AW07: apply one forward-chaining rule. Rule: head(pred,ha0,ha1,ha2) :-
 * body(pred,b0,b1,b2). A var is a negative arg (e.g. -1 = bind to body's
 * value). We support simple rules where head args are constants or copied
 * from the single body fact (var markers -1,-2,-3 copy body args 0,1,2). */
int wubu_rule_apply(wubu_kb_t *kb, const wubu_rule_t *r) {
    if (!kb || !r) return 0;
    int fired = 0;
    for (int i = 0; i < kb->n_facts; i++) {
        if (!fact_matches(&kb->facts[i], r->body_pred, r->body[0], r->body[1], r->body[2]))
            continue;
        /* Head args: >=0 literal; <0 means copy body arg (-1->args[0], etc). */
        int ha[3];
        for (int k = 0; k < 3; k++) {
            if (r->head_args[k] < 0) ha[k] = kb->facts[i].args[-(r->head_args[k] + 1)];
            else ha[k] = r->head_args[k];
        }
        /* Skip if already known. */
        if (wubu_fact_query(kb, r->head_pred, ha[0], ha[1], ha[2])) continue;
        wubu_fact_add(kb, r->head_pred, ha[0], ha[1], ha[2]);
        fired = 1;
    }
    return fired;
}

int wubu_rules_run(wubu_kb_t *kb, const wubu_rule_t *rules, int n_rules) {
    if (!kb || !rules || n_rules <= 0) return 0;
    int changed = 1, total = 0;
    while (changed) {
        changed = 0;
        for (int i = 0; i < n_rules; i++)
            if (wubu_rule_apply(kb, &rules[i])) changed = 1;
        if (++total > WUBU_KB_MAX_FACTS) break;  /* fixpoint guard */
    }
    return total;
}

/* AW05: constraint checker. An action `act` is permitted only if it does NOT
 * violate any constraint rule. Constraint: forbidden(pred, a0, a1, a2) fact
 * present in KB -> action denied. Default-deny: unknown -> denied. */
int wubu_constraint_permits(const wubu_kb_t *kb, int act_pred, int a0, int a1, int a2) {
    if (!kb) return 0;
    /* Hard-deny list: predicate id 999 = forbidden. */
    if (wubu_fact_query(kb, 999, act_pred, a0, a1)) return 0;  /* denied */
    /* If no explicit permission fact (pred 998), deny by default. */
    return wubu_fact_query(kb, 998, act_pred, a0, a1) ? 1 : 0; /* default-deny */
}
