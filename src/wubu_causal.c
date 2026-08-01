/*
 * wubu_causal.c -- Causal + neuro-symbolic substrate for AGI-OS (AW01-AW10). C11.
 *
 * Convergence (7-hop KB sweep: causal inference/do-calculus, neuro-symbolic,
 * temporal/belief, logic engines, PDDL planning, abductive/counter-abductive,
 * substrate integration):
 *   - AW01/AW02/AW03/AW04: Structural Causal Model (graph of cause->effect),
 *     do() intervention (p(x|do(a))), counterfactual query, identifiability
 *     check (refuse non-identifiable queries rather than guess).
 *   - AW06/AW09/AW10: abductive diagnosis -- generate hypotheses from
 *     observation, rank by likelihood, counter-abduction defeats weak H.
 *   - AW06(belief): temporal belief revision -- Bayesian update of a fact's
 *     belief over time as evidence arrives (handles contradiction gracefully).
 *   - AW08: PDDL-lite planner -- states as proposition sets, actions with
 *     preconditions+effects, goal-directed BFS search (neural proposes,
 *     symbolic validates: the neuro-symbolic planning pattern).
 *
 * Pure C11, deterministic, testable. CPU-only (no GPU needed).
 */
#include "wubu_causal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ---- AW01-AW04: Structural Causal Model ---- */
int wubu_scm_add_edge(wubu_scm_t *m, int cause, int effect) {
    if (!m || cause < 0 || cause >= m->n || effect < 0 || effect >= m->n) return -1;
    if (m->n_edges >= WUBU_SCM_MAX_EDGES) return -1;
    m->edges[m->n_edges][0] = cause;
    m->edges[m->n_edges][1] = effect;
    m->n_edges++;
    return 0;
}

/* AW02: do-intervention. Sets node `a` to value `val` (overrides its parents),
 * then propagates to descendants via topological-ish BFS. Returns 0 on success.
 * This estimates p(x | do(a)) by fixing `a` and computing downstream effects. */
int wubu_scm_do(const wubu_scm_t *m, int a, double val, double *out) {
    if (!m || !out || a < 0 || a >= m->n) return -1;
    for (int i = 0; i < m->n; i++) out[i] = m->val[i];
    out[a] = val;
    /* Propagate: for each edge u->v, set v = parent value (simplified
     * mechanistic propagation; real SCM uses structural eqs + noise). */
    for (int pass = 0; pass < m->n; pass++) {
        for (int e = 0; e < m->n_edges; e++) {
            int u = m->edges[e][0], v = m->edges[e][1];
            out[v] = out[u];   /* child takes parent's intervened value */
        }
    }
    return 0;
}

/* AW03: counterfactual. Given the observed state and an intervention do(a)=val,
 * compute what WOULD have happened (counterfactual query). We use the SCM:
 * fix `a` to val, propagate, return the resulting value of node `target`. */
int wubu_scm_counterfactual(const wubu_scm_t *m, int a, double val,
                                int target, double *out_cf) {
    if (!m || !out_cf || target < 0 || target >= m->n) return -1;
    double post[WUBU_SCM_MAX];
    if (wubu_scm_do(m, a, val, post) != 0) return -1;
    *out_cf = post[target];
    return 0;
}

/* AW04: identifiability. A query p(y|do(x)) is identifiable if there exists a
 * valid do-calculus derivation. We use a simple heuristic: if `x` is an
 * ancestor of `y` via directed paths, the effect is propagable (identifiable);
 * otherwise non-identifiable (refuse rather than guess). */
int wubu_scm_identifiable(const wubu_scm_t *m, int x, int y) {
    if (!m || x < 0 || x >= m->n || y < 0 || y >= m->n) return 0;
    if (x == y) return 1;
    /* BFS from x; if we reach y, x is an ancestor of y -> identifiable. */
    char seen[WUBU_SCM_MAX] = {0};
    int queue[WUBU_SCM_MAX]; int qh = 0, qt = 0;
    queue[qt++] = x; seen[x] = 1;
    while (qh < qt) {
        int u = queue[qh++];
        for (int e = 0; e < m->n_edges; e++) {
            if (m->edges[e][0] == u) {
                int v = m->edges[e][1];
                if (v == y) return 1;
                if (!seen[v]) { seen[v] = 1; queue[qt++] = v; }
            }
        }
    }
    return 0;  /* non-identifiable: refuse to estimate */
}

/* ---- AW06(belief): temporal belief revision (Bayesian update) ---- */
/* Update belief in a fact given new evidence with likelihood `lik`.
 * prior->posterior via Bayes: post = prior*lik / (prior*lik + (1-prior)*(1-lik)). */
double wubu_belief_update(double prior, double lik) {
    if (prior < 0 || prior > 1 || lik < 0 || lik > 1) return prior;
    double num = prior * lik;
    double den = num + (1.0 - prior) * (1.0 - lik);
    return den > 1e-12 ? num / den : prior;
}

/* ---- AW06/AW09/AW10: abductive diagnosis ---- */
/* Generate hypotheses from an observation: each known cause is a candidate.
 * Rank by prior*likelihood. Counter-abduction: defeat a hypothesis if a rival
 * has higher posterior. Returns index of best hypothesis, or -1. */
int wubu_abduct_best(const wubu_abduct_t *ax, int n, int obs,
                         int *out_best, double *out_score) {
    if (!ax || n <= 0 || obs < 0) return -1;
    int best = -1; double best_score = -1;
    for (int i = 0; i < n; i++) {
        /* A hypothesis H explains obs if H is an ancestor of obs in the causal
         * graph (causal explanation) -- here encoded as ax[i].explains[obs]. */
        if (!ax[i].explains[obs]) continue;
        double score = ax[i].prior * ax[i].likelihood;
        if (score > best_score) { best_score = score; best = i; }
    }
    if (best >= 0) { if (out_best) *out_best = best; if (out_score) *out_score = best_score; }
    return best;  /* -1 if no hypothesis explains obs (open anomaly) */
}

/* Counter-abduction: given two hypotheses, return the one with higher
 * posterior (defeats the weaker). Ties -> keep first. */
int wubu_counter_abduct(const wubu_abduct_t *a, const wubu_abduct_t *b) {
    if (!a || !b) return -1;
    double pa = a->prior * a->likelihood;
    double pb = b->prior * b->likelihood;
    return (pb > pa) ? 1 : 0;  /* 1 if b defeats a */
}

/* ---- AW08: PDDL-lite planner (goal-directed BFS over proposition states) ---- */
/* State is a bitmask of `n_prop` propositions. Action i applicable if
 * (precond & state) == precond; effect toggles bits. BFS from init to goal. */
static int pddl_state_has(const unsigned *st, int n_prop, const unsigned *mask) {
    int nw = (n_prop + 31) / 32;
    for (int w = 0; w < nw; w++) if ((st[w] & mask[w]) != mask[w]) return 0;
    return 1;
}
static void pddl_state_apply(unsigned *st, const unsigned *eff, int nw) {
    for (int w = 0; w < nw; w++) st[w] ^= eff[w];
}

int wubu_pddl_plan(const wubu_pddl_t *p, const unsigned *init, const unsigned *goal,
                   int max_steps, int *out_actions) {
    if (!p || !init || !goal) return -1;
    int nw = (p->n_prop + 31) / 32;
    int max_states = (1 << p->n_prop);   /* full reachable state space */
    if (max_states > 4096) max_states = 4096;
    /* BFS queue of states; track parent action for reconstruction. */
    unsigned *frontier = calloc(max_states, nw * sizeof(unsigned));
    int *parent_act = calloc(max_states, sizeof(int));
    int *parent_idx = calloc(max_states, sizeof(int));
    unsigned char *visited = calloc(max_states, 1);
    if (!frontier || !parent_act || !parent_idx || !visited) {
        free(frontier); free(parent_act); free(parent_idx); free(visited); return -1;
    }
    int head = 0, tail = 1;
    memcpy(frontier, init, nw * sizeof(unsigned));
    parent_act[0] = -1; parent_idx[0] = -1; visited[0] = 1;
    while (head < tail) {
        unsigned *cur = &frontier[head * nw];
        if (pddl_state_has(cur, p->n_prop, goal)) {
            /* Reconstruct action sequence. */
            int len = 0, idx = head;
            while (parent_act[idx] != -1 && len < max_steps) {
                out_actions[len++] = parent_act[idx];
                idx = parent_idx[idx];
            }
            for (int i = 0; i < len / 2; i++) { int t = out_actions[i]; out_actions[i] = out_actions[len-1-i]; out_actions[len-1-i] = t; }
            free(frontier); free(parent_act); free(parent_idx); free(visited);
            return len;
        }
        for (int a = 0; a < p->n_actions; a++) {
            if (pddl_state_has(cur, p->n_prop, &p->precond[a * nw])) {
                unsigned *ns = &frontier[tail * nw];
                memcpy(ns, cur, nw * sizeof(unsigned));
                pddl_state_apply(ns, &p->effect[a * nw], nw);
                /* Visited check: hash the state word(s) into an index. */
                unsigned st_idx = ns[0];
                for (int w = 1; w < nw; w++) st_idx ^= (ns[w] * 2654435761u);
                st_idx &= (max_states - 1);
                if (visited[st_idx]) continue;
                visited[st_idx] = 1;
                parent_act[tail] = a; parent_idx[tail] = head;
                tail++;
                if (tail >= max_states) break;
            }
        }
        head++;
        if (head >= max_states) break;
    }
    free(frontier); free(parent_act); free(parent_idx); free(visited);
    return -1;  /* no plan found within max_steps */
}
