/*
 * test_causal_symbolic.c -- AW01-AW10 causal + neuro-symbolic verification.
 */
#include "wubu_causal.h"
#include "wubu_symbolic.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_causal_symbolic (AW01-AW10) ===\n");

    /* AW01-AW04: SCM with do / counterfactual / identifiability */
    wubu_scm_t m; m.n = 4; m.n_edges = 0;
    for (int i = 0; i < 4; i++) m.val[i] = 0;
    /* Graph: 0->1->2, 0->3 (0 is root cause, 2 is downstream effect). */
    wubu_scm_add_edge(&m, 0, 1);
    wubu_scm_add_edge(&m, 1, 2);
    wubu_scm_add_edge(&m, 0, 3);
    CHECK(wubu_scm_identifiable(&m, 0, 2) == 1, "SCM: 0 is ancestor of 2 (identifiable)");
    CHECK(wubu_scm_identifiable(&m, 2, 0) == 0, "SCM: 2->0 non-identifiable (refuse)");
    double post[4] = {0};
    wubu_scm_do(&m, 0, 5.0, post);
    CHECK(post[0] == 5.0, "SCM do(0)=5 sets node 0");
    CHECK(post[1] == 5.0 && post[2] == 5.0 && post[3] == 5.0, "SCM do(0) propagates to descendants");
    double cf = 0;
    wubu_scm_counterfactual(&m, 0, 9.0, 2, &cf);
    CHECK(cf == 9.0, "SCM counterfactual do(0)=9 -> node2=9");

    /* AW06(belief): Bayesian belief revision */
    double b = wubu_belief_update(0.5, 0.9);   /* prior 0.5, strong evidence */
    CHECK(b > 0.5, "belief increases with confirming evidence");
    double b2 = wubu_belief_update(0.5, 0.1);  /* contradicting evidence */
    CHECK(b2 < 0.5, "belief decreases with contradicting evidence");

    /* AW06/AW09/AW10: abductive diagnosis + counter-abduction */
    wubu_abduct_t ax[2];
    /* obs=2 (effect observed). H0: cause 0 (prior 0.6, lik 0.8). H1: cause 1 (prior 0.3, lik 0.5). */
    memset(ax, 0, sizeof(ax));
    ax[0].prior = 0.6; ax[0].likelihood = 0.8; ax[0].explains[2] = 1;
    ax[1].prior = 0.3; ax[1].likelihood = 0.5; ax[1].explains[2] = 1;
    int best = -1; double sc = 0;
    CHECK(wubu_abduct_best(ax, 2, 2, &best, &sc) == 0, "abduct finds best hypothesis (H0)");
    CHECK(best == 0, "abduct: H0 has higher prior*lik than H1");
    /* Counter-abduction: a rival with higher posterior defeats H0. */
    wubu_abduct_t rival; memset(&rival, 0, sizeof(rival));
    rival.prior = 0.9; rival.likelihood = 0.95; rival.explains[2] = 1;
    CHECK(wubu_counter_abduct(&ax[0], &rival) == 1, "counter-abduction: rival defeats H0");

    /* AW08: PDDL-lite planner (4 propositions, 2 actions). */
    int nprop = 4, nact = 2, nw = (nprop + 31) / 32;
    unsigned *pre = calloc(nact * nw, sizeof(unsigned));
    unsigned *eff = calloc(nact * nw, sizeof(unsigned));
    /* Action 0: precond {0}, effect toggles {1}. Action 1: precond {1}, effect toggles {2}. */
    pre[0] |= (1u << 0);
    eff[0] |= (1u << 1);
    pre[1] |= (1u << 1);
    eff[1] |= (1u << 2);
    wubu_pddl_t p; p.n_prop = nprop; p.n_actions = nact; p.precond = pre; p.effect = eff;
    unsigned init[1] = { (1u << 0) };     /* state {0} */
    unsigned goal[1] = { (1u << 2) };     /* want {2} */
    int actions[8];
    int len = wubu_pddl_plan(&p, init, goal, 8, actions);
    CHECK(len == 2, "PDDL: plan length 2 (act0 then act1)");
    CHECK(actions[0] == 0 && actions[1] == 1, "PDDL: correct action order");

    /* AW07: symbolic rule engine (forward chaining). */
    wubu_kb_t kb; kb.n_facts = 0;
    /* Facts: human(Ali), human(Muthu). Rule: mortal(X) :- human(X). */
    wubu_fact_add(&kb, 1 /*human*/, 10 /*Ali*/, -1, -1);
    wubu_fact_add(&kb, 1 /*human*/, 11 /*Muthu*/, -1, -1);
    wubu_rule_t r; memset(&r, 0, sizeof(r));
    r.head_pred = 2 /*mortal*/; r.head_args[0] = -1;   /* copy body arg 0 */
    r.body_pred = 1 /*human*/; r.body[0] = -1; r.body[1] = -1; r.body[2] = -1;  /* any human */
    wubu_rules_run(&kb, &r, 1);
    CHECK(wubu_fact_query(&kb, 2, 10, -1, -1), "symbolic: mortal(Ali) derived");
    CHECK(wubu_fact_query(&kb, 2, 11, -1, -1), "symbolic: mortal(Muthu) derived");

    /* AW05: constraint checker (default-deny). */
    wubu_kb_t kb2; kb2.n_facts = 0;
    /* Permission: agent may read file 5. Forbid: agent may delete file 5. */
    wubu_fact_add(&kb2, 998 /*permit*/, 100 /*read*/, 5, -1);
    wubu_fact_add(&kb2, 999 /*forbid*/, 101 /*delete*/, 5, -1);
    CHECK(wubu_constraint_permits(&kb2, 100, 5, -1, -1) == 1, "constraint: read permitted");
    CHECK(wubu_constraint_permits(&kb2, 101, 5, -1, -1) == 0, "constraint: delete denied");
    CHECK(wubu_constraint_permits(&kb2, 102, 5, -1, -1) == 0, "constraint: unknown -> default-deny");

    free(pre); free(eff);
    if (failures == 0) { printf("ALL CAUSAL-SYMBOLIC TESTS PASSED\n"); return 0; }
    printf("%d CAUSAL-SYMBOLIC TEST(S) FAILED\n", failures);
    return 1;
}
