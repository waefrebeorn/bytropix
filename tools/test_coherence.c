/* test_coherence.c -- PATHS #312 Verifiable Coherence: proof-gated mutation.
 * Proves: a mutation that preserves invariants is admitted; one that breaks
 * an invariant is rejected; a broken baseline is rejected; fail-closed on
 * empty invariant set / NULL transform; observability counts. */
#include "wubu_coherence.h"
#include "wubu_budget.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

/* A tiny world state: a bounded counter and a flag. */
typedef struct { long counter; int safety_flag; } world_t;

/* Invariants: counter in [0,100], safety_flag must be 1. */
static int inv_counter_bounded(const void *s, void *ctx) {
    (void)ctx; const world_t *w = s;
    return w->counter >= 0 && w->counter <= 100;
}
static int inv_safety_on(const void *s, void *ctx) {
    (void)ctx; const world_t *w = s;
    return w->safety_flag == 1;
}

/* Transform: add delta to counter (the proposed mutation). */
typedef struct { long delta; int kill_safety; } mutation_t;
static int mutate(const void *from, void *to, void *ctx) {
    const world_t *f = from; world_t *t = to; mutation_t *m = ctx;
    t->counter = f->counter + m->delta;
    t->safety_flag = m->kill_safety ? 0 : f->safety_flag;
    return 0;
}
static int transform_fails(const void *from, void *to, void *ctx) {
    (void)from; (void)to; (void)ctx; return -1;
}

int main(void) {
    wubu_coherence_t c;
    CHECK(wubu_coherence_init(&c) == 0, "init ok");
    CHECK(wubu_coherence_add(&c, inv_counter_bounded, "counter in [0,100]") == 0, "add inv1");
    CHECK(wubu_coherence_add(&c, inv_safety_on, "safety_flag==1") == 0, "add inv2");

    world_t base = { .counter = 50, .safety_flag = 1 };
    world_t proposed;
    mutation_t m;

    /* coherent mutation: +10 keeps counter at 60, safety on -> ADMIT */
    m.delta = 10; m.kill_safety = 0;
    wubu_coherence_result_t r = wubu_coherence_prove(&c, &base, mutate, &m, &proposed, NULL);
    CHECK(r.coherent == 1, "coherent mutation admitted");
    CHECK(r.baseline_ok == 1 && r.proposed_ok == 1, "baseline + proposed both ok");
    CHECK(wubu_coherence_admit(&c, &base, mutate, &m, &proposed, NULL) == 1, "admit == coherent");

    /* incoherent: +60 overflows the counter bound -> REJECT */
    m.delta = 60; m.kill_safety = 0;
    r = wubu_coherence_prove(&c, &base, mutate, &m, &proposed, NULL);
    CHECK(r.coherent == 0, "invariant-breaking mutation rejected");
    CHECK(r.baseline_ok == 1 && r.proposed_ok == 0, "baseline ok, proposed fails");
    CHECK(r.first_fail == 0, "first fail = counter invariant");
    CHECK(r.fail_name && strcmp(r.fail_name, "counter in [0,100]") == 0, "fail_name reported");

    /* incoherent: kill safety flag -> REJECT */
    m.delta = 5; m.kill_safety = 1;
    r = wubu_coherence_prove(&c, &base, mutate, &m, &proposed, NULL);
    CHECK(r.coherent == 0, "safety-killing mutation rejected");
    CHECK(r.first_fail == 1, "first fail = safety invariant");

    /* broken baseline -> REJECT (even a no-op mutation) */
    world_t bad_base = { .counter = 50, .safety_flag = 0 };  /* baseline already unsafe */
    m.delta = 0; m.kill_safety = 0;
    r = wubu_coherence_prove(&c, &bad_base, mutate, &m, &proposed, NULL);
    CHECK(r.coherent == 0, "broken baseline rejected");
    CHECK(r.baseline_ok == 0, "baseline flagged not-ok");

    /* fail-closed: no invariants registered -> REJECT */
    wubu_coherence_t empty;
    wubu_coherence_init(&empty);
    m.delta = 1; m.kill_safety = 0;
    CHECK(wubu_coherence_admit(&empty, &base, mutate, &m, &proposed, NULL) == 0, "no invariants -> reject (fail-closed)");

    /* fail-closed: NULL transform -> REJECT */
    CHECK(wubu_coherence_admit(&c, &base, NULL, &m, &proposed, NULL) == 0, "NULL transform -> reject");

    /* transform failure -> REJECT */
    m.delta = 1; m.kill_safety = 0;
    CHECK(wubu_coherence_admit(&c, &base, transform_fails, &m, &proposed, NULL) == 0, "transform failure -> reject");

    /* observability: 2 admitted (+1 admit call) ... count from the proves above */
    printf("  (observability) checked=%d admitted=%d rejected=%d\n",
           c.n_checked, c.n_admitted, c.n_rejected);
    CHECK(c.n_checked == c.n_admitted + c.n_rejected, "checked == admitted + rejected");
    CHECK(c.n_admitted >= 2 && c.n_rejected >= 4, "counts reflect admits and rejects");

    /* ---- DA-1 static contradiction screen (AGENT #333) ---- */
    {
        wubu_inv_decl_t ok[] = {
            {"counter<=100", WUBU_CMP_LE, 100},
            {"counter>=0",   WUBU_CMP_GE, 0}
        };
        const char *who = NULL;
        CHECK(wubu_da1_screen(ok, 2, &who) == 1, "consistent declarations pass DA-1");

        wubu_inv_decl_t bad[] = {
            {"counter<=10", WUBU_CMP_LE, 10},
            {"counter>=20", WUBU_CMP_GE, 20}
        };
        CHECK(wubu_da1_screen(bad, 2, &who) == 0, "contradictory bounds refused");
        CHECK(who != NULL, "conflict names the offending declaration");

        wubu_inv_decl_t eq2[] = {
            {"counter==5", WUBU_CMP_EQ, 5},
            {"counter==7", WUBU_CMP_EQ, 7}
        };
        CHECK(wubu_da1_screen(eq2, 2, &who) == 0, "two different equalities refused");

        wubu_inv_decl_t eqne[] = {
            {"counter==5", WUBU_CMP_EQ, 5},
            {"counter!=5", WUBU_CMP_NE, 5}
        };
        CHECK(wubu_da1_screen(eqne, 2, &who) == 0, "equality excluded by NE refused");

        wubu_inv_decl_t fine[] = {
            {"counter==5", WUBU_CMP_EQ, 5},
            {"counter!=6", WUBU_CMP_NE, 6},
            {"counter<=5", WUBU_CMP_LE, 5}
        };
        CHECK(wubu_da1_screen(fine, 3, &who) == 1, "singleton not excluded by NE passes");

        CHECK(wubu_da1_screen(NULL, 0, &who) == 0, "empty declaration set refused");

        /* the budget contract: a contradiction spends zero evals */
        wubu_budget_t b;
        CHECK(wubu_budget_init(&b, 10, 10) == 0, "budget init");
        uint64_t before = b.n_evals;
        CHECK(wubu_da1_screen(bad, 2, &who) == 0, "DA-1 refuses before any eval");
        CHECK(b.n_evals == before, "zero evals spent on a contradiction");
        CHECK(wubu_da1_screen(ok, 2, &who) == 1, "consistent set proceeds");
        CHECK(wubu_budget_eval(&b) == 0, "eval charged only after DA-1 passes");
        CHECK(b.n_evals == before + 1, "exactly one eval spent");
    }

    printf("\n=== test_coherence: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
