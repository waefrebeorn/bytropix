/*
 * test_continual.c -- BB01-BB04 verification.
 */
#include "wubu_experibuf.h"
#include "wubu_ewc.h"
#include "wubu_taskbd.h"
#include "wubu_distill.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { failures++; printf("FAIL: %s\n", msg); } \
    else printf("  ok: %s\n", msg); \
} while(0)

int main() {
    int fails = 0; (void)fails;

    /* BB01: Experience replay (reservoir sampling) */
    printf("=== BB01: Experience Replay ===\n");
    wubu_experibuf_t r;
    CHECK(wubu_experibuf_init(&r) == 0, "replay init");
    double p1[15] = {1.0};
    double p2[15] = {2.0, 3.0};
    double p3[15] = {4.0};
    CHECK(wubu_experibuf_add(&r, p1, 1, "512K-SRT", 27.0, 0.0, 1) == 0, "replay add p1");
    CHECK(wubu_experibuf_add(&r, p2, 2, "512K-MQA", 26.5, 0.1, 1) == 0, "replay add p2");
    CHECK(wubu_experibuf_add(&r, p3, 1, "512K-SRT", 25.0, 0.2, 0) == 0, "replay add p3");
    CHECK(wubu_experibuf_size(&r) == 3, "replay size == 3");
    wubu_transition_t t;
    CHECK(wubu_experibuf_sample(&r, 42, &t) == 0, "replay sample");
    CHECK(t.tok_s > 0.0, "sampled transition has data");
    /* Fill beyond capacity to test reservoir eviction */
    double pfill[15] = {99.0};
    for (int i = 0; i < WUBU_REPLAY_CAPACITY + 10; i++)
        wubu_experibuf_add(&r, pfill, 1, "fill", 99.0, 0.5, 1);
    CHECK(wubu_experibuf_size(&r) == WUBU_REPLAY_CAPACITY, "replay capped at capacity after reservoir");

    /* BB02: EWC consolidation */
    printf("\n=== BB02: EWC Consolidation ===\n");
    wubu_ewc_t e;
    double anchor[15] = {1.0, 2.0, 3.0, 4.0, 5.0};
    CHECK(wubu_ewc_init(&e, anchor, 5, 100.0) == 0, "ewc init");
    double grads[5] = {0.5, 0.0, 3.0, 0.0, 1.0};
    CHECK(wubu_ewc_estimate_fisher(&e, grads, 5) == 0, "ewc estimate fisher");
    CHECK(e.fisher[0] > 0.0, "fisher[0] > 0 (was gradient 0.5)");
    CHECK(e.fisher[1] == 0.0, "fisher[1] == 0 (grad was 0.0)");
    CHECK(e.fisher[2] > e.fisher[0], "fisher[2] > fisher[0] (grad 3.0 > 0.5)");
    CHECK(wubu_ewc_is_stable(&e, 0) == 0, "dim 0 not stable (fisher 0.25 < 1.0 threshold)");
    CHECK(wubu_ewc_is_stable(&e, 1) == 0, "dim 1 not stable (fisher == 0)");
    double penalty = wubu_ewc_penalty(&e, anchor, 5);
    CHECK(penalty == 0.0, "penalty == 0 at anchor");
    double shifted[15] = {2.0, 2.0, 3.0, 4.0, 5.0};
    penalty = wubu_ewc_penalty(&e, shifted, 5);
    CHECK(penalty > 0.0, "penalty > 0 after shifting stable dim 0");
    CHECK(wubu_ewc_stable_count(&e, 1.0) >= 1, "at least 1 stable dim after fisher estimation");

    /* BB03: Task boundary detection */
    printf("\n=== BB03: Task Boundary Detection ===\n");
    wubu_taskbd_t tb;
    CHECK(wubu_taskbd_init(&tb, 2.0) == 0, "taskbd init");
    /* Fill baseline with steady ~27 tok/s */
    int boundary = 0;
    for (int i = 0; i < WUBU_TASKBD_WINDOW; i++)
        boundary = wubu_taskbd_observe(&tb, 27.0);
    CHECK(boundary == 0, "no boundary during steady state");
    CHECK(tb.baseline_ready == 1, "baseline established");
    /* Inject a sudden drop: 27 → 5 (huge divergence) */
    boundary = 0;
    for (int i = 0; i < WUBU_TASKBD_WINDOW; i++)
        boundary = wubu_taskbd_observe(&tb, 5.0);
    CHECK(boundary == 1, "boundary detected after tok/s drop to 5");
    /* Steady again: no boundary */
    wubu_taskbd_t tb2;
    wubu_taskbd_init(&tb2, 2.0);
    boundary = 0;
    for (int i = 0; i < WUBU_TASKBD_WINDOW; i++)
        boundary = wubu_taskbd_observe(&tb2, 27.0);
    CHECK(boundary == 0, "no boundary when steady");

    /* BB04: Knowledge distillation */
    printf("\n=== BB04: Knowledge Distillation ===\n");
    wubu_distill_t d;
    double teacher[15] = {10.0, 20.0, 30.0};
    CHECK(wubu_distill_set_teacher(&d, teacher, 3, 2.0) == 0, "distill set teacher");
    CHECK(d.has_teacher == 1, "teacher stored");
    double student_same[15] = {10.0, 20.0, 30.0};
    double kl = wubu_distill_kl_loss(&d, student_same);
    CHECK(kl < 1.0, "KL ≈ 0 when teacher == student");
    double student_diff[15] = {1.0, 1.0, 1.0};
    kl = wubu_distill_kl_loss(&d, student_diff);
    CHECK(kl > 0.0, "KL > 0 when student differs from teacher");
    double total = wubu_distill_total_loss(&d, 100.0, student_same, 0.5);
    CHECK(fabs(total - 100.0) < 0.1, "total_loss ≈ hard_loss when KL≈0");
    total = wubu_distill_total_loss(&d, 100.0, student_diff, 0.5);
    CHECK(total > 100.0, "total_loss > hard_loss when KL>0");

    /* Integration: EWC protects dims, replay stores past configs,
       taskbd detects boundary, distill keeps old config as teacher */
    printf("\n=== Integration: Continual Learning Loop ===\n");
    wubu_ewc_t ewc; wubu_ewc_init(&ewc, anchor, 5, 100.0);
    wubu_ewc_estimate_fisher(&ewc, grads, 5);
    /* After consolidation, stable dims should be protected */
    int stable = wubu_ewc_is_stable(&ewc, 0);  /* grad was 0.5 → fisher=0.25 < 1 */
    /* Actually 0.5^2 = 0.25 < 1.0, so NOT stable. Let's check dim 2 (grad 3.0 → fisher 9.0) */
    stable = wubu_ewc_is_stable(&ewc, 2);  /* 3.0^2 = 9.0 >= 1.0 */
    CHECK(stable == 1, "dim 2 stable (fisher 9.0 >= 1.0) — protected from forgetting");

    if (failures > 0) {
        printf("\n%d TEST(S) FAILED\n", failures);
        return 1;
    }
    printf("\nALL CONTINUAL TESTS PASSED\n");
    return 0;
}
