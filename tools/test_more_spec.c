/*
 * test_more_spec.c -- M07/M08/M09/M10/M15/M17/M18/M19/M20 verification.
 */
#include "wubu_more_spec.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_more_spec (M07-M10/M15/M17-M20) ===\n");

    /* M07 REST: small residual accepted. */
    CHECK(wubu_rest_accept(0.001f, 0.01f) == 1, "small resid -> accept");
    CHECK(wubu_rest_accept(0.5f, 0.01f) == 0, "large resid -> reject");

    /* M08 tree restructure: shallow -> restructure; full -> verify. */
    CHECK(wubu_tree_restructure(1, 4) == 1, "shallow -> restructure");
    CHECK(wubu_tree_restructure(4, 4) == 0, "full -> verify");

    /* M09 contrastive: accept only if draft >= ref. */
    CHECK(wubu_contrastive_accept(0.6f, 0.5f) == 1, "draft>=ref -> accept");
    CHECK(wubu_contrastive_accept(0.4f, 0.5f) == 0, "draft<ref -> reject");

    /* M10 distil gate. */
    CHECK(wubu_distil_gate(0.9f, 0.8f) == 1, "quality>=min -> swap");
    CHECK(wubu_distil_gate(0.5f, 0.8f) == 0, "quality<min -> keep");

    /* M15 spec MoE skip. */
    CHECK(wubu_spec_moe_skip(0.01f, 0.1f) == 1, "low route -> skip expert");
    CHECK(wubu_spec_moe_skip(0.5f, 0.1f) == 0, "high route -> keep expert");

    /* M17 cascade. */
    CHECK(wubu_cascade_accept(1) == 1, "draft match -> accept");
    CHECK(wubu_cascade_accept(0) == 0, "draft mismatch -> reject");

    /* M18 swap after patience. */
    int streak = 0;
    wubu_swap_check(&streak, 0.1f, 0.3f, 3);
    wubu_swap_check(&streak, 0.1f, 0.3f, 3);
    CHECK(wubu_swap_check(&streak, 0.1f, 0.3f, 3) == 1, "swap fires after 3 low steps");
    int s2 = 0;
    CHECK(wubu_swap_check(&s2, 0.9f, 0.3f, 3) == 0, "high acceptance -> no swap");

    /* M19 layer resume. */
    CHECK(wubu_layer_resume(5, 10) == 1, "partially streamed -> resume");
    CHECK(wubu_layer_resume(10, 10) == 0, "fully streamed -> no resume");

    /* M20 cascade + early-exit. */
    CHECK(wubu_cascade_earlyexit(1, 2, 5) == 1, "match + shallow -> early-exit");
    CHECK(wubu_cascade_earlyexit(1, 9, 5) == 0, "match + deep -> no early-exit");
    CHECK(wubu_cascade_earlyexit(0, 2, 5) == 0, "no match -> no early-exit");

    if (failures == 0) { printf("ALL MORE-SPEC TESTS PASSED\n"); return 0; }
    printf("%d MORE-SPEC TEST(S) FAILED\n", failures);
    return 1;
}
