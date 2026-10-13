/* test_modelcard.c -- BRAIN #128 math model card (AGI disclosure).
 * Proves: init, add items by kind, count by kind, render, disclosable gate,
 * fail-closed on NULL/bad args. */
#include "wubu_modelcard.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_modelcard_t mc;
    CHECK(wubu_mcard_init(&mc, "WuBu-Math", "1.0", 35, "assist formal math reasoning") == 0, "init ok");

    /* incomplete card is not disclosable */
    CHECK(wubu_mcard_disclosable(&mc) == 0, "empty card not disclosable");

    /* add items */
    CHECK(wubu_mcard_add(&mc, WUBU_MCARD_CAPABILITY, "solves olympiad algebra") == 0, "add capability");
    CHECK(wubu_mcard_add(&mc, WUBU_MCARD_LIMITATION, "no real-time web access") == 0, "add limitation");
    CHECK(wubu_mcard_add(&mc, WUBU_MCARD_EVALUATION, "gauntlet 247/247") == 0, "add evaluation");

    /* still not disclosable? no -- now it is (1 of each) */
    CHECK(wubu_mcard_disclosable(&mc) == 1, "card with 1 of each is disclosable");
    CHECK(wubu_mcard_count(&mc, WUBU_MCARD_CAPABILITY) == 1, "1 capability");
    CHECK(wubu_mcard_count(&mc, WUBU_MCARD_LIMITATION) == 1, "1 limitation");
    CHECK(wubu_mcard_count(&mc, WUBU_MCARD_EVALUATION) == 1, "1 evaluation");

    /* add more */
    wubu_mcard_add(&mc, WUBU_MCARD_CAPABILITY, "proves Lean lemmas");
    wubu_mcard_add(&mc, WUBU_MCARD_LIMITATION, "cannot verify its own proofs unaided");
    CHECK(wubu_mcard_count(&mc, WUBU_MCARD_CAPABILITY) == 2, "2 capabilities");
    CHECK(wubu_mcard_count(&mc, WUBU_MCARD_LIMITATION) == 2, "2 limitations");

    /* render */
    char buf[2048];
    int len = wubu_mcard_render(&mc, buf, sizeof buf);
    CHECK(len > 0, "render produces output");
    CHECK(strstr(buf, "WuBu-Math") != NULL, "render includes name");
    CHECK(strstr(buf, "35M") != NULL, "render includes params");
    CHECK(strstr(buf, "Capabilities") != NULL, "render has Capabilities section");
    CHECK(strstr(buf, "Limitations") != NULL, "render has Limitations section");
    CHECK(strstr(buf, "Evaluation") != NULL, "render has Evaluation section");
    CHECK(strstr(buf, "gauntlet 247/247") != NULL, "render includes evaluation item");

    /* fail-closed */
    CHECK(wubu_mcard_init(NULL, "x", "1", 1, "y") == -1, "NULL init -> -1");
    CHECK(wubu_mcard_add(&mc, 99, "x") == -1, "bad kind -> -1");
    CHECK(wubu_mcard_add(&mc, WUBU_MCARD_CAPABILITY, NULL) == -1, "NULL text -> -1");
    CHECK(wubu_mcard_count(NULL, WUBU_MCARD_CAPABILITY) == 0, "NULL count -> 0");
    CHECK(wubu_mcard_render(&mc, NULL, 100) == -1, "NULL render buf -> -1");
    CHECK(wubu_mcard_disclosable(NULL) == 0, "NULL disclosable -> 0");

    printf("\n=== test_modelcard: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
