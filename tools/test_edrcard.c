/* test_edrcard.c -- AGIEDR #311 AGI-EDR model card + disclosure.
 * Proves: init, add by kind, count, render, disclosable honesty gate
 * (requires a blind spot), fail-closed on NULL. */
#include "wubu_edrcard.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_edrcard_t c;
    CHECK(wubu_edr_init(&c, "untrusted tool output + retrieved docs") == 0, "init ok");
    CHECK(wubu_edr_disclosable(&c) == 0, "empty card not disclosable");

    /* detections only -- still not disclosable (no blind spot disclosed) */
    wubu_edr_add(&c, WUBU_EDR_DETECTS, "prompt injection in tool output");
    wubu_edr_add(&c, WUBU_EDR_DETECTS, "privilege escalation attempts");
    CHECK(wubu_edr_disclosable(&c) == 0, "detections but NO blind spot -> not disclosable");

    /* add a blind spot -> now disclosable (honest) */
    wubu_edr_add(&c, WUBU_EDR_BLINDSPOT, "cannot detect side channels via timing");
    CHECK(wubu_edr_disclosable(&c) == 1, "detection + blind spot -> disclosable");
    CHECK(wubu_edr_count(&c, WUBU_EDR_DETECTS) == 2, "2 detections");
    CHECK(wubu_edr_count(&c, WUBU_EDR_BLINDSPOT) == 1, "1 blind spot");

    /* add a response action */
    wubu_edr_add(&c, WUBU_EDR_RESPONSE, "kill + contain + rollback");
    CHECK(wubu_edr_count(&c, WUBU_EDR_RESPONSE) == 1, "1 response");

    /* render */
    char buf[2048];
    int len = wubu_edr_render(&c, buf, sizeof buf);
    CHECK(len > 0, "render ok");
    CHECK(strstr(buf, "Threat model") != NULL, "render has threat model");
    CHECK(strstr(buf, "Detects") != NULL, "render has Detects");
    CHECK(strstr(buf, "blind spots") != NULL, "render has blind spots");
    CHECK(strstr(buf, "Response actions") != NULL, "render has Response");
    CHECK(strstr(buf, "prompt injection") != NULL, "render includes a detection");
    CHECK(strstr(buf, "timing") != NULL, "render includes the blind spot");

    /* fail-closed */
    CHECK(wubu_edr_init(NULL, "x") == -1, "NULL init -> -1");
    CHECK(wubu_edr_add(&c, 99, "x") == -1, "bad kind -> -1");
    CHECK(wubu_edr_add(&c, WUBU_EDR_DETECTS, NULL) == -1, "NULL text -> -1");
    CHECK(wubu_edr_count(NULL, WUBU_EDR_DETECTS) == 0, "NULL count -> 0");
    CHECK(wubu_edr_render(&c, NULL, 100) == -1, "NULL buf -> -1");
    CHECK(wubu_edr_disclosable(NULL) == 0, "NULL disclosable -> 0");

    printf("\n=== test_edrcard: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
