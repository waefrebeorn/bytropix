/* test_ssidoc.c -- PATHS #330 safe-superintelligence framework doc.
 * Proves: layer registration, coverage, completeness gate (ALL required),
 * missing-layer listing, fail-closed on NULL. */
#include "wubu_ssidoc.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_ssidoc_t s;
    CHECK(wubu_ssi_init(&s) == 0, "init ok");

    /* the canonical SSI safety layers (matching the modules built in this repo) */
    wubu_ssi_add(&s, "deterministic-controls", "safety enforced in code, not prompts", 1);
    wubu_ssi_add(&s, "cap-authority", "capability-backed authority", 1);
    wubu_ssi_add(&s, "zero-trust", "per-action re-verification", 1);
    wubu_ssi_add(&s, "containment", "anomaly -> containment -> halt", 1);
    wubu_ssi_add(&s, "kill-switch", "human one-way freeze", 1);
    wubu_ssi_add(&s, "supply-chain-attest", "tools verified before use", 1);
    wubu_ssi_add(&s, "rollback", "fail-closed boot/firmware rollback", 1);
    wubu_ssi_add(&s, "self-mod-safety", "DGM invariants + evolution sandbox", 0);  /* NOT yet */

    CHECK(s.n == 8, "8 safety layers");
    /* coverage = 7/8 = 0.875 */
    CHECK(wubu_ssi_coverage(&s) > 0.87 && wubu_ssi_coverage(&s) < 0.88, "coverage = 7/8");
    /* NOT complete (self-mod-safety missing) */
    CHECK(wubu_ssi_complete(&s) == 0, "not SSI-complete (1 layer missing)");

    /* the missing layer is listed */
    char missing[4][WUBU_SSI_NAMELEN];
    int nmiss = wubu_ssi_missing(&s, missing, 4);
    CHECK(nmiss == 1, "1 missing layer");
    CHECK(strcmp(missing[0], "self-mod-safety") == 0, "missing = self-mod-safety");

    /* implement it -> now complete */
    CHECK(wubu_ssi_implement(&s, "self-mod-safety") == 0, "implement self-mod-safety");
    CHECK(wubu_ssi_coverage(&s) == 1.0, "coverage = 1.0");
    CHECK(wubu_ssi_complete(&s) == 1, "SSI-complete (all layers present)");
    CHECK(wubu_ssi_missing(&s, missing, 4) == 0, "0 missing");

    /* fail-closed */
    CHECK(wubu_ssi_init(NULL) == -1, "NULL init -> -1");
    CHECK(wubu_ssi_add(&s, NULL, "x", 1) == -1, "NULL name -> -1");
    CHECK(wubu_ssi_add(&s, "x", NULL, 1) == -1, "NULL purpose -> -1");
    CHECK(wubu_ssi_implement(&s, "nope") == -1, "implement unknown -> -1");
    CHECK(wubu_ssi_coverage(NULL) == 0.0, "NULL coverage -> 0");
    CHECK(wubu_ssi_complete(NULL) == 0, "NULL complete -> 0");
    CHECK(wubu_ssi_missing(&s, NULL, 4) == 0, "NULL missing out -> 0");

    printf("\n=== test_ssidoc: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
