/* test_egress.c -- AGI-EDR network egress control: block exfil (AGIEDR #299).
 * Proves: default-deny, allowlist match (exact + wildcard + port), per-rule
 * byte budget exhausts, global window budget caps, fail-closed on NULL,
 * window reset, denial observability. */
#include "wubu_egress.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_egress_t e;
    CHECK(wubu_egress_init(&e, 1000) == 0, "init ok");

    /* default-deny: nothing allowed before any rule */
    CHECK(wubu_egress_check(&e, "evil.com", 443, 10) == 0, "default-deny unknown host");
    CHECK(wubu_egress_allowed(&e, "evil.com", 443) == 0, "peek unknown = not allowed");

    /* allow api.example.com:443 with 100-byte budget */
    CHECK(wubu_egress_allow(&e, "api.example.com", 443, 100) == 0, "add allow rule");
    CHECK(wubu_egress_check(&e, "api.example.com", 443, 50) == 1, "allowed host within budget");
    CHECK(wubu_egress_check(&e, "api.example.com", 443, 60) == 0, "per-rule budget exhausted");
    /* wrong port denied */
    CHECK(wubu_egress_check(&e, "api.example.com", 80, 10) == 0, "wrong port denied");

    /* wildcard suffix allow */
    wubu_egress_t w;
    wubu_egress_init(&w, 10000);
    wubu_egress_allow(&w, "*.example.com", 0, 500);   /* any port */
    CHECK(wubu_egress_check(&w, "cdn.example.com", 443, 100) == 1, "wildcard subdomain allowed");
    CHECK(wubu_egress_check(&w, "example.com", 443, 100) == 1, "wildcard matches apex too");
    CHECK(wubu_egress_check(&w, "notexample.com", 443, 100) == 0, "wildcard no false match");
    CHECK(wubu_egress_check(&w, "evil.org", 443, 100) == 0, "wildcard denies other domain");

    /* global window budget caps total egress across rules */
    wubu_egress_t g;
    wubu_egress_init(&g, 200);   /* tiny global budget */
    wubu_egress_allow(&g, "a.com", 0, 1000);
    wubu_egress_allow(&g, "b.com", 0, 1000);
    CHECK(wubu_egress_check(&g, "a.com", 80, 150) == 1, "a.com 150 within global");
    CHECK(wubu_egress_check(&g, "b.com", 80, 100) == 0, "global window budget caps b.com");
    /* window reset re-enables */
    wubu_egress_window_reset(&g);
    CHECK(wubu_egress_check(&g, "b.com", 80, 100) == 1, "after window reset b.com allowed");

    /* fail-closed on NULL */
    CHECK(wubu_egress_check(NULL, "x.com", 80, 1) == 0, "NULL controller denies");

    /* denial observability */
    wubu_egress_t d;
    wubu_egress_init(&d, 1000);
    wubu_egress_allow(&d, "ok.com", 0, 10);
    wubu_egress_check(&d, "bad.com", 80, 1);
    wubu_egress_check(&d, "bad2.com", 80, 1);
    CHECK(wubu_egress_denied(&d) == 2, "denial count = 2");

    printf("\n=== test_egress: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
