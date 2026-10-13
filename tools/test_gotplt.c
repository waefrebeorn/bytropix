/* test_gotplt.c -- RUNTIME #346 dynamic linker GOT/PLT relocation.
 * Proves: BIND_NOW resolves all eagerly, LAZY resolves data + defers funcs to
 * resolver, first-call bind resolves a func, GOT/PLT values, fail-closed. */
#include "wubu_gotplt.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    /* ---- BIND_NOW: eager resolution ---- */
    wubu_gotplt_t g;
    CHECK(wubu_gp_init(&g, WUBU_GP_NOW, 0xDEAD0000ULL) == 0, "init BIND_NOW");
    wubu_gp_add_sym(&g, "malloc", 0x1000ULL);
    wubu_gp_add_sym(&g, "errno",  0x2000ULL);
    wubu_gp_add_reloc(&g, "malloc", 1);   /* func */
    wubu_gp_add_reloc(&g, "errno",  0);   /* data */
    int n = wubu_gp_link(&g);
    CHECK(n == 2, "BIND_NOW: 2 resolved eagerly");
    CHECK(wubu_gp_is_bound(&g, "malloc") == 1, "malloc bound (func)");
    CHECK(wubu_gp_is_bound(&g, "errno") == 1, "errno bound (data)");
    /* malloc GOT -> 0x1000, PLT -> 0x1000 */
    CHECK(g.relocs[0].got == 0x1000ULL, "malloc GOT = symbol addr");
    CHECK(g.relocs[0].plt == 0x1000ULL, "malloc PLT = symbol addr");
    /* errno GOT -> 0x2000, no PLT (data) */
    CHECK(g.relocs[1].got == 0x2000ULL, "errno GOT = symbol addr");
    CHECK(g.relocs[1].plt == 0, "errno PLT = 0 (data has no PLT)");

    /* ---- LAZY: data resolved, funcs deferred to resolver ---- */
    wubu_gotplt_t l;
    CHECK(wubu_gp_init(&l, WUBU_GP_LAZY, 0xDEAD0000ULL) == 0, "init LAZY");
    wubu_gp_add_sym(&l, "malloc", 0x1000ULL);
    wubu_gp_add_sym(&l, "errno",  0x2000ULL);
    wubu_gp_add_reloc(&l, "malloc", 1);
    wubu_gp_add_reloc(&l, "errno",  0);
    int nl = wubu_gp_link(&l);
    CHECK(nl == 1, "LAZY: only data (errno) resolved at link");
    CHECK(wubu_gp_is_bound(&l, "errno") == 1, "errno bound at link (data)");
    CHECK(wubu_gp_is_bound(&l, "malloc") == 0, "malloc NOT bound (lazy)");
    /* lazy func GOT -> resolver */
    CHECK(l.relocs[0].got == 0xDEAD0000ULL, "malloc GOT -> resolver (lazy)");
    CHECK(l.relocs[0].plt == 0xDEAD0000ULL, "malloc PLT -> resolver (lazy)");

    /* first call binds the func */
    CHECK(wubu_gp_bind(&l, "malloc") == 0, "bind malloc on first call");
    CHECK(wubu_gp_is_bound(&l, "malloc") == 1, "malloc bound after first call");
    CHECK(l.relocs[0].got == 0x1000ULL, "malloc GOT now = symbol addr");
    CHECK(l.relocs[0].plt == 0x1000ULL, "malloc PLT now = symbol addr");

    /* cannot bind a data symbol or unknown */
    CHECK(wubu_gp_bind(&l, "errno") == -1, "bind data symbol -> -1");
    CHECK(wubu_gp_bind(&l, "nope") == -1, "bind unknown -> -1");

    /* undefined symbol left unbound */
    wubu_gotplt_t u;
    wubu_gp_init(&u, WUBU_GP_NOW, 0);
    wubu_gp_add_reloc(&u, "missing", 1);
    CHECK(wubu_gp_link(&u) == 0, "undefined symbol not resolved");
    CHECK(wubu_gp_is_bound(&u, "missing") == 0, "undefined symbol unbound");

    /* fail-closed on NULL */
    CHECK(wubu_gp_init(NULL, WUBU_GP_NOW, 0) == -1, "NULL init -> -1");
    CHECK(wubu_gp_link(NULL) == -1, "NULL link -> -1");
    CHECK(wubu_gp_bind(NULL, "x") == -1, "NULL bind -> -1");
    CHECK(wubu_gp_is_bound(NULL, "x") == 0, "NULL is_bound -> 0");

    printf("\n=== test_gotplt: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
