/* test_isa_eff.c -- PATHS #323 custom-ISA compiler efficiency path.
 * Proves: custom path wins (fewer instrs + no ABI overhead), speedup > 1,
 * cycles computed correctly, fail-closed on NULL/bad params. */
#include "wubu_isa_eff.h"
#include <stdio.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)
#define NEAR(a,b,t) (fabs((a)-(b)) < (t))

int main(void) {
    wubu_isa_eff_t e;
    /* custom: 100 instrs @ 1.0 cpi, no overhead = 100 cycles
     * portable: 130 instrs @ 1.2 cpi + 50 overhead = 206 cycles */
    CHECK(wubu_isa_eff_compare(&e, "matmul",
                               100, 1.0, 130, 1.2, 50.0) == 0, "compare ok");

    /* cycles */
    CHECK(NEAR(wubu_isa_eff_cycles(&e.custom), 100.0, 1e-9), "custom cycles = 100");
    CHECK(NEAR(wubu_isa_eff_cycles(&e.portable), 130*1.2 + 50, 1e-9), "portable cycles = 206");

    /* custom wins */
    CHECK(wubu_isa_eff_custom_wins(&e) == 1, "custom-ISA path wins");
    CHECK(NEAR(wubu_isa_eff_speedup(&e), 206.0/100.0, 1e-9), "speedup = 2.06x");

    /* the win comes from BOTH fewer instrs AND no ABI overhead. Isolate the
     * ABI-overhead contribution: same instrs/cpi, portable pays overhead. */
    wubu_isa_eff_t abi;
    wubu_isa_eff_compare(&abi, "same-code", 100, 1.0, 100, 1.0, 40.0);
    CHECK(wubu_isa_eff_custom_wins(&abi) == 1, "same code: custom wins via no ABI overhead");
    CHECK(NEAR(wubu_isa_eff_speedup(&abi), 140.0/100.0, 1e-9), "ABI overhead alone = 1.4x");

    /* a path where portable is somehow tighter -> custom doesn't win */
    wubu_isa_eff_t lose;
    wubu_isa_eff_compare(&lose, "lose", 200, 1.0, 100, 1.0, 0.0);
    CHECK(wubu_isa_eff_custom_wins(&lose) == 0, "custom loses when it has more instrs");

    /* fail-closed */
    CHECK(wubu_isa_eff_compare(&e, "bad", -1, 1.0, 100, 1.0, 0) == -1, "negative instrs -> -1");
    CHECK(wubu_isa_eff_compare(&e, "bad", 100, 0.0, 100, 1.0, 0) == -1, "zero cpi -> -1");
    CHECK(wubu_isa_eff_compare(NULL, "x", 1, 1, 1, 1, 0) == -1, "NULL compare -> -1");
    CHECK(wubu_isa_eff_custom_wins(NULL) == 0, "NULL custom_wins -> 0");
    CHECK(wubu_isa_eff_speedup(NULL) == 0.0, "NULL speedup -> 0");
    CHECK(wubu_isa_eff_cycles(NULL) == 0.0, "NULL cycles -> 0");

    printf("\n=== test_isa_eff: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
