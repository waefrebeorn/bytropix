/* test_irqaff.c -- KERNEL #21 IRQ affinity settable via /n.
 * Proves: register defaults to all CPUs, set affinity, target cpu, pinned
 * detection, invalid masks rejected, fail-closed on NULL. */
#include "wubu_irqaff.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_irqaff_t a;
    CHECK(wubu_irqaff_init(&a, 8) == 0, "init ok (8 cpus)");
    CHECK(wubu_irqaff_register(&a, 32) == 0, "register irq 32");
    CHECK(wubu_irqaff_register(&a, 33) == 0, "register irq 33");

    /* default: all 8 cpus */
    CHECK(wubu_irqaff_get(&a, 32) == 0xFFULL, "default affinity = all 8 cpus");
    CHECK(wubu_irqaff_target_cpu(&a, 32) == 0, "target cpu 0 (lowest)");
    CHECK(wubu_irqaff_pinned(&a, 32) == 0, "not pinned (8 cpus)");

    /* pin irq 32 to cpu 3 */
    CHECK(wubu_irqaff_set(&a, 32, 1ULL << 3) == 0, "set affinity cpu 3");
    CHECK(wubu_irqaff_get(&a, 32) == (1ULL << 3), "affinity = cpu 3");
    CHECK(wubu_irqaff_target_cpu(&a, 32) == 3, "target cpu 3");
    CHECK(wubu_irqaff_pinned(&a, 32) == 1, "pinned to single cpu");

    /* pin irq 33 to cpus {1,2} */
    CHECK(wubu_irqaff_set(&a, 33, (1ULL<<1)|(1ULL<<2)) == 0, "set affinity {1,2}");
    CHECK(wubu_irqaff_target_cpu(&a, 33) == 1, "target cpu 1 (lowest of {1,2})");
    CHECK(wubu_irqaff_pinned(&a, 33) == 0, "not pinned (2 cpus)");

    /* invalid masks rejected */
    CHECK(wubu_irqaff_set(&a, 32, 0) == -1, "empty mask -> -1");
    CHECK(wubu_irqaff_set(&a, 32, 1ULL << 9) == -1, "offline cpu (bit 9) -> -1");
    CHECK(wubu_irqaff_set(&a, 99, 1) == -1, "unknown irq -> -1");

    /* duplicate register rejected */
    CHECK(wubu_irqaff_register(&a, 32) == -1, "duplicate register -> -1");

    /* unknown irq get -> 0 */
    CHECK(wubu_irqaff_get(&a, 99) == 0, "unknown irq get -> 0");
    CHECK(wubu_irqaff_target_cpu(&a, 99) == -1, "unknown irq target -> -1");

    /* fail-closed on NULL */
    CHECK(wubu_irqaff_init(NULL, 8) == -1, "NULL init -> -1");
    CHECK(wubu_irqaff_init(&a, 0) == -1, "zero cpus -> -1");
    CHECK(wubu_irqaff_set(NULL, 32, 1) == -1, "NULL set -> -1");
    CHECK(wubu_irqaff_get(NULL, 32) == 0, "NULL get -> 0");
    CHECK(wubu_irqaff_target_cpu(NULL, 32) == -1, "NULL target -> -1");

    printf("\n=== test_irqaff: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
