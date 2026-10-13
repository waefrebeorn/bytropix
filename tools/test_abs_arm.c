/* test_abs_arm.c -- OMARCHY #287 A/B boot slots (atomic arm).
 * Proves: arm slot, atomic flip, boot result, rollback on failure,
 * flip requires armed slot, fail-closed on NULL. */
#include "wubu_abs_arm.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_abs_arm_t s;
    CHECK(wubu_abs_arm_init(&s) == 0, "init ok (both slots unarmed)");
    CHECK(wubu_abs_active(&s) == WUBU_SLOT_A, "active = A");

    /* can't flip to B: B not armed */
    CHECK(wubu_abs_flip(&s) == -1, "flip to unarmed B -> -1");

    /* arm B (the update), flip to it */
    CHECK(wubu_abs_arm_slot(&s, WUBU_SLOT_B) == 0, "arm B");
    CHECK(wubu_abs_flip(&s) == 0, "flip to B ok");
    CHECK(wubu_abs_active(&s) == WUBU_SLOT_B, "active = B");

    /* B boots ok */
    CHECK(wubu_abs_boot_result(&s, 1) == 0, "B booted ok");
    CHECK(s.boot_ok[WUBU_SLOT_B] == 1, "B marked boot-ok");

    /* now arm A with a NEW update and flip to it */
    CHECK(wubu_abs_arm_slot(&s, WUBU_SLOT_A) == 0, "arm A (new update)");
    CHECK(wubu_abs_flip(&s) == 0, "flip to A ok");
    CHECK(wubu_abs_active(&s) == WUBU_SLOT_A, "active = A");

    /* A FAILS to boot */
    CHECK(wubu_abs_boot_result(&s, 0) == 0, "A boot failed");
    /* roll back to B (last-known-good) */
    int rb = wubu_abs_rollback(&s);
    CHECK(rb == WUBU_SLOT_B, "rolled back to B");
    CHECK(wubu_abs_active(&s) == WUBU_SLOT_B, "active = B after rollback");

    /* rollback when the good slot isn't boot-ok -> -1 */
    wubu_abs_arm_t s2;
    wubu_abs_arm_init(&s2);
    wubu_abs_arm_slot(&s2, WUBU_SLOT_B);
    /* B armed but never booted ok; A is active but boot_ok[A]=0 */
    CHECK(wubu_abs_rollback(&s2) == -1, "rollback with no known-good slot -> -1");

    /* flip requires a valid slot */
    CHECK(wubu_abs_arm_slot(&s2, 99) == -1, "arm invalid slot -> -1");

    /* fail-closed */
    CHECK(wubu_abs_arm_init(NULL) == -1, "NULL init -> -1");
    CHECK(wubu_abs_flip(NULL) == -1, "NULL flip -> -1");
    CHECK(wubu_abs_rollback(NULL) == -1, "NULL rollback -> -1");
    CHECK(wubu_abs_active(NULL) == -1, "NULL active -> -1");

    printf("\n=== test_abs_arm: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
