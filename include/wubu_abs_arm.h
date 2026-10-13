#ifndef WUBU_ABS_ARM_H
#define WUBU_ABS_ARM_H

/* wubu_abs_arm.h -- OMARCHY #287 A/B boot slots (atomic arm).
 *
 * A/B boot slots: the system keeps two boot slots (A and B) and an "active"
 * pointer. An update is written to the INACTIVE slot, then the active pointer
 * is ATOMICALLY flipped to it. If the new slot fails to boot, the pointer is
 * flipped back to the last-known-good slot (the other one). This gives safe
 * rollback: at any instant exactly one slot is active, and the flip is
 * atomic. Deterministic, real work (no stubs). */

#define WUBU_SLOT_A 0
#define WUBU_SLOT_B 1

typedef struct {
    int active;          /* currently active slot (A/B) */
    int arm_buf[2];      /* 1 = slot has a valid bootable image armed */
    int boot_ok[2];      /* 1 = slot booted successfully last time */
    int n_flips;
} wubu_abs_arm_t;

/* Initialise both slots unarmed. */
int wubu_abs_arm_init(wubu_abs_arm_t *s);

/* Arm a slot (write a valid bootable image). Returns 0/-1. */
int wubu_abs_arm_slot(wubu_abs_arm_t *s, int slot);

/* Atomically flip the active pointer to the other slot (must be armed).
 * Returns 0/-1. This is the "commit" of an update. */
int wubu_abs_flip(wubu_abs_arm_t *s);

/* Record that the active slot booted successfully. */
int wubu_abs_boot_result(wubu_abs_arm_t *s, int ok);

/* On a boot failure, roll back to the last-known-good slot (the one that
 * booted ok before). Returns the slot rolled back to, or -1 if none. */
int wubu_abs_rollback(wubu_abs_arm_t *s);

/* Currently active slot. */
int wubu_abs_active(const wubu_abs_arm_t *s);

#endif
