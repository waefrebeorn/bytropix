/* wubu_abs_arm.c -- OMARCHY #287 A/B boot slots (atomic arm).
 * See wubu_abs_arm.h. */
#include "wubu_abs_arm.h"

int wubu_abs_arm_init(wubu_abs_arm_t *s) {
    if (!s) return -1;
    s->active = WUBU_SLOT_A;
    s->arm_buf[0] = s->arm_buf[1] = 0;
    s->boot_ok[0] = s->boot_ok[1] = 0;
    s->n_flips = 0;
    return 0;
}

int wubu_abs_arm_slot(wubu_abs_arm_t *s, int slot) {
    if (!s || (slot != WUBU_SLOT_A && slot != WUBU_SLOT_B)) return -1;
    s->arm_buf[slot] = 1;
    return 0;
}

int wubu_abs_flip(wubu_abs_arm_t *s) {
    if (!s) return -1;
    int next = 1 - s->active;   /* the other slot */
    if (!s->arm_buf[next]) return -1;   /* target slot must be armed */
    s->active = next;           /* atomic flip */
    s->n_flips++;
    return 0;
}

int wubu_abs_boot_result(wubu_abs_arm_t *s, int ok) {
    if (!s) return -1;
    s->boot_ok[s->active] = ok ? 1 : 0;
    return 0;
}

int wubu_abs_rollback(wubu_abs_arm_t *s) {
    if (!s) return -1;
    int other = 1 - s->active;
    /* roll back to the other slot if it's armed and booted ok before */
    if (s->arm_buf[other] && s->boot_ok[other]) {
        s->active = other;
        return other;
    }
    return -1;
}

int wubu_abs_active(const wubu_abs_arm_t *s) {
    return s ? s->active : -1;
}
