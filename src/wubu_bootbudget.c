/* wubu_bootbudget.c -- KERNEL #22 boot time budgeted + reported.
 * See wubu_bootbudget.h. */
#include "wubu_bootbudget.h"
#include <string.h>

int wubu_bb_init(wubu_bootbudget_t *b) {
    if (!b) return -1;
    memset(b->phases, 0, sizeof b->phases);
    b->n = 0;
    return 0;
}

int wubu_bb_add(wubu_bootbudget_t *b, const char *name, uint64_t budget) {
    if (!b || !name || b->n >= WUBU_BB_MAX) return -1;
    if (strlen(name) >= 24) return -1;
    strncpy(b->phases[b->n].name, name, 23); b->phases[b->n].name[23] = '\0';
    b->phases[b->n].budget = budget;
    b->phases[b->n].actual = 0;
    b->n++;
    return 0;
}

int wubu_bb_record(wubu_bootbudget_t *b, const char *name, uint64_t actual) {
    if (!b || !name) return -1;
    for (int i = 0; i < b->n; i++) {
        if (strcmp(b->phases[i].name, name) == 0) {
            b->phases[i].actual = actual;
            return actual > b->phases[i].budget ? 1 : 0;   /* 1 = breach */
        }
    }
    return -1;   /* unknown phase */
}

uint64_t wubu_bb_total(const wubu_bootbudget_t *b) {
    if (!b) return 0;
    uint64_t t = 0;
    for (int i = 0; i < b->n; i++) t += b->phases[i].actual;
    return t;
}

uint64_t wubu_bb_total_budget(const wubu_bootbudget_t *b) {
    if (!b) return 0;
    uint64_t t = 0;
    for (int i = 0; i < b->n; i++) t += b->phases[i].budget;
    return t;
}

int wubu_bb_breaches(const wubu_bootbudget_t *b) {
    if (!b) return 0;
    int c = 0;
    for (int i = 0; i < b->n; i++) if (b->phases[i].actual > b->phases[i].budget) c++;
    return c;
}

int wubu_bb_within(const wubu_bootbudget_t *b) {
    if (!b) return 0;
    return wubu_bb_total(b) <= wubu_bb_total_budget(b) ? 1 : 0;
}
