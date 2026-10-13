/* wubu_consolidate.c -- PATHS #320 memory consolidation (sleep replay).
 * See wubu_consolidate.h. */
#include "wubu_consolidate.h"
#include <string.h>

int wubu_consol_init(wubu_consolidate_t *c, double salience_threshold) {
    if (!c) return -1;
    memset(c->mem, 0, sizeof c->mem);
    c->n = 0;
    c->salience_threshold = salience_threshold >= 0 ? salience_threshold : 0.1;
    return 0;
}

int wubu_consol_add(wubu_consolidate_t *c, int id, double salience) {
    if (!c || c->n >= WUBU_CONSOL_MAX) return -1;
    if (salience < 0) salience = 0;
    if (salience > 1) salience = 1;
    wubu_consol_mem_t *m = &c->mem[c->n];
    m->id = id;
    m->salience = salience;
    m->replay_count = 0;
    m->strength = salience;   /* initial strength = salience */
    m->pruned = 0;
    return c->n++;
}

int wubu_consol_sleep(wubu_consolidate_t *c, int replay_topk) {
    if (!c || replay_topk < 0) return -1;
    /* mark the top-K most salient live memories and strengthen them */
    char marked[WUBU_CONSOL_MAX] = {0};
    for (int k = 0; k < replay_topk; k++) {
        int best = -1;
        double best_sal = -1;
        for (int i = 0; i < c->n; i++) {
            if (c->mem[i].pruned || marked[i]) continue;
            if (c->mem[i].salience > best_sal) { best_sal = c->mem[i].salience; best = i; }
        }
        if (best < 0) break;
        marked[best] = 1;
        c->mem[best].replay_count++;
        c->mem[best].strength += 0.1 * c->mem[best].salience;   /* strengthen */
        if (c->mem[best].strength > 1.0) c->mem[best].strength = 1.0;
    }
    /* prune: live memories below threshold that were NOT replayed this pass */
    int pruned = 0;
    for (int i = 0; i < c->n; i++) {
        if (c->mem[i].pruned) continue;
        if (!marked[i] && c->mem[i].salience < c->salience_threshold &&
            c->mem[i].replay_count == 0) {
            c->mem[i].pruned = 1;
            pruned++;
        }
    }
    return pruned;
}

double wubu_consol_strength(const wubu_consolidate_t *c, int id) {
    if (!c) return 0.0;
    for (int i = 0; i < c->n; i++)
        if (c->mem[i].id == id && !c->mem[i].pruned) return c->mem[i].strength;
    return 0.0;
}

int wubu_consol_live_count(const wubu_consolidate_t *c) {
    if (!c) return 0;
    int live = 0;
    for (int i = 0; i < c->n; i++)
        if (!c->mem[i].pruned) live++;
    return live;
}
