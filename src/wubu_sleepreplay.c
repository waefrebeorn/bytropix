/* wubu_sleepreplay.c -- BRAIN #125 memory consolidation sleep-replay.
 * See wubu_sleepreplay.h. */
#include "wubu_sleepreplay.h"
#include <string.h>

int wubu_sr_init(wubu_sleepreplay_t *s, double decay, double replay_gain) {
    if (!s) return -1;
    memset(s->eps, 0, sizeof s->eps);
    s->n = 0;
    s->decay = (decay >= 0 && decay <= 1) ? decay : 0.1;
    s->replay_gain = (replay_gain >= 0 && replay_gain <= 1) ? replay_gain : 0.3;
    return 0;
}

int wubu_sr_record(wubu_sleepreplay_t *s, int id) {
    if (!s || s->n >= WUBU_SR_MAX) return -1;
    s->eps[s->n].id = id;
    s->eps[s->n].strength = 0.5;   /* fresh episodic memory */
    s->eps[s->n].replay_count = 0;
    s->n++;
    return 0;
}

static wubu_sr_ep_t *find(wubu_sleepreplay_t *s, int id) {
    for (int i = 0; i < s->n; i++) if (s->eps[i].id == id) return &s->eps[i];
    return NULL;
}

int wubu_sr_sleep(wubu_sleepreplay_t *s, const int *replay_ids, int n) {
    if (!s) return -1;
    /* 1. all episodes decay */
    for (int i = 0; i < s->n; i++) {
        s->eps[i].strength -= s->decay;
        if (s->eps[i].strength < 0) s->eps[i].strength = 0;
    }
    /* 2. replayed episodes gain strength (capped at 1) */
    if (replay_ids) {
        for (int j = 0; j < n; j++) {
            wubu_sr_ep_t *e = find(s, replay_ids[j]);
            if (e) {
                e->strength += s->replay_gain;
                if (e->strength > 1.0) e->strength = 1.0;
                e->replay_count++;
            }
        }
    }
    /* 3. count consolidated (strength >= 0.8) */
    int consolidated = 0;
    for (int i = 0; i < s->n; i++) if (s->eps[i].strength >= 0.8) consolidated++;
    return consolidated;
}

double wubu_sr_strength(const wubu_sleepreplay_t *s, int id) {
    if (!s) return 0;
    for (int i = 0; i < s->n; i++) if (s->eps[i].id == id) return s->eps[i].strength;
    return 0;
}

int wubu_sr_consolidated(const wubu_sleepreplay_t *s, int id) {
    if (!s) return 0;
    return wubu_sr_strength(s, id) >= 0.8 ? 1 : 0;
}

int wubu_sr_n_consolidated(const wubu_sleepreplay_t *s) {
    if (!s) return 0;
    int c = 0;
    for (int i = 0; i < s->n; i++) if (s->eps[i].strength >= 0.8) c++;
    return c;
}
