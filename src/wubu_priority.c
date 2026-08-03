/* wubu_priority.c -- the AGI's institutional memory (model-level).
 *
 * See wubu_priority.h. The store makes amoeba mutations directional:
 * the shame list (rolled-back events) prevents repeating failures, the
 * BI snapshot prevents re-shrinking critical layers, and the ledger
 * gives the DGM oracle the survival record it needs to judge whether a
 * mutation made the organism better.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_priority.h"

void wubu_priority_init(wubu_priority_t *p)
{
    if (!p) return;
    memset(p, 0, sizeof *p);
    p->initialized = 1;
    p->checkpoint_path[0] = '\0';
}

int wubu_priority_set_bi(wubu_priority_t *p, const float *bis, int n_layers)
{
    if (!p || !bis || n_layers < 1 || n_layers > WUBU_PRI_MAX_LAYERS)
        return -1;
    memcpy(p->block_importance, bis, (size_t)n_layers * sizeof(float));
    p->n_layers = n_layers;
    return 0;
}

int wubu_priority_should_shrink(const wubu_priority_t *p, int layer,
                                float bi_threshold, float critical_bar)
{
    if (!p || layer < 0 || layer >= p->n_layers) return 0;
    float bi = p->block_importance[layer];
    /* never shrink a layer that proved critical */
    if (bi >= critical_bar) return 0;
    /* never shrink a layer below the "redundant" bar */
    if (bi > bi_threshold) return 0;
    /* never repeat a rolled-back shrink */
    if (wubu_priority_was_rolled_back(p, WUBU_PRI_EVT_SHRINK, layer)) return 0;
    return 1;
}

int wubu_priority_should_grow(const wubu_priority_t *p, int layer)
{
    if (!p || layer < 0 || layer >= p->n_layers) return 0;
    if (wubu_priority_was_rolled_back(p, WUBU_PRI_EVT_GROW, layer)) return 0;
    return 1;
}

int wubu_priority_log_event(wubu_priority_t *p, wubu_pri_event_kind_t kind,
                            int layer, float loss_before, float loss_after,
                            int accepted)
{
    if (!p || !p->initialized || p->n_events >= WUBU_PRI_MAX_EVENTS)
        return -1;
    wubu_pri_event_t *e = &p->events[p->n_events++];
    e->kind = kind;
    e->layer = layer;
    e->loss_before = loss_before;
    e->loss_after = loss_after;
    e->step = p->step;
    e->accepted = accepted;
    if (kind == WUBU_PRI_EVT_GROW || kind == WUBU_PRI_EVT_SHRINK) {
        if (accepted) p->mutation_count++;
        else p->rollback_count++;
    }
    return 0;
}

int wubu_priority_was_rolled_back(const wubu_priority_t *p,
                                  wubu_pri_event_kind_t kind, int layer)
{
    if (!p) return 0;
    for (int i = 0; i < p->n_events; i++) {
        const wubu_pri_event_t *e = &p->events[i];
        if (e->kind == kind && e->layer == layer && !e->accepted)
            return 1;
    }
    return 0;
}

int wubu_priority_save(const wubu_priority_t *p, const char *path)
{
    if (!p || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t w = fwrite(p, sizeof *p, 1, f);
    fclose(f);
    return (w == 1) ? 0 : -1;
}

int wubu_priority_load(wubu_priority_t *p, const char *path)
{
    if (!p || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t r = fread(p, sizeof *p, 1, f);
    fclose(f);
    if (r != 1) return -1;
    /* guard against garbage: the store must look initialized */
    if (!p->initialized || p->n_layers < 0 || p->n_layers > WUBU_PRI_MAX_LAYERS
        || p->n_events < 0 || p->n_events > WUBU_PRI_MAX_EVENTS) {
        memset(p, 0, sizeof *p);
        p->initialized = 1;
        return -1;
    }
    return 0;
}
