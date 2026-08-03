/* wubu_priority.h -- the AGI's institutional memory (model-level).
 *
 * The amoeba grows/shrinks the body; the priority store remembers what
 * it learned: per-block importance (BI), a mutation ledger (grow/shrink
 * events with the outcome), and a saved checkpoint pointer. Every
 * mutation consults this store FIRST so the amoeba never re-shrinks a
 * layer that proved critical, never re-grows a layer that proved
 * redundant, and can roll back to the last good state.
 *
 * The ledger is the "prestige ledger" of the model itself — the
 * survival record that makes self-improvement directional instead of
 * random.
 */
#ifndef WUBU_PRIORITY_H
#define WUBU_PRIORITY_H

#include <stdint.h>

#define WUBU_PRI_MAX_LAYERS 64
#define WUBU_PRI_MAX_EVENTS 256

typedef enum {
    WUBU_PRI_EVT_GROW = 0,
    WUBU_PRI_EVT_SHRINK = 1,
    WUBU_PRI_EVT_TRAIN = 2,
    WUBU_PRI_EVT_ROLLBACK = 3
} wubu_pri_event_kind_t;

typedef struct {
    wubu_pri_event_kind_t kind;
    int      layer;          /* which layer the event touched */
    float    loss_before;    /* measured before the event */
    float    loss_after;     /* measured after the event */
    uint64_t step;           /* global training step when it happened */
    int      accepted;       /* 1 = mutation kept, 0 = rolled back */
} wubu_pri_event_t;

typedef struct {
    float block_importance[WUBU_PRI_MAX_LAYERS]; /* last BI per layer */
    int   n_layers;
    int   mutation_count;    /* total grow/shrink events accepted */
    int   rollback_count;    /* total rollbacks */
    uint64_t step;           /* last training step seen */
    wubu_pri_event_t events[WUBU_PRI_MAX_EVENTS];
    int   n_events;
    char  checkpoint_path[256]; /* last-good checkpoint */
    int   initialized;
} wubu_priority_t;

/* Init/clear the store. */
void wubu_priority_init(wubu_priority_t *p);

/* Record the latest BI snapshot (call after wubu_bi_compute). */
int wubu_priority_set_bi(wubu_priority_t *p, const float *bis, int n_layers);

/* Should we shrink layer l? Never if its BI is above the "critical"
 * bar (it proved important) or if a shrink of it was rolled back. */
int wubu_priority_should_shrink(const wubu_priority_t *p, int layer,
                                float bi_threshold, float critical_bar);

/* Should we grow layer l? Never if a grow of it was rolled back. */
int wubu_priority_should_grow(const wubu_priority_t *p, int layer);

/* Record a mutation event + its outcome. */
int wubu_priority_log_event(wubu_priority_t *p, wubu_pri_event_kind_t kind,
                            int layer, float loss_before, float loss_after,
                            int accepted);

/* Was a mutation of this kind+layer ever rolled back? (the shame list) */
int wubu_priority_was_rolled_back(const wubu_priority_t *p,
                                  wubu_pri_event_kind_t kind, int layer);

/* Persist the store next to the checkpoint (safetensors-style sidecar). */
int wubu_priority_save(const wubu_priority_t *p, const char *path);
int wubu_priority_load(wubu_priority_t *p, const char *path);

#endif /* WUBU_PRIORITY_H */
