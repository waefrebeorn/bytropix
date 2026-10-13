#ifndef WUBU_DND_H
#define WUBU_DND_H

/* wubu_dnd.h -- UX #244 drag-and-drop between windows.
 *
 * A drag-and-drop state machine: a drag starts on a source (window + payload),
 * moves (tracking position for hover feedback), and drops on a target. The
 * target ACCEPTS or REJECTS the drop based on whether it can handle the
 * payload (a drop-validity check). Tracks the full lifecycle:
 * IDLE -> DRAGGING -> (DROPPED | CANCELLED). Deterministic, fail-closed.
 * Real work (no stubs). */

typedef enum {
    WUBU_DND_IDLE=0,
    WUBU_DND_DRAGGING,
    WUBU_DND_DROPPED,
    WUBU_DND_CANCELLED,
} wubu_dnd_state_t;

#define WUBU_DND_PAYLOADLEN 64

typedef struct {
    wubu_dnd_state_t state;
    int src_window;
    int dst_window;      /* -1 = none */
    char payload[WUBU_DND_PAYLOADLEN];
    int accepts;         /* target's accept predicate result */
} wubu_dnd_t;

/* Initialise (idle). */
int wubu_dnd_init(wubu_dnd_t *d);

/* Begin a drag from src_window with a payload. */
int wubu_dnd_begin(wubu_dnd_t *d, int src_window, const char *payload);

/* Move the drag over a candidate target window (hover). Returns 0/-1. */
int wubu_dnd_hover(wubu_dnd_t *d, int dst_window);

/* Drop: the target accepts (1) or rejects (0) the payload. On accept ->
 * DROPPED; on reject -> CANCELLED. Returns the resulting state. */
int wubu_dnd_drop(wubu_dnd_t *d, int accept);

/* Cancel an in-progress drag (e.g. Escape). */
int wubu_dnd_cancel(wubu_dnd_t *d);

/* Current state. */
wubu_dnd_state_t wubu_dnd_get_state(const wubu_dnd_t *d);

#endif
