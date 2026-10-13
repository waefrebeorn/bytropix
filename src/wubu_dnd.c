/* wubu_dnd.c -- UX #244 drag-and-drop between windows.
 * See wubu_dnd.h. */
#include "wubu_dnd.h"
#include <string.h>

int wubu_dnd_init(wubu_dnd_t *d) {
    if (!d) return -1;
    d->state = WUBU_DND_IDLE;
    d->src_window = -1;
    d->dst_window = -1;
    d->payload[0] = '\0';
    d->accepts = 0;
    return 0;
}

int wubu_dnd_begin(wubu_dnd_t *d, int src_window, const char *payload) {
    if (!d || !payload) return -1;
    if (strlen(payload) >= WUBU_DND_PAYLOADLEN) return -1;
    d->state = WUBU_DND_DRAGGING;
    d->src_window = src_window;
    d->dst_window = -1;
    strncpy(d->payload, payload, WUBU_DND_PAYLOADLEN - 1);
    d->payload[WUBU_DND_PAYLOADLEN - 1] = '\0';
    d->accepts = 0;
    return 0;
}

int wubu_dnd_hover(wubu_dnd_t *d, int dst_window) {
    if (!d || d->state != WUBU_DND_DRAGGING) return -1;
    d->dst_window = dst_window;
    return 0;
}

int wubu_dnd_drop(wubu_dnd_t *d, int accept) {
    if (!d || d->state != WUBU_DND_DRAGGING) return -1;
    if (accept && d->dst_window >= 0) {
        d->accepts = 1;
        d->state = WUBU_DND_DROPPED;
    } else {
        d->accepts = 0;
        d->state = WUBU_DND_CANCELLED;
    }
    return d->state;
}

int wubu_dnd_cancel(wubu_dnd_t *d) {
    if (!d || d->state != WUBU_DND_DRAGGING) return -1;
    d->state = WUBU_DND_CANCELLED;
    d->accepts = 0;
    return 0;
}

wubu_dnd_state_t wubu_dnd_get_state(const wubu_dnd_t *d) {
    return d ? d->state : WUBU_DND_IDLE;
}
