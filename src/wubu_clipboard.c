/* wubu_clipboard.c -- UX #245 clipboard copy/cut/paste across apps.
 * See wubu_clipboard.h. */
#include "wubu_clipboard.h"
#include <string.h>

int wubu_clip_init(wubu_clipboard_t *c) {
    if (!c) return -1;
    memset(c->slots, 0, sizeof c->slots);
    c->head = 0;
    c->count = 0;
    return 0;
}

int wubu_clip_set(wubu_clipboard_t *c, const char *text, wubu_clip_op_t op, int owner_app) {
    if (!c || !text) return -1;
    if (strlen(text) >= WUBU_CLIP_LEN) return -1;
    c->head = (c->head + 1) % WUBU_CLIP_MAX;
    wubu_clip_entry_t *e = &c->slots[c->head];
    strncpy(e->data, text, WUBU_CLIP_LEN - 1);
    e->data[WUBU_CLIP_LEN - 1] = '\0';
    e->op = op;
    e->owner_app = owner_app;
    e->valid = 1;
    if (c->count < WUBU_CLIP_MAX) c->count++;
    return 0;
}

int wubu_clip_paste(wubu_clipboard_t *c, char *out, size_t outsz) {
    if (!c || !out || outsz == 0) return -1;
    if (c->count == 0) return -1;
    wubu_clip_entry_t *e = &c->slots[c->head];
    if (!e->valid) return -1;
    strncpy(out, e->data, outsz - 1);
    out[outsz - 1] = '\0';
    int op = e->op;
    if (op == WUBU_CLIP_CUT) {
        /* cut consumes the slot (source cleared on paste) */
        e->valid = 0;
        e->data[0] = '\0';
        if (c->count > 0) c->count--;
    }
    return op;
}

int wubu_clip_peek(const wubu_clipboard_t *c, char *out, size_t outsz) {
    if (!c || !out || outsz == 0 || c->count == 0) return -1;
    const wubu_clip_entry_t *e = &c->slots[c->head];
    if (!e->valid) return -1;
    strncpy(out, e->data, outsz - 1);
    out[outsz - 1] = '\0';
    return e->op;
}

int wubu_clip_empty(const wubu_clipboard_t *c) {
    if (!c || c->count == 0) return 1;
    return c->slots[c->head].valid ? 0 : 1;
}
