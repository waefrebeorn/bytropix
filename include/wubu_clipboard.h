#ifndef WUBU_CLIPBOARD_H
#define WUBU_CLIPBOARD_H
#include <stddef.h>

/* wubu_clipboard.h -- UX #245 clipboard copy/cut/paste across apps.
 *
 * A system clipboard: apps copy/cut data into it and paste from it. Supports
 * multiple formats (text), copy vs cut (cut clears the source on paste), an
 * ownership model (which app owns the clipboard), and a small history so
 * paste-after-cut semantics work across apps. Deterministic, fixed-size,
 * fail-closed. Real work (no stubs). */

#define WUBU_CLIP_MAX 8
#define WUBU_CLIP_LEN 256

typedef enum { WUBU_CLIP_COPY=0, WUBU_CLIP_CUT } wubu_clip_op_t;

typedef struct {
    char data[WUBU_CLIP_LEN];
    wubu_clip_op_t op;      /* copy or cut */
    int owner_app;          /* app id that placed it */
    int valid;
} wubu_clip_entry_t;

typedef struct {
    wubu_clip_entry_t slots[WUBU_CLIP_MAX];
    int head;               /* most-recent slot */
    int count;
} wubu_clipboard_t;

/* Initialise the clipboard. */
int wubu_clip_init(wubu_clipboard_t *c);

/* Copy or cut text into the clipboard (becomes current). Returns 0/-1. */
int wubu_clip_set(wubu_clipboard_t *c, const char *text, wubu_clip_op_t op, int owner_app);

/* Paste the current clipboard content into `out`. Returns the op (copy/cut),
 * or -1 if empty. On a CUT, the slot is consumed (cleared) after paste. */
int wubu_clip_paste(wubu_clipboard_t *c, char *out, size_t outsz);

/* Peek the current content without consuming. Returns op or -1 if empty. */
int wubu_clip_peek(const wubu_clipboard_t *c, char *out, size_t outsz);

/* Is the clipboard empty? */
int wubu_clip_empty(const wubu_clipboard_t *c);

#endif
