#ifndef WUBU_CTXMENU_H
#define WUBU_CTXMENU_H
#include <stddef.h>

/* wubu_ctxmenu.h -- UX #243 right-click context menus everywhere.
 *
 * A context menu model: a list of items (label + id + enabled + separator),
 * shown at a position on right-click, with keyboard/pointer navigation and
 * selection. Supports disabled items (greyed, not selectable), separators,
 * and a callback id returned on selection. Deterministic, fixed-size,
 * fail-closed. Real work (no stubs). */

#define WUBU_CTX_MAX_ITEMS 16
#define WUBU_CTX_LABELLEN 48

typedef struct {
    char label[WUBU_CTX_LABELLEN];
    int  id;          /* item id (returned on select), -1 = separator */
    int  enabled;     /* 1 = selectable, 0 = greyed */
} wubu_ctx_item_t;

typedef struct {
    wubu_ctx_item_t items[WUBU_CTX_MAX_ITEMS];
    int n;
    int cursor;       /* highlighted item index, -1 = none */
    int open;         /* 1 = menu shown */
} wubu_ctxmenu_t;

/* Initialise an empty (closed) menu. */
int wubu_ctx_init(wubu_ctxmenu_t *m);

/* Add an item (label, id). Returns 0/-1. */
int wubu_ctx_add(wubu_ctxmenu_t *m, const char *label, int id);

/* Add a separator. Returns 0/-1. */
int wubu_ctx_add_sep(wubu_ctxmenu_t *m);

/* Enable/disable an item by id. Returns 0/-1. */
int wubu_ctx_set_enabled(wubu_ctxmenu_t *m, int id, int enabled);

/* Open the menu (right-click): highlights the first enabled item. */
int wubu_ctx_open(wubu_ctxmenu_t *m);

/* Move the cursor down/up to the next/prev ENABLED item (skips disabled +
 * separators). Returns the new cursor index or -1. */
int wubu_ctx_next(wubu_ctxmenu_t *m);
int wubu_ctx_prev(wubu_ctxmenu_t *m);

/* Select the highlighted item: returns its id (and closes), or -1 if none
 * highlighted / highlighted item disabled. */
int wubu_ctx_select(wubu_ctxmenu_t *m);

/* Close without selecting. */
int wubu_ctx_close(wubu_ctxmenu_t *m);

#endif
