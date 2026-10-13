/* wubu_ctxmenu.c -- UX #243 right-click context menus everywhere.
 * See wubu_ctxmenu.h. */
#include "wubu_ctxmenu.h"
#include <string.h>

int wubu_ctx_init(wubu_ctxmenu_t *m) {
    if (!m) return -1;
    memset(m->items, 0, sizeof m->items);
    m->n = 0;
    m->cursor = -1;
    m->open = 0;
    return 0;
}

int wubu_ctx_add(wubu_ctxmenu_t *m, const char *label, int id) {
    if (!m || !label || m->n >= WUBU_CTX_MAX_ITEMS) return -1;
    if (strlen(label) >= WUBU_CTX_LABELLEN) return -1;
    wubu_ctx_item_t *it = &m->items[m->n];
    strncpy(it->label, label, WUBU_CTX_LABELLEN - 1);
    it->label[WUBU_CTX_LABELLEN - 1] = '\0';
    it->id = id;
    it->enabled = 1;
    m->n++;
    return 0;
}

int wubu_ctx_add_sep(wubu_ctxmenu_t *m) {
    if (!m || m->n >= WUBU_CTX_MAX_ITEMS) return -1;
    m->items[m->n].label[0] = '\0';
    m->items[m->n].id = -1;   /* separator marker */
    m->items[m->n].enabled = 0;
    m->n++;
    return 0;
}

int wubu_ctx_set_enabled(wubu_ctxmenu_t *m, int id, int enabled) {
    if (!m) return -1;
    for (int i = 0; i < m->n; i++)
        if (m->items[i].id == id) { m->items[i].enabled = enabled ? 1 : 0; return 0; }
    return -1;
}

int wubu_ctx_open(wubu_ctxmenu_t *m) {
    if (!m) return -1;
    m->open = 1;
    m->cursor = -1;
    /* highlight first enabled item */
    for (int i = 0; i < m->n; i++)
        if (m->items[i].enabled && m->items[i].id >= 0) { m->cursor = i; break; }
    return 0;
}

static int is_selectable(const wubu_ctxmenu_t *m, int i) {
    return i >= 0 && i < m->n && m->items[i].enabled && m->items[i].id >= 0;
}

int wubu_ctx_next(wubu_ctxmenu_t *m) {
    if (!m || !m->open) return -1;
    for (int i = m->cursor + 1; i < m->n; i++)
        if (is_selectable(m, i)) { m->cursor = i; return i; }
    /* wrap to first */
    for (int i = 0; i < m->n; i++)
        if (is_selectable(m, i)) { m->cursor = i; return i; }
    return -1;
}

int wubu_ctx_prev(wubu_ctxmenu_t *m) {
    if (!m || !m->open) return -1;
    for (int i = m->cursor - 1; i >= 0; i--)
        if (is_selectable(m, i)) { m->cursor = i; return i; }
    /* wrap to last */
    for (int i = m->n - 1; i >= 0; i--)
        if (is_selectable(m, i)) { m->cursor = i; return i; }
    return -1;
}

int wubu_ctx_select(wubu_ctxmenu_t *m) {
    if (!m || !m->open || !is_selectable(m, m->cursor)) return -1;
    int id = m->items[m->cursor].id;
    m->open = 0;
    m->cursor = -1;
    return id;
}

int wubu_ctx_close(wubu_ctxmenu_t *m) {
    if (!m) return -1;
    m->open = 0;
    m->cursor = -1;
    return 0;
}
