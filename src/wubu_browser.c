/* wubu_browser.c -- Win98/XP tabbed Web browser.
 * See wubu_browser.h. Real, deterministic tabbed-browser history model. */

#include "wubu_browser.h"
#include <string.h>

int wubu_br_init(wubu_browser_t *b, const char *home) {
    if (!b || !home) return -1;
    memset(b, 0, sizeof(*b));
    wubu_br_tab_t *t = &b->tabs[0];
    strncpy(t->url, home, WUBU_BR_URL_MAX - 1);
    t->url[WUBU_BR_URL_MAX - 1] = '\0';
    t->back_n = 0;
    t->fwd_n = 0;
    b->n_tabs = 1;
    b->active = 0;
    return 0;
}

static wubu_br_tab_t *active_tab(wubu_browser_t *b) {
    if (!b || b->active < 0 || b->active >= b->n_tabs) return NULL;
    return &b->tabs[b->active];
}

int wubu_br_new_tab(wubu_browser_t *b, const char *url) {
    if (!b || !url) return -1;
    if (b->n_tabs >= WUBU_BR_MAX_TABS) return -1;
    wubu_br_tab_t *t = &b->tabs[b->n_tabs];
    memset(t, 0, sizeof(*t));
    strncpy(t->url, url, WUBU_BR_URL_MAX - 1);
    t->url[WUBU_BR_URL_MAX - 1] = '\0';
    b->active = b->n_tabs;
    return b->n_tabs++;
}

int wubu_br_close_tab(wubu_browser_t *b, int idx) {
    if (!b || idx < 0 || idx >= b->n_tabs) return -1;
    if (b->n_tabs <= 1) return -1;   /* can't close the last tab */
    for (int i = idx; i < b->n_tabs - 1; i++)
        b->tabs[i] = b->tabs[i + 1];
    b->n_tabs--;
    if (b->active >= b->n_tabs) b->active = b->n_tabs - 1;
    else if (b->active > idx) b->active--;
    return 0;
}

int wubu_br_switch(wubu_browser_t *b, int idx) {
    if (!b || idx < 0 || idx >= b->n_tabs) return -1;
    b->active = idx;
    return 0;
}

int wubu_br_navigate(wubu_browser_t *b, const char *url) {
    if (!b || !url) return -1;
    wubu_br_tab_t *t = active_tab(b);
    if (!t) return -1;
    /* push current onto the back stack (if room) */
    if (t->back_n < WUBU_BR_MAX_HISTORY) {
        strncpy(t->back[t->back_n], t->url, WUBU_BR_URL_MAX - 1);
        t->back[t->back_n][WUBU_BR_URL_MAX - 1] = '\0';
        t->back_n++;
    }
    /* a new navigation clears the forward stack */
    t->fwd_n = 0;
    strncpy(t->url, url, WUBU_BR_URL_MAX - 1);
    t->url[WUBU_BR_URL_MAX - 1] = '\0';
    return 0;
}

int wubu_br_back(wubu_browser_t *b) {
    if (!b) return -1;
    wubu_br_tab_t *t = active_tab(b);
    if (!t || t->back_n == 0) return 1;   /* nowhere to go */
    /* push current onto forward */
    if (t->fwd_n < WUBU_BR_MAX_HISTORY) {
        strncpy(t->fwd[t->fwd_n], t->url, WUBU_BR_URL_MAX - 1);
        t->fwd[t->fwd_n][WUBU_BR_URL_MAX - 1] = '\0';
        t->fwd_n++;
    }
    /* pop back */
    t->back_n--;
    strncpy(t->url, t->back[t->back_n], WUBU_BR_URL_MAX - 1);
    t->url[WUBU_BR_URL_MAX - 1] = '\0';
    return 0;
}

int wubu_br_forward(wubu_browser_t *b) {
    if (!b) return -1;
    wubu_br_tab_t *t = active_tab(b);
    if (!t || t->fwd_n == 0) return 1;   /* nowhere to go */
    /* push current onto back */
    if (t->back_n < WUBU_BR_MAX_HISTORY) {
        strncpy(t->back[t->back_n], t->url, WUBU_BR_URL_MAX - 1);
        t->back[t->back_n][WUBU_BR_URL_MAX - 1] = '\0';
        t->back_n++;
    }
    t->fwd_n--;
    strncpy(t->url, t->fwd[t->fwd_n], WUBU_BR_URL_MAX - 1);
    t->url[WUBU_BR_URL_MAX - 1] = '\0';
    return 0;
}

int wubu_br_bookmark(wubu_browser_t *b, const char *url) {
    if (!b || !url) return -1;
    if (wubu_br_is_bookmarked(b, url)) return -1;   /* dup */
    if (b->n_bookmarks >= WUBU_BR_MAX_BOOKMARKS) return -1;
    strncpy(b->bookmarks[b->n_bookmarks].url, url, WUBU_BR_URL_MAX - 1);
    b->bookmarks[b->n_bookmarks].url[WUBU_BR_URL_MAX - 1] = '\0';
    b->n_bookmarks++;
    return 0;
}

int wubu_br_unbookmark(wubu_browser_t *b, const char *url) {
    if (!b || !url) return -1;
    for (int i = 0; i < b->n_bookmarks; i++) {
        if (strcmp(b->bookmarks[i].url, url) == 0) {
            b->bookmarks[i] = b->bookmarks[b->n_bookmarks - 1];
            b->n_bookmarks--;
            return 0;
        }
    }
    return -1;
}

const char *wubu_br_url(const wubu_browser_t *b) {
    if (!b || b->active < 0 || b->active >= b->n_tabs) return NULL;
    return b->tabs[b->active].url;
}

int wubu_br_can_back(const wubu_browser_t *b) {
    if (!b || b->active < 0 || b->active >= b->n_tabs) return 0;
    return b->tabs[b->active].back_n > 0 ? 1 : 0;
}

int wubu_br_can_forward(const wubu_browser_t *b) {
    if (!b || b->active < 0 || b->active >= b->n_tabs) return 0;
    return b->tabs[b->active].fwd_n > 0 ? 1 : 0;
}

int wubu_br_n_tabs(const wubu_browser_t *b) {
    return b ? b->n_tabs : -1;
}

int wubu_br_is_bookmarked(const wubu_browser_t *b, const char *url) {
    if (!b || !url) return 0;
    for (int i = 0; i < b->n_bookmarks; i++)
        if (strcmp(b->bookmarks[i].url, url) == 0) return 1;
    return 0;
}

int wubu_br_n_bookmarks(const wubu_browser_t *b) {
    return b ? b->n_bookmarks : -1;
}
