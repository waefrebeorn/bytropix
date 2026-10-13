#ifndef WUBU_BROWSER_H
#define WUBU_BROWSER_H

/* wubu_browser.h -- Win98/XP tabbed Web browser.
 *
 * Models a tabbed browser: each tab has its own navigation history (a back
 * stack + forward stack + current URL), and there is a shared bookmark store.
 * Navigate pushes the current URL onto the back stack and clears forward;
 * back/forward move between the stacks. Tabs can be opened/closed/switched;
 * only the active tab is addressable for navigation. Deterministic, no stubs. */

#include <stddef.h>

#define WUBU_BR_MAX_TABS 16
#define WUBU_BR_MAX_HISTORY 64
#define WUBU_BR_MAX_BOOKMARKS 64
#define WUBU_BR_URL_MAX 128

typedef struct {
    char url[WUBU_BR_URL_MAX];
    char back[WUBU_BR_MAX_HISTORY][WUBU_BR_URL_MAX];
    int back_n;
    char fwd[WUBU_BR_MAX_HISTORY][WUBU_BR_URL_MAX];
    int fwd_n;
} wubu_br_tab_t;

typedef struct {
    char url[WUBU_BR_URL_MAX];
} wubu_br_bookmark_t;

typedef struct {
    wubu_br_tab_t tabs[WUBU_BR_MAX_TABS];
    int n_tabs;
    int active;                          /* active tab index */
    wubu_br_bookmark_t bookmarks[WUBU_BR_MAX_BOOKMARKS];
    int n_bookmarks;
} wubu_browser_t;

/* Initialise with one tab at `home`. Returns 0/-1. */
int wubu_br_init(wubu_browser_t *b, const char *home);

/* Open a new tab at `url` (becomes active). Returns tab index, -1 on full. */
int wubu_br_new_tab(wubu_browser_t *b, const char *url);

/* Close a tab by index. Returns 0/-1 (can't close the last tab). */
int wubu_br_close_tab(wubu_browser_t *b, int idx);

/* Switch the active tab. Returns 0/-1. */
int wubu_br_switch(wubu_browser_t *b, int idx);

/* Navigate the active tab to a URL. Returns 0/-1. */
int wubu_br_navigate(wubu_browser_t *b, const char *url);

/* Go back/forward in the active tab. Returns 0 on success, 1 if there is
 * nowhere to go, -1 on bad args. */
int wubu_br_back(wubu_browser_t *b);
int wubu_br_forward(wubu_browser_t *b);

/* Add/remove a bookmark. Returns 0/-1 (dup add / not found remove). */
int wubu_br_bookmark(wubu_browser_t *b, const char *url);
int wubu_br_unbookmark(wubu_browser_t *b, const char *url);

/* Query. */
const char *wubu_br_url(const wubu_browser_t *b);       /* active tab URL */
int wubu_br_can_back(const wubu_browser_t *b);
int wubu_br_can_forward(const wubu_browser_t *b);
int wubu_br_n_tabs(const wubu_browser_t *b);
int wubu_br_is_bookmarked(const wubu_browser_t *b, const char *url);
int wubu_br_n_bookmarks(const wubu_browser_t *b);

#endif
