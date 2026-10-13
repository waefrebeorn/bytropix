/* wubu_landlock.c -- OMARCHY #276 Landlock unprivileged filesystem sandbox.
 * See wubu_landlock.h. Default-deny path-scoped ruleset, fail-closed. */
#include "wubu_landlock.h"
#include <string.h>

int wubu_landlock_init(wubu_landlock_t *l) {
    if (!l) return -1;
    for (int i = 0; i < WUBU_LL_MAX_RULES; i++) { l->rules[i].path[0] = '\0'; l->rules[i].rights = 0; }
    l->n_rules = 0;
    l->handled = WUBU_LL_EXECUTE|WUBU_LL_WRITE|WUBU_LL_READ|WUBU_LL_READ_DIR|
                 WUBU_LL_REMOVE|WUBU_LL_MAKE_DIR|WUBU_LL_MAKE_REG;
    return 0;
}

int wubu_landlock_allow(wubu_landlock_t *l, const char *path, unsigned rights) {
    if (!l || !path || l->n_rules >= WUBU_LL_MAX_RULES) return -1;
    size_t pl = strlen(path);
    if (pl == 0 || pl >= WUBU_LL_PATHLEN) return -1;
    wubu_ll_rule_t *r = &l->rules[l->n_rules++];
    strncpy(r->path, path, WUBU_LL_PATHLEN - 1);
    r->path[WUBU_LL_PATHLEN - 1] = '\0';
    /* normalise: strip trailing slash (keep a lone "/") */
    if (pl > 1 && r->path[pl - 1] == '/') r->path[pl - 1] = '\0';
    r->rights = rights & l->handled;   /* only governed rights can be granted */
    return 0;
}

/* Is `path` within `subtree`? subtree is a directory prefix with a boundary
 * (so "/etc" does not match "/etcfoo"). */
static int within(const char *path, const char *subtree) {
    size_t sl = strlen(subtree);
    if (strncmp(path, subtree, sl) != 0) return 0;
    /* exact match, or next char is a path separator */
    return path[sl] == '\0' || path[sl] == '/';
}

int wubu_landlock_check(const wubu_landlock_t *l, const char *path, wubu_ll_right_t right) {
    if (!l || !path) return 0;   /* fail-closed */
    if (!(l->handled & (unsigned)right)) return 0;   /* ungoverned right: deny */
    for (int i = 0; i < l->n_rules; i++) {
        if (within(path, l->rules[i].path) && (l->rules[i].rights & (unsigned)right))
            return 1;
    }
    return 0;   /* default-deny */
}
