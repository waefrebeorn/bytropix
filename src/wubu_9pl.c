/* wubu_9pl.c -- NAMESPACE #194 9P2000.L openat + renameat.
 * See wubu_9pl.h. */
#include "wubu_9pl.h"
#include <string.h>

int wubu_9pl_init(wubu_9pl_t *t) {
    if (!t) return -1;
    memset(t->nodes, 0, sizeof t->nodes);
    t->n = 0;
    return 0;
}

int wubu_9pl_mkdir(wubu_9pl_t *t, int parent, const char *name) {
    if (!t || !name || t->n >= WUBU_9PL_MAX) return -1;
    if (parent < -1 || parent >= t->n) return -1;
    if (strlen(name) >= WUBU_9PL_NAMELEN) return -1;
    int id = t->n;
    strncpy(t->nodes[id].name, name, WUBU_9PL_NAMELEN - 1); t->nodes[id].name[WUBU_9PL_NAMELEN-1] = '\0';
    t->nodes[id].parent = parent;
    t->n++;
    return id;
}

static int find_child(const wubu_9pl_t *t, int dir, const char *name) {
    for (int i = 0; i < t->n; i++)
        if (t->nodes[i].parent == dir && strcmp(t->nodes[i].name, name) == 0) return i;
    return -1;
}

int wubu_9pl_openat(const wubu_9pl_t *t, int dir, const char *name) {
    if (!t || !name || dir < 0 || dir >= t->n) return -1;
    return find_child(t, dir, name);
}

int wubu_9pl_renameat(wubu_9pl_t *t, int olddir, const char *oldname,
                      int newdir, const char *newname) {
    if (!t || !oldname || !newname) return -1;
    if (olddir < 0 || olddir >= t->n || newdir < 0 || newdir >= t->n) return -1;
    int src = find_child(t, olddir, oldname);
    if (src < 0) return -1;                       /* source doesn't exist */
    if (find_child(t, newdir, newname) >= 0) return -1;   /* target name taken */
    /* refuse to move a directory into its own subtree (would create a cycle) */
    int cur = newdir;
    while (cur >= 0) {
        if (cur == src) return -1;
        cur = t->nodes[cur].parent;
    }
    if (strlen(newname) >= WUBU_9PL_NAMELEN) return -1;
    strncpy(t->nodes[src].name, newname, WUBU_9PL_NAMELEN - 1);
    t->nodes[src].name[WUBU_9PL_NAMELEN-1] = '\0';
    t->nodes[src].parent = newdir;
    return 0;
}
