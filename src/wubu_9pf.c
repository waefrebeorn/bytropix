/* wubu_9pf.c -- NAMESPACE #190 namespace is queryable (find /n -name).
 * See wubu_9pf.h. */
#include "wubu_9pf.h"
#include <string.h>
#include <stdio.h>

int wubu_9pf_init(wubu_9pf_t *t) {
    if (!t) return -1;
    memset(t->nodes, 0, sizeof t->nodes);
    t->n = 0;
    return 0;
}

int wubu_9pf_add(wubu_9pf_t *t, int parent, const char *name) {
    if (!t || !name || t->n >= WUBU_9PF_MAX_NODES) return -1;
    if (parent < -1 || parent >= t->n) return -1;
    if (strlen(name) >= 32) return -1;
    int id = t->n;
    strncpy(t->nodes[id].name, name, 31); t->nodes[id].name[31] = '\0';
    t->nodes[id].parent = parent;
    t->n++;
    return id;
}

int wubu_9pf_path(const wubu_9pf_t *t, int id, char *out, int cap) {
    if (!t || !out || cap <= 0 || id < 0 || id >= t->n) return -1;
    /* walk up collecting names, then emit root-first */
    const char *parts[WUBU_9PF_MAX_NODES];
    int depth = 0, cur = id;
    while (cur >= 0 && depth < WUBU_9PF_MAX_NODES) {
        parts[depth++] = t->nodes[cur].name;
        cur = t->nodes[cur].parent;
    }
    int len = 0;
    for (int i = depth - 1; i >= 0; i--) {
        int written = snprintf(out + len, cap - len, "/%s", parts[i]);
        if (written < 0 || len + written >= cap) return -1;
        len += written;
    }
    if (len == 0 && cap > 0) { out[0] = '/'; len = 1; if (cap > 1) out[1] = '\0'; }
    return len;
}

int wubu_9pf_glob(const char *pattern, const char *name) {
    if (!pattern || !name) return 0;
    const char *p = pattern, *s = name;
    const char *star = NULL, *ss = NULL;
    while (*s) {
        if (*p == '?' || *p == *s) { p++; s++; }
        else if (*p == '*') { star = p++; ss = s; }
        else if (star) { p = star + 1; s = ++ss; }
        else return 0;
    }
    while (*p == '*') p++;
    return *p == '\0' ? 1 : 0;
}

static int in_subtree(const wubu_9pf_t *t, int node, int root) {
    int cur = node;
    while (cur >= 0) {
        if (cur == root) return 1;
        cur = t->nodes[cur].parent;
    }
    return 0;
}

int wubu_9pf_find(const wubu_9pf_t *t, const char *pattern, int root,
                  int *out_ids, int max) {
    if (!t || !pattern || !out_ids || max <= 0) return 0;
    int count = 0;
    for (int i = 0; i < t->n; i++) {
        if (root >= 0 && !in_subtree(t, i, root)) continue;
        if (wubu_9pf_glob(pattern, t->nodes[i].name)) {
            if (count < max) out_ids[count] = i;
            count++;
        }
    }
    return count;
}
