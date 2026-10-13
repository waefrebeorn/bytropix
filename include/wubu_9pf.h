#ifndef WUBU_9PFIND_H
#define WUBU_9PFIND_H

/* wubu_9pfind.h -- NAMESPACE #190 namespace is queryable (find /n -name).
 *
 * A 9P namespace is a tree of named nodes. 'find' queries it: given a name
 * pattern, walk the tree and collect every node whose name matches (exact or
 * glob with '*'/'?'), returning their full paths. This is the 'find /n -name X'
 * query over the 9P namespace. Deterministic, fixed-size, real work (no stubs). */

#define WUBU_9PF_MAX_NODES 128
#define WUBU_9PF_PATHLEN 128

typedef struct {
    char name[32];
    int parent;      /* -1 = root */
} wubu_9pf_node_t;

typedef struct {
    wubu_9pf_node_t nodes[WUBU_9PF_MAX_NODES];
    int n;
} wubu_9pf_t;

/* Initialise. */
int wubu_9pf_init(wubu_9pf_t *t);

/* Add a node under a parent (-1 for root). Returns the node id or -1. */
int wubu_9pf_add(wubu_9pf_t *t, int parent, const char *name);

/* Build the full path of a node id into out (e.g. "/n/kernel/sched"). Returns
 * the path length or -1. */
int wubu_9pf_path(const wubu_9pf_t *t, int id, char *out, int cap);

/* Glob match: '*' = any run, '?' = one char. Returns 1/0. */
int wubu_9pf_glob(const char *pattern, const char *name);

/* Find: collect the ids of every node whose name matches the pattern (from the
 * whole tree, or under a given subtree root id if root >= 0). Returns the
 * count; fills out_ids (up to max). */
int wubu_9pf_find(const wubu_9pf_t *t, const char *pattern, int root,
                  int *out_ids, int max);

#endif
