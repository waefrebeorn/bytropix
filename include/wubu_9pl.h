#ifndef WUBU_9PL_H
#define WUBU_9PL_H

/* wubu_9pl.h -- NAMESPACE #194 9P2000.L openat + renameat.
 *
 * The 9P2000.L Linux extensions add path-relative operations: openat (open a
 * file by name relative to a directory fid, not the global root) and renameat
 * (rename a file relative to a directory). These avoid walking from the root
 * for every operation and let a process operate within its namespace subtree.
 * This module implements openat + renameat over the namespace tree. Deterministic,
 * real work (no stubs). */

#define WUBU_9PL_MAX 128
#define WUBU_9PL_NAMELEN 32

typedef struct {
    char name[WUBU_9PL_NAMELEN];
    int parent;      /* -1 = root */
} wubu_9pl_node_t;

typedef struct {
    wubu_9pl_node_t nodes[WUBU_9PL_MAX];
    int n;
} wubu_9pl_t;

/* Initialise. */
int wubu_9pl_init(wubu_9pl_t *t);

/* Create a node under a parent. Returns id or -1. */
int wubu_9pl_mkdir(wubu_9pl_t *t, int parent, const char *name);

/* 9P2000.L openat: open the node named 'name' relative to directory fid 'dir'.
 * Returns the node id or -1 if not found / dir invalid. */
int wubu_9pl_openat(const wubu_9pl_t *t, int dir, const char *name);

/* 9P2000.L renameat: rename node 'oldname' (relative to 'olddir') to 'newname'
 * (relative to 'newdir'), MOVING it into newdir. Returns 0/-1. Fails if the
 * source doesn't exist, the new parent is invalid, or newname already exists. */
int wubu_9pl_renameat(wubu_9pl_t *t, int olddir, const char *oldname,
                      int newdir, const char *newname);

#endif
