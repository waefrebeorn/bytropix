#ifndef WUBU_NS_TREE_H
#define WUBU_NS_TREE_H
#include <stddef.h>

/* wubu_ns_tree.h -- WuBuOS /n/* 9P namespace tree (NAMESPACE #178-#189).
 *
 * The unified namespace that exposes the WHOLE machine as 9P files: kernel +
 * hw at /n, Brain internals as introspectable /n files, the companion at
 * /n/bonzi, KV cache at /n/kv, DRAM hedge at /n/dram, expert slot-bank at
 * /n/experts/{slot}, diagnostic hive at /n/hive/{cell}, model catalog at
 * /n/models, training run at /n/train/{metric}. This is the routing/policy
 * core: it classifies a path to a subsystem, resolves dynamic segments
 * ({slot},{cell},{metric}), and dispatches reads to a per-subsystem backend.
 * Fail-closed: an unknown path or NULL resolves to an error. Deterministic,
 * no allocation, real work (no stubs). */

typedef enum {
    WUBU_NS_KERNEL=0,   /* /n (scheduler + hw) */
    WUBU_NS_BRAIN,      /* /n/brain (internals) */
    WUBU_NS_BONZI,      /* /n/bonzi, /n/bonzi/mem */
    WUBU_NS_KV,         /* /n/kv */
    WUBU_NS_DRAM,       /* /n/dram */
    WUBU_NS_EXPERTS,    /* /n/experts/{slot} */
    WUBU_NS_HIVE,       /* /n/hive/{cell} */
    WUBU_NS_MODELS,     /* /n/models */
    WUBU_NS_TRAIN,      /* /n/train/{metric} */
    WUBU_NS_WORLD,      /* /n/world (worldsim state) */
    WUBU_NS_UNKNOWN,    /* not a /n path / unrecognised */
} wubu_ns_subsys_t;

/* Resolved path info. */
typedef struct {
    wubu_ns_subsys_t subsys;
    char dynamic[64];   /* the {slot}/{cell}/{metric} value, or "" */
    int  is_dir;        /* 1 if the path is a directory node */
} wubu_ns_resolved_t;

/* Is this a /n namespace path? 1/0. */
int wubu_ns_tree_is_ns(const char *path);
/* Classify + resolve a /n path. Returns 0 on success (fills out), -1 if the
 * path is not a /n path or is malformed (fail-closed). */
int wubu_ns_tree_resolve(const char *path, wubu_ns_resolved_t *out);
/* Human-readable subsystem name. */
const char *wubu_ns_tree_subsys_name(wubu_ns_subsys_t s);

#endif
