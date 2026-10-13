/* wubu_ns_tree.c -- WuBuOS /n/* 9P namespace tree (NAMESPACE #178-#189).
 * See wubu_ns_tree.h. Routing/policy core, fail-closed. */
#include "wubu_ns_tree.h"
#include <string.h>

const char *wubu_ns_tree_subsys_name(wubu_ns_subsys_t s) {
    switch (s) {
        case WUBU_NS_KERNEL: return "kernel";
        case WUBU_NS_BRAIN:  return "brain";
        case WUBU_NS_BONZI:  return "bonzi";
        case WUBU_NS_KV:     return "kv";
        case WUBU_NS_DRAM:   return "dram";
        case WUBU_NS_EXPERTS:return "experts";
        case WUBU_NS_HIVE:   return "hive";
        case WUBU_NS_MODELS: return "models";
        case WUBU_NS_TRAIN:  return "train";
        default:             return "unknown";
    }
}

int wubu_ns_tree_is_ns(const char *path) {
    return path && strncmp(path, "/n", 2) == 0 && (path[2] == '\0' || path[2] == '/');
}

/* Split the first path segment after "/n/". Returns 1 if a segment was read. */
static int first_seg(const char *rest, char *seg, size_t segcap, const char **after) {
    if (!rest || rest[0] == '\0') return 0;
    const char *slash = strchr(rest, '/');
    size_t len = slash ? (size_t)(slash - rest) : strlen(rest);
    if (len == 0 || len >= segcap) return 0;
    memcpy(seg, rest, len);
    seg[len] = '\0';
    *after = slash ? slash + 1 : rest + len;
    return 1;
}

int wubu_ns_tree_resolve(const char *path, wubu_ns_resolved_t *out) {
    if (!out) return -1;
    memset(out, 0, sizeof *out);
    out->subsys = WUBU_NS_UNKNOWN;
    if (!wubu_ns_tree_is_ns(path)) return -1;

    /* /n itself or /n/ = kernel root (dir) */
    if (path[2] == '\0' || (path[2] == '/' && path[3] == '\0')) {
        out->subsys = WUBU_NS_KERNEL;
        out->is_dir = 1;
        return 0;
    }

    const char *rest = path + 3;   /* skip "/n/" */
    char seg[64];
    const char *after;
    if (!first_seg(rest, seg, sizeof seg, &after)) return -1;

    if (strcmp(seg, "brain") == 0) {
        out->subsys = WUBU_NS_BRAIN;
        out->is_dir = (after[0] == '\0');
        return 0;
    }
    if (strcmp(seg, "bonzi") == 0) {
        out->subsys = WUBU_NS_BONZI;
        /* /n/bonzi is a dir; /n/bonzi/mem is a file; /n/bonzi/enabled file */
        out->is_dir = (after[0] == '\0');
        return 0;
    }
    if (strcmp(seg, "kv") == 0) {
        out->subsys = WUBU_NS_KV;
        out->is_dir = (after[0] == '\0');
        return 0;
    }
    if (strcmp(seg, "dram") == 0) {
        out->subsys = WUBU_NS_DRAM;
        out->is_dir = (after[0] == '\0');
        return 0;
    }
    if (strcmp(seg, "models") == 0) {
        out->subsys = WUBU_NS_MODELS;
        out->is_dir = (after[0] == '\0');
        return 0;
    }
    if (strcmp(seg, "experts") == 0) {
        out->subsys = WUBU_NS_EXPERTS;
        out->is_dir = (after[0] == '\0');
        /* /n/experts/{slot}: capture the dynamic slot */
        if (after[0] != '\0') {
            const char *a2;
            if (first_seg(after, out->dynamic, sizeof out->dynamic, &a2) && out->dynamic[0])
                out->is_dir = (a2[0] == '\0');
        }
        return 0;
    }
    if (strcmp(seg, "hive") == 0) {
        out->subsys = WUBU_NS_HIVE;
        out->is_dir = (after[0] == '\0');
        if (after[0] != '\0') {
            const char *a2;
            if (first_seg(after, out->dynamic, sizeof out->dynamic, &a2) && out->dynamic[0])
                out->is_dir = (a2[0] == '\0');
        }
        return 0;
    }
    if (strcmp(seg, "train") == 0) {
        out->subsys = WUBU_NS_TRAIN;
        out->is_dir = (after[0] == '\0');
        if (after[0] != '\0') {
            const char *a2;
            if (first_seg(after, out->dynamic, sizeof out->dynamic, &a2) && out->dynamic[0])
                out->is_dir = (a2[0] == '\0');
        }
        return 0;
    }
    /* unknown subsystem under /n */
    out->subsys = WUBU_NS_UNKNOWN;
    return -1;
}
