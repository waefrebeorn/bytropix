/*
 * wubu_kv_styx.c — KV-cache structured exposure for external inspection.
 *
 * This module is the self-contained bridge between wubuwizard's internal
 * KV-cache allocator (`wubu_kv_runtime.c`) and WuBuOS's 9P namespace.
 * It does NOT pull in the full wubu_ns_bridge/styxfs graph — instead it
 * exposes a minimal structured view that any 9P client can walk.
 *
 * C11, opaque-struct, no god headers. Caller owns the lifetime of the
 * backing KV storage; this module only cobbles read-only views of it.
 */
#define _POSIX_C_SOURCE 200809L
#include "wubu_kv_styx.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

/* ---- tiny slab allocator for metadata handles ------------------- */
typedef struct kv_meta_slab {
    char path[256];
    void  *kv_ptr;
    size_t kv_bytes;
    struct kv_meta_slab *next;
} kv_meta_slab_t;

typedef struct {
    kv_meta_slab_t *head;
    int             count;
} kv_meta_root_t;

static kv_meta_root_t g_kvroot;

int wubu_kv_styx_init(void) {
    memset(&g_kvroot, 0, sizeof(g_kvroot));
    return 0;
}

void wubu_kv_styx_shutdown(void) {
    kv_meta_slab_t *cur = g_kvroot.head;
    while (cur) { kv_meta_slab_t *next = cur->next; free(cur); cur = next; }
    g_kvroot.head = NULL; g_kvroot.count = 0;
}

/* ---- public API ------------------------------------------------- */
int wubu_kv_styx_register(const char *layer_path,
                          void *kv_ptr, size_t kv_bytes)
{
    if (!layer_path || !kv_ptr || kv_bytes == 0) return -1;
    kv_meta_slab_t *node = (kv_meta_slab_t *)calloc(1, sizeof(*node));
    if (!node) return -1;
    snprintf(node->path, sizeof(node->path), "%s", layer_path);
    node->kv_ptr  = kv_ptr;
    node->kv_bytes = kv_bytes;
    node->next = g_kvroot.head;
    g_kvroot.head = node;
    g_kvroot.count++;
    return 0;
}

int wubu_kv_styx_unregister(const char *layer_path) {
    kv_meta_slab_t **pp = &g_kvroot.head;
    while (*pp) {
        kv_meta_slab_t *cur = *pp;
        if (strcmp(cur->path, layer_path) == 0) {
            *pp = cur->next; free(cur); g_kvroot.count--; return 0;
        }
        pp = &cur->next;
    }
    return -1;
}

const void *wubu_kv_styx_lookup(const char *layer_path, size_t *out_bytes) {
    kv_meta_slab_t *cur = g_kvroot.head;
    while (cur) {
        if (strcmp(cur->path, layer_path) == 0) {
            if (out_bytes) *out_bytes = cur->kv_bytes;
            return cur->kv_ptr;
        }
        cur = cur->next;
    }
    return NULL;
}

int wubu_kv_styx_registered_count(void) { return g_kvroot.count; }

/* ---- JSON snapshot exposed to any 9P client --------------------- */
char *wubu_kv_styx_snapshot_json(size_t *out_len) {
    (void)out_len;
    /* Caller frees. Return a minimal JSON summary; the caller mounts this
     * at /n/kv/ and serves it with their styxfs callback. */
    size_t cap = 4096 + (size_t)g_kvroot.count * 256;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;
    size_t pos = (size_t)snprintf(buf, cap,
        "{\"registered\":%d,\"layers\":[", g_kvroot.count);
    kv_meta_slab_t *cur = g_kvroot.head;
    while (cur) {
        pos += (size_t)snprintf(buf + pos, cap - pos,
            "{\"path\":\"%s\",\"bytes\":%zu}", cur->path, cur->kv_bytes);
        cur = cur->next;
        if (cur) pos += (size_t)snprintf(buf + pos, cap - pos, ",");
    }
    pos += (size_t)snprintf(buf + pos, cap - pos, "]}");
    return buf;
}
