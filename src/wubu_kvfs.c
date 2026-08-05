/* wubu_kvfs.c — KV namespace layer (G1 implementation)
 *
 * Path-addressable KV cache. Every mount is a directory
 * entry mapping a path prefix to a contiguous block range
 * in the flat KV tensor. Read and write route through
 * the mount table — the same address translation as
 * PagedAttention (SOSP'23), just exposed as a filesystem.
 *
 * The read/write API takes the KV tensor base pointer
 * so the namespace can compute absolute offsets and
 * copy data directly into/out of the KV cache.
 *
 * C11, opaque struct, minimal includes. No third-party deps.
 */
#define _POSIX_C_SOURCE 200809L
#include "wubu_kvfs.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- namespace handle ---- */
struct wubu_kvfs {
    wubu_kvfs_mount_t *mounts;
    int                 n_mounts;
    int                 cap_mounts;
    uint32_t            block_size;
    uint32_t            total_blocks;
    uint32_t            used_blocks;
};

wubu_kvfs_t *wubu_kvfs_create(uint32_t block_size, uint32_t total_blocks) {
    wubu_kvfs_t *fs = (wubu_kvfs_t *)calloc(1, sizeof(*fs));
    if (!fs) return NULL;
    fs->block_size   = block_size;
    fs->total_blocks = total_blocks;
    fs->cap_mounts   = 8;
    fs->mounts = (wubu_kvfs_mount_t *)calloc(fs->cap_mounts, sizeof(*fs->mounts));
    if (!fs->mounts) { free(fs); return NULL; }
    return fs;
}

static int grow_mounts(wubu_kvfs_t *fs) {
    int new_cap = fs->cap_mounts * 2;
    wubu_kvfs_mount_t *tmp = (wubu_kvfs_mount_t *)realloc(fs->mounts,
                                    new_cap * sizeof(*tmp));
    if (!tmp) return -1;
    fs->mounts = tmp;
    memset(fs->mounts + fs->cap_mounts, 0,
           (new_cap - fs->cap_mounts) * sizeof(*tmp));
    fs->cap_mounts = new_cap;
    return 0;
}

int wubu_kvfs_mount(wubu_kvfs_t *fs, const char *path,
                      uint32_t start_block, uint32_t n_blocks) {
    if (!fs || !path || !*path) return -1;
    if (start_block + n_blocks > fs->total_blocks) return -1;
    /* duplicate check */
    for (int i = 0; i < fs->n_mounts; i++) {
        if (strcmp(fs->mounts[i].path, path) == 0) return -1;
    }
    if (fs->n_mounts >= fs->cap_mounts) {
        if (grow_mounts(fs) < 0) return -1;
    }
    wubu_kvfs_mount_t *m = &fs->mounts[fs->n_mounts++];
    snprintf(m->path, sizeof(m->path), "%s", path);
    m->start_block = start_block;
    m->n_blocks    = n_blocks;
    m->block_size  = fs->block_size;
    fs->used_blocks += n_blocks;
    return 0;
}

int wubu_kvfs_unmount(wubu_kvfs_t *fs, const char *path) {
    if (!fs || !path) return -1;
    for (int i = 0; i < fs->n_mounts; i++) {
        if (strcmp(fs->mounts[i].path, path) == 0) {
            fs->used_blocks -= fs->mounts[i].n_blocks;
            for (int j = i; j < fs->n_mounts - 1; j++)
                fs->mounts[j] = fs->mounts[j + 1];
            fs->n_mounts--;
            memset(&fs->mounts[fs->n_mounts], 0, sizeof(*fs->mounts));
            return 0;
        }
    }
    return -1;
}

/* longest-prefix match: find the mount whose path is the
 * longest prefix of `lookup`. */
int wubu_kvfs_lookup(const wubu_kvfs_t *fs, const char *path,
                       uint32_t *out_block, size_t *out_offset) {
    if (!fs || !path || !*path) return -1;
    const wubu_kvfs_mount_t *best = NULL;
    size_t best_len = 0;
    for (int i = 0; i < fs->n_mounts; i++) {
        const wubu_kvfs_mount_t *m = &fs->mounts[i];
        size_t plen = strlen(m->path);
        if (plen > best_len && strncmp(path, m->path, plen) == 0 &&
            (path[plen] == '/' || path[plen] == '\0')) {
            best = m;
            best_len = plen;
        }
    }
    if (!best) return -1;
    if (out_block) *out_block = best->start_block;
    if (out_offset) *out_offset = 0;
    return 0;
}

/* Find the mount that owns this path. Returns NULL if not found. */
static const wubu_kvfs_mount_t *find_mount(const wubu_kvfs_t *fs,
                                               const char *path) {
    if (!fs || !path) return NULL;
    const wubu_kvfs_mount_t *best = NULL;
    size_t best_len = 0;
    for (int i = 0; i < fs->n_mounts; i++) {
        const wubu_kvfs_mount_t *m = &fs->mounts[i];
        size_t plen = strlen(m->path);
        if (plen > best_len && strncmp(path, m->path, plen) == 0 &&
            (path[plen] == '/' || path[plen] == '\0')) {
            best = m;
            best_len = plen;
        }
    }
    return best;
}

/* Read KV data from a path into dst. Copies n_floats floats
 * from the KV tensor at the resolved mount offset into dst.
 * kv_base is the base pointer of the flat KV tensor (float*).
 * The read is bounded by the mount's block range. */
int wubu_kvfs_read(const wubu_kvfs_t *fs, const char *path,
                       const float *kv_base, float *dst, size_t n_floats) {
    if (!fs || !path || !kv_base || !dst || n_floats == 0) return -1;
    const wubu_kvfs_mount_t *m = find_mount(fs, path);
    if (!m) return -1;
    size_t abs_offset = (size_t)m->start_block * m->block_size;
    if (abs_offset + n_floats > (size_t)m->start_block * m->block_size +
                                   (size_t)m->n_blocks * m->block_size)
        return -1; /* read would exceed the mount */
    memcpy(dst, kv_base + abs_offset, n_floats * sizeof(float));
    return 0;
}

/* Write KV data from src into a path. Copies n_floats floats
 * from src into the KV tensor at the resolved mount offset.
 * kv_base is the base pointer of the flat KV tensor (float*).
 * The write is bounded by the mount's block range. */
int wubu_kvfs_write(wubu_kvfs_t *fs, const char *path,
                        float *kv_base, const float *src, size_t n_floats) {
    if (!fs || !path || !kv_base || !src || n_floats == 0) return -1;
    const wubu_kvfs_mount_t *m = find_mount(fs, path);
    if (!m) return -1;
    size_t abs_offset = (size_t)m->start_block * m->block_size;
    if (abs_offset + n_floats > (size_t)m->start_block * m->block_size +
                                   (size_t)m->n_blocks * m->block_size)
        return -1; /* write would exceed the mount */
    memcpy(kv_base + abs_offset, src, n_floats * sizeof(float));
    return 0;
}

char *wubu_kvfs_snapshot_json(const wubu_kvfs_t *fs, size_t *out_len) {
    if (!fs) return NULL;
    size_t buf_cap = 4096 + fs->n_mounts * 256;
    char *buf = (char *)malloc(buf_cap);
    if (!buf) return NULL;
    size_t pos = 0;
    pos += (size_t)snprintf(buf + pos, buf_cap - pos,
            "{\"block_size\":%u,\"total_blocks\":%u,"
            "\"used_blocks\":%u,\"registered\":%d,\"mounts\":[",
            fs->block_size, fs->total_blocks,
            fs->used_blocks, fs->n_mounts);
    for (int i = 0; i < fs->n_mounts; i++) {
        if (i > 0) pos += (size_t)snprintf(buf + pos, buf_cap - pos, ",");
        pos += (size_t)snprintf(buf + pos, buf_cap - pos,
                "{\"path\":\"%s\",\"start_block\":%u,\"n_blocks\":%u}",
                fs->mounts[i].path,
                fs->mounts[i].start_block,
                fs->mounts[i].n_blocks);
    }
    pos += (size_t)snprintf(buf + pos, buf_cap - pos, "]}");
    if (out_len) *out_len = pos;
    return buf;
}

int wubu_kvfs_mount_count(const wubu_kvfs_t *fs) {
    return fs ? fs->n_mounts : 0;
}

void wubu_kvfs_free(wubu_kvfs_t *fs) {
    if (!fs) return;
    free(fs->mounts);
    free(fs);
}