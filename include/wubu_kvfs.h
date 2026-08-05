/* wubu_kvfs.h — KV namespace layer (G1: path-addressable KV cache)
 *
 * The KV cache is a file system. This module provides the address
 * layer: paths resolve to (block, offset) pairs that route
 * directly into the KV cache tensors. Read and write go through
 * the namespace — every datum is a file with a path.
 *
 * C11, opaque struct, minimal includes. No third-party deps.
 *
 * API:
 *   wubu_kvfs_create(block_size, n_blocks) — create the namespace
 *   wubu_kvfs_mount(fs, path, start_block, n_blocks) — mount a region
 *   wubu_kvfs_unmount(fs, path)            — unmount a subtree
 *   wubu_kvfs_lookup(fs, path, out_block, out_offset) — resolve path
 *   wubu_kvfs_read(fs, path, dst, n_floats)  — read KV data by path
 *   wubu_kvfs_write(fs, path, src, n_floats) — write KV data by path
 *   wubu_kvfs_snapshot_json(fs, out_len)   — JSON view of the namespace
 *   wubu_kvfs_mount_count(fs)              — registered mount count
 *   wubu_kvfs_free(fs)                     — destroy
 *
 * The namespace IS the KV cache. Every datum is a file with a path.
 * The mount table is the address translation layer (virtual→physical,
 * same idea as OS page tables / PagedAttention). The tensor layer
 * (paged/ring KV, MLA latent space) stays intact; only the addressing
 * layer is new.
 *
 * Design lineage (7-hop):
 *   PagedAttention (SOSP'23) → block tables = virtual memory
 *   RadixAttention (2312.07104) → radix tree = directory structure
 *   MemGPT (2310.08560) → the model pages its own memory
 *   Mooncake (FAST'25) → disaggregated KV pool over CPU/DRAM/SSD
 *   LMCache (2510.09665) → tiered persistent KV chunks as files
 *   Infini-attention (2404.07143) → compressive synthesized writes
 *   Gemma 3 12B (2503.19786) → single encoder, all inputs are data
 */

#ifndef WUBU_KVFS_H
#define WUBU_KVFS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque namespace handle */
typedef struct wubu_kvfs wubu_kvfs_t;

/* A mounted KV region: a path prefix that maps to a contiguous
 * block range in the flat KV cache tensor. */
typedef struct {
    char path[256];          /* mount path, e.g. "/kv/in" */
    uint32_t start_block;    /* first block in the flat KV range */
    uint32_t n_blocks;       /* how many blocks this mount covers */
    uint32_t block_size;     /* tokens per block (same as paged_kv) */
} wubu_kvfs_mount_t;

/* Create a KV namespace with a given block size and total block
 * count. Returns NULL on allocation failure. */
wubu_kvfs_t *wubu_kvfs_create(uint32_t block_size, uint32_t total_blocks);

/* Mount a KV region at a path. The path becomes a directory entry.
 * Returns 0 on success, -1 if the path is already mounted or out
 * of blocks. */
int wubu_kvfs_mount(wubu_kvfs_t *fs, const char *path,
                    uint32_t start_block, uint32_t n_blocks);

/* Unmount a subtree (removes the mount and all children).
 * Returns 0 on success, -1 if path not found. */
int wubu_kvfs_unmount(wubu_kvfs_t *fs, const char *path);

/* Resolve a path to a (block, offset) pair. The block is the
 * index into the flat KV tensor; offset is the byte offset
 * within that block. Returns 0 on success, -1 if the path
 * is not mounted or does not exist. */
int wubu_kvfs_lookup(const wubu_kvfs_t *fs, const char *path,
                     uint32_t *out_block, size_t *out_offset);

/* Read KV data from a path into dst. Reads n_floats floats
 * starting at the resolved (block, offset). kv_base is the
 * base pointer of the flat KV tensor. Returns 0 on success,
 * -1 if the path is not mounted or the read would exceed
 * the mount's block range. */
int wubu_kvfs_read(const wubu_kvfs_t *fs, const char *path,
                       const float *kv_base, float *dst, size_t n_floats);

/* Write KV data from src into a path. Writes n_floats floats
 * starting at the resolved (block, offset). kv_base is the
 * base pointer of the flat KV tensor. Returns 0 on success,
 * -1 if the path is not mounted or the write would exceed
 * the mount's block range. */
int wubu_kvfs_write(wubu_kvfs_t *fs, const char *path,
                        float *kv_base, const float *src, size_t n_floats);

/* JSON snapshot of the namespace: every mount point with its
 * block range. Caller frees with free(). */
char *wubu_kvfs_snapshot_json(const wubu_kvfs_t *fs, size_t *out_len);

/* Registered mount count. */
int wubu_kvfs_mount_count(const wubu_kvfs_t *fs);

/* Destroy the namespace. Does NOT free the backing KV tensor. */
void wubu_kvfs_free(wubu_kvfs_t *fs);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_KVFS_H */
