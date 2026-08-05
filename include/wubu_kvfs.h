/* wubu_kvfs.h — KV namespace layer (G1: radix-tree address space)
 *
 * The KV cache is a file system. This module provides the address
 * layer: paths resolve through a radix tree whose leaves are
 * KV block ranges (PagedAttention blocks, wubu_paged_kv_t).
 *
 * C11, opaque struct, minimal includes. No third-party deps.
 *
 * API:
 *   wubu_kvfs_create(block_size, n_blocks) — create the namespace
 *   wubu_kvfs_mount(fs, path, paged_kv)    — mount a paged KV at a path
 *   wubu_kvfs_lookup(fs, path, out_block, out_offset) — resolve path → block+offset
 *   wubu_kvfs_unmount(fs, path)            — unmount a subtree
 *   wubu_kvfs_snapshot_json(fs, out_len)   — JSON view of the namespace
 *   wubu_kvfs_free(fs)                     — destroy
 *
 * The namespace IS the KV cache. Every datum is a file with a path.
 * The radix tree is the directory structure; the block table is the
 * address translation layer (virtual→physical, same idea as OS
 * virtual memory). The tier (wubu_kv_tier) is the backing store
 * for cold blocks; wubu_lmcache provides persistent chunk files.
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

/* Opaque radix-node (internal) */
typedef struct wubu_kvfs_node wubu_kvfs_node_t;

/* A mounted KV region: a path prefix that maps to a paged KV
 * block range. The block table (start_block, n_blocks) is the
 * virtual→physical translation — same idea as OS page tables. */
typedef struct {
    char path[256];          /* mount path, e.g. "/kv/in" */
    uint32_t start_block;    /* first block in the paged KV range */
    uint32_t n_blocks;       /* how many blocks this mount covers */
    uint32_t block_size;     /* tokens per block (same as paged_kv) */
} wubu_kvfs_mount_t;

/* Create a KV namespace with a given block size and total block
 * count. Returns NULL on allocation failure. */
wubu_kvfs_t *wubu_kvfs_create(uint32_t block_size, uint32_t total_blocks);

/* Mount a paged KV region at a path. The path becomes a
 * directory entry in the radix tree. Returns 0 on success,
 * -1 if the path is already mounted or out of blocks. */
int wubu_kvfs_mount(wubu_kvfs_t *fs, const char *path,
                    uint32_t start_block, uint32_t n_blocks);

/* Unmount a subtree (removes the mount and all children).
 * Returns 0 on success, -1 if path not found. */
int wubu_kvfs_unmount(wubu_kvfs_t *fs, const char *path);

/* Resolve a path to a (block, offset) pair. The block is the
 * index into the paged KV's block table; offset is the byte
 * offset within that block. Returns 0 on success, -1 if the
 * path is not mounted or does not exist. */
int wubu_kvfs_lookup(const wubu_kvfs_t *fs, const char *path,
                     uint32_t *out_block, size_t *out_offset);

/* JSON snapshot of the namespace: every mount point with its
 * block range. Caller frees with free(). */
char *wubu_kvfs_snapshot_json(const wubu_kvfs_t *fs, size_t *out_len);

/* Registered mount count. */
int wubu_kvfs_mount_count(const wubu_kvfs_t *fs);

/* Destroy the namespace. Does NOT free the backing paged KV. */
void wubu_kvfs_free(wubu_kvfs_t *fs);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_KVFS_H */
