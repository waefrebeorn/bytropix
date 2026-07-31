/*
 * wubu_hive.h — C11 Hive data structure (Vector/List/Hive comparison).
 *
 * A Hive is a linked list of fixed-capacity blocks. Each block has:
 *   - void **slots:      contiguous array of void* pointers
 *   - uint8_t *skip:     bitmask — 1 = slot occupied, 0 = free
 *   - size_t live, cap:  live count and block capacity
 *   - struct block *next: pointer to next block (NULL = end)
 *
 * Operations:
 *   - Insert: reuses a free slot (freelist pop) or allocates a new block
 *   - Erase:  marks slot free (skip=0), pushes to freelist
 *   - Iterate: skips free slots (O(live) not O(cap))
 *   - Pointer stability: slots never move (unlike Vector)
 *
 * Performance profile (vs Vector vs List):
 *   - Random insert/erase: O(1) amortized (freelist) vs O(N) Vector shift
 *   - Pointer stability: YES (stable void* per slot) vs NO Vector realloc
 *   - Cache locality: BETTER than List (contiguous slots per block)
 *   - Modulo arithmetic: O(1) direct access vs O(N) List traversal
 *
 * Self-contained C11. No third-party deps. Opaque struct.
 */

#ifndef WUBU_HIVE_H
#define WUBU_HIVE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_hive_block wubu_hive_block_t;
typedef struct wubu_hive wubu_hive_t;

/* Create a new Hive. block_cap = slots per block (default 64).
 * Returns NULL on OOM. */
wubu_hive_t *wubu_hive_create(size_t block_cap);

/* Destroy the Hive and free all blocks. */
void wubu_hive_destroy(wubu_hive_t *hive);

/* Insert value into the Hive. Reuses a free slot or allocates a new block.
 * Returns 0 on success, -1 on OOM. */
int wubu_hive_insert(wubu_hive_t *hive, void *value);

/* Erase value from the Hive. Marks slot free, pushes to freelist.
 * Returns 0 if found and erased, -1 if not found. */
int wubu_hive_erase(wubu_hive_t *hive, void *value);

/* Iterate over all live slots. cb receives (void *value, size_t index).
 * Returns 0 on success, -1 if callback returns non-zero (early stop). */
int wubu_hive_iterate(const wubu_hive_t *hive,
                      int (*cb)(void *value, size_t index, void *ctx),
                      void *ctx);

/* Get the number of live (occupied) slots. */
size_t wubu_hive_size(const wubu_hive_t *hive);

/* Get the total number of blocks. */
size_t wubu_hive_blocks(const wubu_hive_t *hive);

/* Get the block capacity (slots per block). */
size_t wubu_hive_block_cap(const wubu_hive_t *hive);

/* Lookup: find a value in the Hive. Returns 1 if found, 0 if not. */
int wubu_hive_find(const wubu_hive_t *hive, void *value);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_HIVE_H */
