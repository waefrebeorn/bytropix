/*
 * wubu_pagedkv.c -- Paged KV cache (block table, CoW, prefix share) (HH02). C11.
 *
 * Convergence (vLLM PagedAttention / OS virtual memory 7-hop):
 *   - HH02: split KV into fixed-size blocks (16 tokens); logical block table →
 *     non-contiguous physical blocks. Eliminates internal+external fragmentation.
 *     Copy-on-write: shared prefix blocks (refcount>1) immutable; prefix caching
 *     via global hash → physical block reuse. At home: the 512K-ctx KV grows on
 *     demand (no 512K contiguous reservation), and shared prefixes across sweeps
 *     reuse KV — never materializing full 512K eagerly (no EAMM).
 */
#include "wubu_pagedkv.h"
#include <string.h>

int wubu_pagedkv_init(wubu_pagedkv_t *pk, int n_phys_blocks) {
    if (!pk || n_phys_blocks <= 0 || n_phys_blocks > WUBU_PAGEDKV_MAX_BLOCKS) return -1;
    memset(pk, 0, sizeof(*pk));
    pk->n_phys_blocks = n_phys_blocks;
    pk->n_free = n_phys_blocks;
    for (int i = 0; i < n_phys_blocks; i++) pk->free_list[i] = n_phys_blocks - 1 - i;
    return 0;
}

int wubu_pagedkv_alloc(wubu_pagedkv_t *pk, int seq_id) {
    if (!pk || pk->n_free <= 0) return -1;
    int phys = pk->free_list[--pk->n_free];
    pk->refcount[phys] = 1;
    if (pk->n_blocks < WUBU_PAGEDKV_MAX_SEQ) {
        pk->block_table[pk->n_blocks] = phys;
        pk->n_blocks++;
    }
    (void)seq_id;
    return phys;
}

int wubu_pagedkv_free(wubu_pagedkv_t *pk, int phys_block) {
    if (!pk || phys_block < 0 || phys_block >= pk->n_phys_blocks) return -1;
    if (pk->refcount[phys_block] <= 0) return -1;
    pk->refcount[phys_block]--;
    if (pk->refcount[phys_block] == 0) {
        pk->free_list[pk->n_free++] = phys_block;
    }
    return 0;
}

int wubu_pagedkv_share_prefix(wubu_pagedkv_t *pk, unsigned hash, int phys_block) {
    if (!pk || phys_block < 0 || phys_block >= pk->n_phys_blocks) return -1;
    if (pk->n_prefix >= WUBU_PAGEDKV_MAX_BLOCKS) return -1;
    /* increment refcount (copy-on-write share) */
    pk->refcount[phys_block]++;
    pk->prefix_hash[pk->n_prefix] = hash;
    pk->prefix_phys[pk->n_prefix] = phys_block;
    pk->n_prefix++;
    return 0;
}

int wubu_pagedkv_lookup_prefix(wubu_pagedkv_t *pk, unsigned hash) {
    if (!pk) return -1;
    for (int i = 0; i < pk->n_prefix; i++)
        if (pk->prefix_hash[i] == hash) return pk->prefix_phys[i];
    return -1;
}

float wubu_pagedkv_frag(const wubu_pagedkv_t *pk) {
    if (!pk || pk->n_phys_blocks == 0) return 0.0f;
    return (float)pk->n_free / (float)pk->n_phys_blocks;
}
