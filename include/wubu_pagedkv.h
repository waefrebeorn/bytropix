/*
 * wubu_pagedkv.h -- Paged KV cache (block table, CoW, prefix share) (HH02).
 */
#ifndef WUBU_PAGEDKV_H
#define WUBU_PAGEDKV_H

#define WUBU_PAGEDKV_BLOCK 16   /* tokens per block */
#define WUBU_PAGEDKV_MAX_BLOCKS 4096
#define WUBU_PAGEDKV_MAX_SEQ 256

typedef struct {
    int seq_id;
    int n_blocks;                       /* logical blocks allocated */
    int block_table[WUBU_PAGEDKV_MAX_SEQ]; /* logical→physical (simplified: 1 seq) */
    int refcount[WUBU_PAGEDKV_MAX_BLOCKS];
    int free_list[WUBU_PAGEDKV_MAX_BLOCKS];
    int n_free;
    int n_phys_blocks;
    /* prefix sharing: hash of first block → physical block id (reuse) */
    unsigned prefix_hash[WUBU_PAGEDKV_MAX_BLOCKS];
    int      prefix_phys[WUBU_PAGEDKV_MAX_BLOCKS];
    int      n_prefix;
} wubu_pagedkv_t;

/* Init pool with n_phys_blocks. */
int  wubu_pagedkv_init(wubu_pagedkv_t *pk, int n_phys_blocks);
/* Allocate a logical block for seq; returns physical block id (or -1 if OOM). */
int  wubu_pagedkv_alloc(wubu_pagedkv_t *pk, int seq_id);
/* Free a block (decrements refcount; frees when 0). */
int  wubu_pagedkv_free(wubu_pagedkv_t *pk, int phys_block);
/* Share prefix: register hash→phys for copy-on-write reuse. */
int  wubu_pagedkv_share_prefix(wubu_pagedkv_t *pk, unsigned hash, int phys_block);
/* Lookup prefix: returns phys block id if shared (else -1). */
int  wubu_pagedkv_lookup_prefix(wubu_pagedkv_t *pk, unsigned hash);
/* Fragmentation: 0.0 = no waste (contiguous-equivalent), 1.0 = all free. */
float wubu_pagedkv_frag(const wubu_pagedkv_t *pk);

#endif