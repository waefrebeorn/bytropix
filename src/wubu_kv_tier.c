/*
 * wubu_kv_tier.c -- KV cache multi-tier cold storage (doc 002).
 *
 * Three tiers on a single host:
 *   HOT  = existing gqa_k_cache / gqa_v_cache (CPU RAM, current tokens)
 *   WARM = DRAM-mmap file (pinned pages, evictable under memory pressure)
 *   COLD = NVMe file via POSIX mmap (swap-backed; page faults are latency)
 *
 * Per-block: tier tag + last-access EMA + block_id + offset into file.
 * Eviction policy: EMA-scored LRU (hot stays hot, cold blocks demoted further).
 *
 * This module is self-contained C11, no third-party deps.
 * Integration target: replace raw gqa_k_cache offset math in wubu_gqa_forward
 * with block-id lookups that route through the tier.
 *
 * Research basis: MTDS (s40747-025-02200-4), Hybe (3695053.3731051),
 * Ganjihal arXiv:2604.26968 (2026).
 */

#include "wubu_kv_tier.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>

/* ---- Block pool ---- */

wubu_kv_tier_t *wubu_kv_tier_create(size_t hot_capacity_blocks,
                                    size_t warm_limit_mb,
                                    size_t cold_limit_mb,
                                    const char *warm_path,
                                    const char *cold_path) {
    wubu_kv_tier_t *t = (wubu_kv_tier_t *)calloc(1, sizeof(*t));
    if (!t) return NULL;

    t->hot_capacity = hot_capacity_blocks;
    t->warm_limit_bytes = warm_limit_mb * 1024 * 1024;
    t->cold_limit_bytes = cold_limit_mb * 1024 * 1024;
    t->warm_file_path = warm_path ? strdup(warm_path) : NULL;
    t->cold_file_path = cold_path ? strdup(cold_path) : NULL;

    /* Pre-allocate hot tier arena */
    t->hot_blocks = (wubu_kv_block_t *)calloc(hot_capacity_blocks, sizeof(wubu_kv_block_t));
    if (!t->hot_blocks) { free(t); return NULL; }
    t->hot_used = 0;

    /* Free list for warm/cold blocks */
    t->warm_free_top = -1;
    t->cold_free_top = -1;

    /* Open/create warm file (mmap'd DRAM-backed, but evictable) */
    if (t->warm_file_path) {
        t->warm_fd = open(t->warm_file_path, O_CREAT | O_RDWR, 0600);
        if (t->warm_fd >= 0) {
            struct stat st;
            if (fstat(t->warm_fd, &st) == 0 && st.st_size == 0) {
                /* Pre-extend file to limit */
                if (ftruncate(t->warm_fd, t->warm_limit_bytes) != 0) {
                    close(t->warm_fd);
                    t->warm_fd = -1;
                }
            }
        }
    } else {
        t->warm_fd = -1;
    }

    /* Open/create cold file (NVMe-backed swap file) */
    if (t->cold_file_path) {
        t->cold_fd = open(t->cold_file_path, O_CREAT | O_RDWR, 0600);
        if (t->cold_fd >= 0) {
            struct stat st;
            if (fstat(t->cold_fd, &st) == 0 && st.st_size == 0) {
                if (ftruncate(t->cold_fd, t->cold_limit_bytes) != 0) {
                    close(t->cold_fd);
                    t->cold_fd = -1;
                }
            }
        }
    } else {
        t->cold_fd = -1;
    }

    return t;
}

void wubu_kv_tier_free(wubu_kv_tier_t *t) {
    if (!t) return;
    if (t->hot_blocks) free(t->hot_blocks);
    if (t->warm_file_path) free(t->warm_file_path);
    if (t->cold_file_path) free(t->cold_file_path);
    if (t->warm_fd >= 0) close(t->warm_fd);
    if (t->cold_fd >= 0) close(t->cold_fd);
    free(t);
}

/* ---- Block allocation ---- */

/* Allocate a block in the hot tier. Returns NULL if hot tier is full. */
static wubu_kv_block_t *hot_alloc(wubu_kv_tier_t *t) {
    if (t->hot_used >= t->hot_capacity) return NULL;
    wubu_kv_block_t *b = &t->hot_blocks[t->hot_used++];
    b->tier = WUBU_KV_TIER_HOT;
    b->ref_count = 1;
    b->last_access_ema = 1.0f;
    b->offset_in_file = 0;
    b->file_fd = -1;
    b->mmap_addr = NULL;
    b->mmap_size = 0;
    return b;
}

wubu_kv_block_t *wubu_kv_tier_alloc_block(wubu_kv_tier_t *t,
                                           size_t block_bytes) {
    /* Try hot tier first */
    wubu_kv_block_t *b = hot_alloc(t);
    if (b) {
        b->data = (uint8_t *)calloc(1, block_bytes);
        if (!b->data) {
            /* Roll back */
            t->hot_used--;
            return NULL;
        }
        b->block_bytes = block_bytes;
        b->is_dirty = true;
        return b;
    }

    /* Hot tier full — try warm tier (evict coldest EMA block from hot first) */
    if (t->warm_fd >= 0 && t->warm_used_bytes + block_bytes <= t->warm_limit_bytes) {
        /* Evict the lowest-EMA hot block to warm */
        if (t->hot_used > 0) {
            int victim = 0;
            float lowest_ema = t->hot_blocks[0].last_access_ema;
            for (size_t i = 1; i < t->hot_used; i++) {
                if (t->hot_blocks[i].last_access_ema < lowest_ema) {
                    lowest_ema = t->hot_blocks[i].last_access_ema;
                    victim = (int)i;
                }
            }
            /* Demote victim to warm */
            wubu_kv_block_t *v = &t->hot_blocks[victim];
            /* Write victim data to warm file */
            if (v->is_dirty && v->data) {
                off_t off = lseek(t->warm_fd, 0, SEEK_END);
                if (off >= 0) {
                    v->offset_in_file = (uint64_t)off;
                    v->file_fd = t->warm_fd;
                    ssize_t wr = write(t->warm_fd, v->data, v->block_bytes);
                    if (wr == (ssize_t)v->block_bytes) {
                        v->is_dirty = false;
                        v->tier = WUBU_KV_TIER_WARM;
                        /* Compact hot array: swap with last */
                        t->hot_blocks[victim] = t->hot_blocks[t->hot_used - 1];
                        t->hot_used--;
                        t->warm_used_bytes += v->block_bytes;
                        return wubu_kv_tier_alloc_block(t, block_bytes);
                    }
                }
            }
        }
    }

    return NULL;
}

/* ---- Read / write ---- */

int wubu_kv_tier_read_block(wubu_kv_tier_t *t, wubu_kv_block_t *b, size_t offset,
                             uint8_t *dst, size_t len) {
    if (!b || !b->data || offset + len > b->block_bytes) return -1;
    memcpy(dst, b->data + offset, len);
    b->last_access_ema = b->last_access_ema * 0.9f + 1.0f * 0.1f;
    return 0;
}

int wubu_kv_tier_write_block(wubu_kv_tier_t *t, wubu_kv_block_t *b, size_t offset,
                              const uint8_t *src, size_t len) {
    /* Ensure block is in hot tier */
    if (b->tier != WUBU_KV_TIER_HOT) {
        /* Promote from warm/cold to hot */
        if (b->tier == WUBU_KV_TIER_WARM && b->file_fd >= 0) {
            /* Read from warm file into fresh hot block */
            off_t cur = lseek(b->file_fd, 0, SEEK_CUR);
            lseek(b->file_fd, b->offset_in_file, SEEK_SET);
            uint8_t *tmp = (uint8_t *)malloc(b->block_bytes);
            if (!tmp) return -1;
            ssize_t rd = read(b->file_fd, tmp, b->block_bytes);
            lseek(b->file_fd, cur, SEEK_SET);
            if (rd != (ssize_t)b->block_bytes) { free(tmp); return -1; }
            /* Find or allocate hot slot */
            wubu_kv_block_t *hot = hot_alloc(t);
            if (!hot) { free(tmp); return -1; }
            memcpy(hot->data, tmp, b->block_bytes);
            hot->block_bytes = b->block_bytes;
            hot->is_dirty = true;
            memcpy(b, hot, sizeof(*b));
            free(tmp);
        } else {
            return -1;
        }
    }
    memcpy(b->data + offset, src, len);
    b->is_dirty = true;
    b->last_access_ema += 1.0f;
    return 0;
}

/* ---- Eviction ---- */

void wubu_kv_tier_evict_cold(wubu_kv_tier_t *t, size_t target_evict_bytes) {
    /* Evict low-EMA blocks from warm to cold (or free cold blocks) */
    size_t evicted = 0;
    for (size_t i = 0; i < t->hot_used && evicted < target_evict_bytes; i++) {
        wubu_kv_block_t *b = &t->hot_blocks[i];
        if (b->tier == WUBU_KV_TIER_HOT && b->last_access_ema < 0.3f) {
            /* Demote to warm if possible, otherwise free */
            if (t->warm_fd >= 0 && t->warm_used_bytes + b->block_bytes <= t->warm_limit_bytes) {
                off_t off = lseek(t->warm_fd, 0, SEEK_END);
                if (off >= 0) {
                    b->offset_in_file = (uint64_t)off;
                    b->file_fd = t->warm_fd;
                    write(t->warm_fd, b->data, b->block_bytes);
                    b->tier = WUBU_KV_TIER_WARM;
                    b->is_dirty = false;
                    t->warm_used_bytes += b->block_bytes;
                    evicted += b->block_bytes;
                }
            } else {
                /* Free the block */
                if (b->data) free(b->data);
                /* Swap with last */
                t->hot_blocks[i] = t->hot_blocks[t->hot_used - 1];
                t->hot_used--;
                i--;
            }
        }
    }
}

/* ---- Stats ---- */

void wubu_kv_tier_stats(const wubu_kv_tier_t *t,
                            size_t *hot_blocks, size_t *warm_bytes,
                            size_t *cold_bytes) {
    if (hot_blocks) *hot_blocks = t ? t->hot_used : 0;
    if (warm_bytes) *warm_bytes = t ? t->warm_used_bytes : 0;
    if (cold_bytes) *cold_bytes = 0; /* simplified: cold eviction not yet implemented */
}
