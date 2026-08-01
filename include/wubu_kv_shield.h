/*
 * wubu_kv_shield.h -- Adversarial-robust KV access (L15 / F safety theme).
 * Opaque-free: a remap callback struct + bound-checked accessors.
 */
#ifndef WUBU_KV_SHIELD_H
#define WUBU_KV_SHIELD_H

#include <stddef.h>

/* Optional remap (e.g. StreamingKV logical->physical). Return physical slot. */
typedef struct {
    long (*remap)(void *ud, long logical);
    void *ud;
} wubu_kv_shield_remap;

/* 1 if idx is a safe in-bounds KV slot (after optional remap). 0 otherwise. */
int wubu_kv_shield_check(long idx, int cap, const wubu_kv_shield_remap *remap);

/* Safe read: copies min(n, slot_bytes) from slot idx only if in-bounds.
 * Returns bytes copied, or 0 on any invalid input / OOB. */
int wubu_kv_shield_read(const void *buf, long idx, int cap, int slot_bytes,
                        void *out, int n);

#endif /* WUBU_KV_SHIELD_H */
