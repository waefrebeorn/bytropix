/*
 * wubu_kv_shield.c -- Adversarial-robust KV access (L15 / F safety theme).
 * Self-contained C11.
 *
 * Convergence (adversarial-ML 7-hop + our own Styx crash-fuzz hardening): a KV
 * cache accessed by untrusted indices (e.g. attacker-controlled attention spans,
 * poisoned prompts, malformed 9P/Styx frames mapping to KV ops) must never read
 * or write out of bounds. This module is the *gate* the forward pass calls before
 * touching KV slot `idx`: it clamps/validates against [0, cap) and returns a safe
 * boolean. It is the KV-layer analogue of styx_getstr's `avail` clamp. No OOB,
 * no poison-induced crash. Triple-DA: cap<=0 -> reject all; idx out of range ->
 * reject; idx in range -> accept. Deterministic.
 */
#include "wubu_kv_shield.h"

/* Return 1 if `idx` is a safe, in-bounds KV slot for a cache of `cap` slots.
 * Also supports the StreamingKV remap (if remap != NULL, `idx` is first remapped
 * then checked). On any invalid input (cap<=0, idx<0 after remap) returns 0. */
int wubu_kv_shield_check(long idx, int cap, const wubu_kv_shield_remap *remap) {
    if (cap <= 0) return 0;
    long physical = idx;
    if (remap) {
        /* remap maps logical -> physical; out-of-window logicals clamp to the
         * sink/window set, but we still must bounds-check the result. */
        physical = remap->remap(remap->ud, idx);
    }
    if (physical < 0 || physical >= cap) return 0;
    return 1;
}

/* Safe read: copy up to `n` bytes from KV buffer `buf` at slot `idx` only if the
 * slot is in bounds; otherwise writes nothing and returns 0. NULL buf/out -> 0. */
int wubu_kv_shield_read(const void *buf, long idx, int cap, int slot_bytes,
                        void *out, int n) {
    if (!buf || !out || slot_bytes <= 0 || n <= 0) return 0;
    if (idx < 0 || idx >= cap) return 0;
    const unsigned char *base = (const unsigned char *)buf + (long)idx * slot_bytes;
    int cpy = n < slot_bytes ? n : slot_bytes;
    /* memcpy with a verified in-bounds source; no overrun possible. */
    unsigned char *o = (unsigned char *)out;
    for (int i = 0; i < cpy; i++) o[i] = base[i];
    return cpy;
}
