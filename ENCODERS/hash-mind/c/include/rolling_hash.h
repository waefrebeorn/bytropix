/**
 * rolling_hash.h — Rabin-Karp Sliding Window Hash
 *
 * Computes rolling hashes over a token sequence using a prime modulus.
 * Used by HashMind for dual-source embedding (char + hash features).
 */
#ifndef ROLLING_HASH_H
#define ROLLING_HASH_H

#include <stdint.h>

#define MODULUS    1000000007U
#define BASE_HASH  31

/* Compute a single sliding-window hash over values[0:window_size) */
uint32_t rolling_hash_one(const int* values, int window_size);

/* Compute all sliding-window hashes over values[0:len).
 * output must be at least len - window_size + 1 elements.
 * *out_len is set to the number of hashes produced. */
void rolling_hash_all(const int* values, int len, int window_size,
                      uint32_t* output, int* out_len);

/* Single-step update: given previous hash, remove old_val, add new_val.
 * Returns the updated hash. */
static inline uint32_t rolling_hash_advance(uint32_t prev_hash,
                                             uint32_t old_val,
                                             uint32_t new_val,
                                             uint32_t base_pow) {
    uint32_t h = prev_hash;
    h = (h + MODULUS - (old_val * base_pow) % MODULUS) % MODULUS;
    h = (h * BASE_HASH + new_val) % MODULUS;
    return h;
}

/* Precompute base^window_size (mod MODULUS) — the power needed for sliding */
uint32_t rolling_hash_base_pow(int window_size);

#endif /* ROLLING_HASH_H */
