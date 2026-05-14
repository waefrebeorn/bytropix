#include "rolling_hash.h"

uint32_t rolling_hash_one(const int* values, int window_size) {
    uint32_t hash = 0;
    for (int i = 0; i < window_size; i++)
        hash = (hash * BASE_HASH + (uint32_t)(values[i] % MODULUS)) % MODULUS;
    return hash;
}

void rolling_hash_all(const int* values, int len, int window_size,
                       uint32_t* output, int* out_len) {
    if (len < window_size || window_size <= 0) {
        *out_len = 0;
        return;
    }

    uint32_t hash = rolling_hash_one(values, window_size);
    output[0] = hash;

    uint32_t base_pow = rolling_hash_base_pow(window_size);
    int count = 1;
    for (int i = 1; i < len - window_size + 1; i++) {
        uint32_t old_val = (uint32_t)(values[i - 1] % MODULUS);
        uint32_t new_val = (uint32_t)(values[i + window_size - 1] % MODULUS);
        hash = rolling_hash_advance(hash, old_val, new_val, base_pow);
        output[count++] = hash;
    }
    *out_len = count;
}

uint32_t rolling_hash_base_pow(int window_size) {
    if (window_size <= 0) return 1;
    uint32_t pow = 1;
    for (int i = 0; i < window_size - 1; i++)
        pow = (pow * BASE_HASH) % MODULUS;
    return pow;
}
