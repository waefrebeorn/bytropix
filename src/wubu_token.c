/*
 * wubu_token.c -- tokenization frontier (Theme IT). C11.
 */
#include "wubu_token.h"
#include <math.h>
#include <string.h>

int wubu_tok_bit_bpe_cost(int byte_len, int bits_per_symbol)
{
    if (byte_len < 0 || bits_per_symbol <= 0) return -1;
    /* the bytes above the entropy floor cost the symbol bits */
    return byte_len * 8 + byte_len * bits_per_symbol / 8;
}

int wubu_tok_utf8_embed(const unsigned char *s, int len, float *out, int d)
{
    if (!s || !out || d <= 0) return -1;
    /* byte-level hashing into the embedding dims (tokenizer-free) */
    for (int i = 0; i < d; i++) out[i] = 0;
    for (int i = 0; i < len; i++) {
        int h = (s[i] * 2654435761u) % d;
        out[h] += (i % 2) ? 0.5f : -0.5f;
    }
    return len;
}

float wubu_tok_entropy_merge(const uint32_t *counts, int n)
{
    if (!counts || n <= 0) return 0;
    uint64_t total = 0;
    for (int i = 0; i < n; i++) total += counts[i];
    if (total == 0) return 0;
    double h = 0;
    for (int i = 0; i < n; i++) {
        if (counts[i] == 0) continue;
        double p = (double)counts[i] / (double)total;
        h -= p * log(p);
    }
    return (float)h;
}

int wubu_tok_density_window(int tokens, float density, int max_window)
{
    if (tokens <= 0 || density <= 0 || max_window <= 0) return 0;
    /* dense context -> a shorter effective window (the information is packed) */
    int w = (int)((float)tokens * (1.0f - 0.5f * density));
    return w > max_window ? max_window : (w < 1 ? 1 : w);
}

int wubu_tok_cache_get(wubu_tok_cache_t *c, uint32_t key, int fallback)
{
    if (!c) return fallback;
    if (c->valid && c->key == key) return c->n;
    return fallback;
}

void wubu_tok_cache_put(wubu_tok_cache_t *c, uint32_t key, int n)
{
    if (!c) return;
    c->key = key; c->n = n; c->valid = 1;
}

int wubu_tok_prune(const int *used, int vocab, int *remap, int *kept)
{
    if (!used || !remap || vocab <= 0) return -1;
    int next = 0;
    for (int i = 0; i < vocab; i++) {
        if (used[i]) remap[i] = next++;
        else remap[i] = -1;
    }
    if (kept) *kept = next;
    return 0;
}

int wubu_tok_roundtrip(const unsigned char *s, int len,
                       const unsigned char *back, int back_len)
{
    if (!s || !back) return 0;
    if (len != back_len) return 0;
    return memcmp(s, back, (size_t)len) == 0 ? 1 : 0;
}

float wubu_tok_efficiency(int tokens, int info_bits)
{
    if (tokens <= 0) return 0;
    return (float)info_bits / (float)tokens;
}

int wubu_tok_oov(int token_id, int vocab, int fallback_id)
{
    if (token_id >= 0 && token_id < vocab) return token_id;
    return fallback_id >= 0 ? fallback_id : -1;
}

size_t wubu_tok_entropy_size(const uint32_t *counts, int n, long total)
{
    if (!counts || n <= 0 || total <= 0) return 0;
    double bits = 0;
    for (int i = 0; i < n; i++) {
        if (counts[i] == 0) continue;
        double p = (double)counts[i] / (double)total;
        bits += (double)counts[i] * (-log2(p));
    }
    return (size_t)(bits / 8.0);
}
