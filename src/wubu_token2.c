/*
 * wubu_token2.c -- the tokenization frontier, complete (IT). C11.
 */
#include "wubu_token2.h"
#include <math.h>
#include <string.h>

float wubu_tok2_bench(long tokens, long chars)
{
    if (tokens <= 0) return 0;
    return (float)chars / (float)tokens;   /* chars per token */
}

int wubu_tok2_remap(const int *old_ids, int n, const int *map, int *out)
{
    if (!old_ids || !map || !out || n <= 0) return -1;
    for (int i = 0; i < n; i++) out[i] = map[old_ids[i]];
    return n;
}

int wubu_tok2_shift(const uint32_t *counts, int n, const uint32_t *ref,
                    float th)
{
    if (!counts || !ref || n <= 0) return -1;
    float diff = 0;
    for (int i = 0; i < n; i++) {
        float a = (float)counts[i], b = (float)ref[i];
        if (b > 0) diff += fabsf(a - b) / b;
    }
    return (diff / n) > th ? 1 : 0;
}

float wubu_tok2_pair_score(long pair_count, long a_count, long b_count)
{
    if (a_count <= 0 || b_count <= 0) return 0;
    /* the BPE score: pair/(a*b) -- the classic merge-pair criterion */
    return (float)pair_count / ((float)a_count * (float)b_count);
}

int wubu_tok2_cache_get(wubu_tok2_cache_t *c, uint64_t key, int fallback)
{
    if (!c) return fallback;
    if (c->valid && c->key == key) return c->n;
    return fallback;
}

void wubu_tok2_cache_put(wubu_tok2_cache_t *c, uint64_t key, int n)
{
    if (!c) return;
    c->key = key; c->n = n; c->valid = 1;
}

int wubu_tok2_norm_guard(const unsigned char *s, int len, int allow_nfd)
{
    if (!s) return 0;
    for (int i = 0; i < len; i++) {
        if (s[i] < 0x80) continue;
        if ((s[i] & 0xE0) == 0xC0) { /* 2-byte */
            if (i + 1 >= len || (s[i + 1] & 0xC0) != 0x80) return 0;
            i++;
        } else if ((s[i] & 0xF0) == 0xE0) {
            if (i + 2 >= len) return 0;
            i += 2;
        } else if ((s[i] & 0xF8) == 0xF0) {
            if (i + 3 >= len) return 0;
            i += 3;
        } else return 0;
    }
    (void)allow_nfd;
    return 1;
}

int wubu_tok2_len_reg(long growth, long cap)
{
    return growth <= cap ? 1 : 0;
}

int wubu_tok2_byte_fallback(const unsigned char *s, int len, int *ok)
{
    if (!s || !ok) return -1;
    *ok = 1;
    for (int i = 0; i < len; i++) {
        if ((s[i] & 0xC0) == 0x80) { /* a stray continuation -> corrupt */
            *ok = 0;
            return i;
        }
    }
    return len;
}

int wubu_tok2_pair_freq(uint32_t *freq, int n, int a, int b)
{
    if (!freq || a < 0 || b < 0 || a >= n || b >= n) return -1;
    freq[(a * n + b) % n]++;   /* the coarse pair bucket */
    return 0;
}

float wubu_tok2_density(long tokens, long embedding_bytes)
{
    if (tokens <= 0) return 0;
    return (float)embedding_bytes / (float)tokens;
}

int wubu_tok2_deterministic(const uint32_t *a, const uint32_t *b, int n)
{
    if (!a || !b || n <= 0) return 0;
    return memcmp(a, b, sizeof(uint32_t) * n) == 0 ? 1 : 0;
}

long wubu_tok2_budget_plan(long prompt_len, float growth, long max_budget)
{
    long est = prompt_len + (long)((float)prompt_len * growth);
    return est > max_budget ? max_budget : est;
}

int wubu_tok2_entity_align(int start, int end, int n_tokens)
{
    return (start >= 0 && end <= n_tokens && start < end) ? 1 : 0;
}

int wubu_tok2_stream(wubu_tok2_stream_t *s, unsigned char byte, uint32_t *tok)
{
    if (!s || !tok) return -1;
    s->acc = (s->acc << 8) | byte;
    s->pending++;
    if (s->pending >= 4) {
        *tok = s->acc;
        s->acc = 0; s->pending = 0;
        return 1;
    }
    return 0;
}

int wubu_tok2_dropout(const uint32_t *ids, int n, float p, uint32_t *out)
{
    if (!ids || !out || n <= 0) return -1;
    int k = 0;
    for (int i = 0; i < n; i++) {
        float r = ((float)((i * 2654435761u) % 1000)) / 1000.0f;
        if (r >= p) out[k++] = ids[i];
    }
    return k;
}

float wubu_tok2_byte_rope(float x, int byte_pos, float theta)
{
    float ang = (float)byte_pos / powf(theta, 2.0f * x);
    return ang;
}

int wubu_tok2_next_n(const uint32_t *ids, int n, int k, uint32_t *out)
{
    if (!ids || !out || k <= 0) return -1;
    int m = n < k ? n : k;
    for (int i = 0; i < m; i++) out[i] = ids[n - m + i];
    return m;
}

int wubu_tok2_trie(const uint32_t *ids, int n, uint32_t prefix, int *depth)
{
    if (!ids || !depth) return -1;
    for (int i = 0; i < n; i++) {
        if (ids[i] == prefix) { *depth = i; return 1; }
    }
    *depth = -1;
    return 0;
}

int wubu_tok2_serialize(const uint32_t *vocab, int n, uint8_t *buf, int cap)
{
    if (!vocab || !buf || cap < n * 4) return -1;
    for (int i = 0; i < n; i++) {
        buf[i * 4 + 0] = (uint8_t)(vocab[i] & 0xFF);
        buf[i * 4 + 1] = (uint8_t)((vocab[i] >> 8) & 0xFF);
        buf[i * 4 + 2] = (uint8_t)((vocab[i] >> 16) & 0xFF);
        buf[i * 4 + 3] = (uint8_t)((vocab[i] >> 24) & 0xFF);
    }
    return n * 4;
}

float wubu_tok2_pair_health(long merges, long total)
{
    if (total <= 0) return 0;
    return (float)merges / (float)total;
}

long wubu_tok2_skip_redundant(long tokens, float redundancy)
{
    if (redundancy < 0) redundancy = 0;
    if (redundancy > 1) redundancy = 1;
    return tokens - (long)((float)tokens * redundancy);
}

int wubu_tok2_fallback(const unsigned char *s, int len, uint32_t *out)
{
    if (!s || !out || len <= 0) return -1;
    for (int i = 0; i < len; i++) out[i] = s[i];   /* byte-level ids */
    return len;
}

float wubu_tok2_coverage(long in_vocab, long total)
{
    if (total <= 0) return 0;
    return (float)in_vocab / (float)total;
}

int wubu_tok2_watermark(const uint32_t *ids, int n, uint32_t key)
{
    if (!ids || n <= 0) return 0;
    /* a boundary watermark: the xor-accumulated check */
    uint32_t acc = key;
    for (int i = 0; i < n; i++) acc ^= ids[i] + 0x9e3779b9u;
    return acc == key ? 1 : 0;
}
