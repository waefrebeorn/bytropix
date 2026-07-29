/*
 * wubu_kvvq.c -- data-independent residual subvector VQ for KV (doc 014).
 * Self-contained C11. See header.
 */
#include "wubu_kvvq.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Deterministic splitmix64 (constant seed) -> data-independent codebooks. */
static uint64_t kvq_rng_state = 0x9E3779B97F4A7C15ULL;
static void kvq_srand(uint64_t s) { kvq_rng_state = s; }
static uint64_t kvq_rand64(void) {
    uint64_t z = (kvq_rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double kvq_uniform(void) { return (double)kvq_rand64() / (double)0xFFFFFFFFFFFFFFFFULL; }
static float kvq_gauss(void) {
    double u1 = kvq_uniform(); if (u1 < 1e-12) u1 = 1e-12;
    double u2 = kvq_uniform();
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
}

static void subcb_init(wubu_kvvq_subcb_t *s, int bits, int sub_dim, uint64_t seed) {
    s->bits = bits; s->sub_dim = sub_dim; s->n_codewords = 1 << bits;
    s->codebook = (float *)malloc((size_t)s->n_codewords * sub_dim * sizeof(float));
    kvq_srand(seed);
    for (int c = 0; c < s->n_codewords; c++) {
        float *w = s->codebook + (size_t)c * sub_dim;
        double norm = 0.0;
        for (int i = 0; i < sub_dim; i++) { float g = kvq_gauss(); w[i] = g; norm += (double)g*g; }
        norm = sqrt(norm) + 1e-12;
        for (int i = 0; i < sub_dim; i++) w[i] /= (float)norm;
    }
}

int wubu_kvvq_codebook_init(wubu_kvvq_codebook_t *cb, int bits, int head_dim, int n_sub, int n_stages) {
    if (!cb || bits < 1 || bits > WUBU_KVVQ_MAX_BITS || head_dim < 1 || n_sub < 1 ||
        n_stages < 1 || n_stages > WUBU_KVVQ_MAX_STAGES) return -1;
    if (head_dim % n_sub != 0) return -1;
    cb->bits = bits; cb->head_dim = head_dim; cb->n_sub = n_sub;
    cb->sub_dim = head_dim / n_sub; cb->n_stages = n_stages;
    cb->sub = (wubu_kvvq_subcb_t *)malloc((size_t)n_sub * n_stages * sizeof(wubu_kvvq_subcb_t));
    if (!cb->sub) return -1;
    for (int st = 0; st < n_stages; st++)
        for (int s = 0; s < n_sub; s++) {
            int k = st * n_sub + s;
            subcb_init(&cb->sub[k], bits, cb->sub_dim,
                       0x1234ABCDu + (uint64_t)bits*1000003u + (uint64_t)head_dim*7919u
                       + (uint64_t)s*104729u + (uint64_t)st*1299709u);
        }
    return 0;
}

void wubu_kvvq_codebook_free(wubu_kvvq_codebook_t *cb) {
    if (!cb || !cb->sub) return;
    for (int i = 0; i < cb->n_sub * cb->n_stages; i++) if (cb->sub[i].codebook) free(cb->sub[i].codebook);
    free(cb->sub); cb->sub = NULL;
}

void wubu_kvvq_quantize_vec(const float *vec, const wubu_kvvq_codebook_t *cb, int *indices) {
    float *res = (float *)malloc((size_t)cb->head_dim * sizeof(float));
    memcpy(res, vec, (size_t)cb->head_dim * sizeof(float));
    int o = 0;
    for (int st = 0; st < cb->n_stages; st++) {
        for (int s = 0; s < cb->n_sub; s++) {
            const float *sv = res + (size_t)s * cb->sub_dim;
            const wubu_kvvq_subcb_t *sc = &cb->sub[st * cb->n_sub + s];
            int best = 0; float best_d = -1.0f;
            for (int c = 0; c < sc->n_codewords; c++) {
                const float *w = sc->codebook + (size_t)c * cb->sub_dim;
                float d = 0.0f;
                for (int i = 0; i < cb->sub_dim; i++) { float e = sv[i]-w[i]; d += e*e; }
                if (best_d < 0 || d < best_d) { best_d = d; best = c; }
            }
            indices[o++] = best;
            /* subtract quantized codeword -> residual for next stage */
            const float *w = sc->codebook + (size_t)best * cb->sub_dim;
            float *rv = res + (size_t)s * cb->sub_dim;
            for (int i = 0; i < cb->sub_dim; i++) rv[i] -= w[i];
        }
    }
    free(res);
}

void wubu_kvvq_dequant_vec(const int *indices, const wubu_kvvq_codebook_t *cb, float *out) {
    memset(out, 0, (size_t)cb->head_dim * sizeof(float));
    int o = 0;
    for (int st = 0; st < cb->n_stages; st++)
        for (int s = 0; s < cb->n_sub; s++) {
            int idx = indices[o++]; if (idx < 0 || idx >= cb->sub[s].n_codewords) idx = 0;
            const float *w = cb->sub[st * cb->n_sub + s].codebook + (size_t)idx * cb->sub_dim;
            float *ov = out + (size_t)s * cb->sub_dim;
            for (int i = 0; i < cb->sub_dim; i++) ov[i] += w[i];
        }
}

int wubu_kvvq_packed_bytes(int n_vecs, int n_sub, int n_stages, int bits) {
    return (n_vecs * n_sub * n_stages * bits + 7) / 8;
}
void wubu_kvvq_pack(const int *indices, int n_vecs, int n_sub, int n_stages, int bits, uint8_t *out) {
    int bitpos = 0, total = n_vecs * n_sub * n_stages, mask = (1<<bits)-1;
    memset(out, 0, (size_t)wubu_kvvq_packed_bytes(n_vecs, n_sub, n_stages, bits));
    for (int i = 0; i < total; i++) {
        int v = indices[i] & mask;
        for (int b = bits-1; b >= 0; b--) {
            int byte = bitpos>>3, bit = 7-(bitpos&7);
            if ((v>>b)&1) out[byte] |= (uint8_t)(1u<<bit);
            bitpos++;
        }
    }
}
void wubu_kvvq_unpack(const uint8_t *buf, int n_vecs, int n_sub, int n_stages, int bits, int *indices) {
    int bitpos = 0, total = n_vecs * n_sub * n_stages, mask = (1<<bits)-1;
    for (int i = 0; i < total; i++) {
        int v = 0;
        for (int b = bits-1; b >= 0; b--) {
            int byte = bitpos>>3, bit = 7-(bitpos&7);
            v |= ((buf[byte]>>bit)&1) << b; bitpos++;
        }
        indices[i] = v & mask;
    }
}
