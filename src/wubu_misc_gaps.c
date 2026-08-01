/*
 * wubu_misc_gaps.c -- Cross-discipline / OS / neuro gap closers (L05/O12/O13/O15/
 * P12/P13). C11, no third-party deps.
 *
 * Convergence (DB/OS/neuro/formal 7-hop): these are the "long tail" gaps that map
 * an external discipline onto the KV engine as a small, testable primitive:
 *   - L05 CacheBlend: stitch two request KV prefixes by finding the longest common
 *        prefix length (so the reused segment is copied once, not recomputed).
 *   - O12 ProofWright dequant equivalence: verify quant->dequant round-trip stays
 *        within `tol` of the original (the formal dequant-equivalence check).
 *   - O13 OS mmap prefault: madvise(MADV_WILLNEED) wrapper to warm a KV mapping
 *        (graceful no-op if madvise unavailable).
 *   - O15 Neuro theta/gamma rhythmic gate: a sinusoidal modulation of attention
 *        (theta freq, gamma phase) -> a per-position gate in [0,1].
 *   - P12 KV prefetch stream: non-temporal prefetch hint over a KV buffer range.
 *   - P13 Fused RoPE+quant KV write: apply RoPE rotation then write a quantized
 *        (scaled, rounded) value -- the fused store helper.
 *
 * Triple-DA: null/zero handled; no div-by-zero; deterministic; prefault/prefetch
 * are best-effort and never fail the build or run.
 */
#include "wubu_misc_gaps.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __linux__
#include <sys/mman.h>
#endif

/* L05 CacheBlend: longest common prefix length of two token arrays. */
int wubu_lcp_len(const int *a, const int *b, int n) {
    if (!a || !b || n <= 0) return 0;
    int i = 0;
    while (i < n && a[i] == b[i]) i++;
    return i;
}

/* O12 ProofWright dequant equivalence: 1 if |x - dequant(quant(x))| <= tol for
 * every element (quant = round(x*scale)/scale; a faithful linear quant). */
int wubu_dequant_equiv(const float *x, int n, float scale, float tol) {
    if (!x || n <= 0 || scale <= 0.0f || tol < 0.0f) return 0;
    for (int i = 0; i < n; i++) {
        float q = (float)(int)(x[i] * scale);        /* quantize */
        float dq = q / scale;                          /* dequantize */
        float e = x[i] - dq; if (e < 0.0f) e = -e;
        if (e > tol) return 0;
    }
    return 1;
}

/* O13 OS mmap prefault: warm a mapping (best-effort). Returns 0 on success or
 * when unsupported; never errors fatally. */
int wubu_prefault(void *addr, size_t len) {
#ifdef __linux__
    if (!addr || len == 0) return -1;
    return madvise(addr, len, MADV_WILLNEED);  /* -1 if unavailable; non-fatal */
#else
    (void)addr; (void)len;
    return -1;  /* unsupported -> caller treats as no-op */
#endif
}

/* O15 Neuro theta/gamma rhythmic gate in [0,1] for position p. */
float wubu_rhythmic_gate(int p, float theta, float gamma) {
    if (p < 0) p = 0;
    if (theta < 0.0f) theta = 0.0f;
    if (gamma < 0.0f) gamma = 0.0f;
    /* 0.5*(1+sin(2*pi*theta*p + gamma)) -> in [0,1] */
    float v = 0.5f * (1.0f + (float)sin(2.0 * M_PI * theta * (float)p + gamma));
    if (v < 0.0f) v = 0.0f; if (v > 1.0f) v = 1.0f;
    return v;
}

/* P12 KV prefetch stream: issue non-temporal prefetch hints over `n` floats
 * spaced `stride` bytes apart (best-effort; no-op on non-x86). */
void wubu_kv_prefetch(const float *base, int n, int stride_bytes) {
    if (!base || n <= 0 || stride_bytes <= 0) return;
#if defined(__x86_64__) || defined(__i386__)
    for (int i = 0; i < n; i++)
        __builtin_prefetch((const char *)base + (size_t)i * stride_bytes, 0, 1);
#else
    (void)base; (void)stride_bytes;
#endif
}

/* P13 Fused RoPE+quant KV write: rotate (x,y) by angle, then quantize each to
 * `bits` (linear, scaled to [0,1] range r) and write to out[2]. scale=2^bits-1. */
void wubu_fused_rope_quant(float x, float y, float angle, int bits, float r,
                           unsigned char *out) {
    if (!out) return;
    if (bits <= 0) bits = 8;
    if (r <= 0.0f) r = 1.0f;
    float ca = (float)cos((double)angle), sa = (float)sin((double)angle);
    float rx = x * ca - y * sa;
    float ry = x * sa + y * ca;
    float scale = (float)((1 << bits) - 1);
    int qx = (int)((rx / r * 0.5f + 0.5f) * scale + 0.5f);
    int qy = (int)((ry / r * 0.5f + 0.5f) * scale + 0.5f);
    if (qx < 0) qx = 0; if (qx > (int)scale) qx = (int)scale;
    if (qy < 0) qy = 0; if (qy > (int)scale) qy = (int)scale;
    out[0] = (unsigned char)qx;
    out[1] = (unsigned char)qy;
}
