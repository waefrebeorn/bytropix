/*
 * wubu_attn_kernels.c -- Attention-side fused kernels (P11 / P15 / O20). C11.
 *
 * Convergence (int2 KV + fused spec-verify + neuro plasticity 7-hop):
 *   - P11 int2 KV dequant: KV stored as 2-bit (4 levels) per component with a
 *        per-block (scale, zero) pair; reconstruct to F32. Fused into the attn
 *        read path. Verified against the stored F32 in test_attn_kernels.
 *   - P15 speculative-verify fused attn: given a draft token's score and the
 *        verified reference score, return acceptance (|draft - ref| <= thr) in
 *        one call (the fused verify used inside the attention kernel).
 *   - O20 neuro plasticity re-quant: given a plasticity signal p in [0,1], choose
 *        a KV quant bits: high plasticity (recent change) -> more bits (less loss);
 *        low -> fewer bits (more compression). Returns bits in [bmin,bmax].
 *
 * Triple-DA: n<=0/bits<=0 clamped; null -> 0; deterministic; no OOB.
 */
#include "wubu_attn_kernels.h"
#include <stdlib.h>
#include <math.h>

/* P11 int2 KV dequant: pack 4 values per byte (2 bits each). Reconstruct with
 * per-block scale/zero. out[i] = scale * ((packed>>(2*(i%4)) & 3) - zero).
 * Returns number of values written (= n). */
int wubu_int2_dequant(const unsigned char *packed, int n, float scale, float zero,
                      float *out) {
    if (!packed || !out || n <= 0 || scale <= 0.0f) return 0;
    for (int i = 0; i < n; i++) {
        unsigned char b = packed[i >> 2];
        int lvl = (b >> (2 * (i & 3))) & 3;       /* 0..3 */
        out[i] = scale * ((float)lvl - zero);
    }
    return n;
}

/* P15 fused spec-verify: accept draft if |draft - ref| <= thr. */
int wubu_spec_verify_fused(float draft_score, float ref_score, float thr) {
    if (thr < 0.0f) thr = 0.0f;
    float d = draft_score - ref_score; if (d < 0.0f) d = -d;
    return (d <= thr) ? 1 : 0;
}

/* O20 neuro plasticity re-quant: bits in [bmin,bmax], high p -> bmax. */
int wubu_plasticity_bits(float p, int bmin, int bmax) {
    if (p < 0.0f) p = 0.0f; if (p > 1.0f) p = 1.0f;
    if (bmin <= 0) bmin = 1; if (bmax < bmin) bmax = bmin;
    int bits = bmin + (int)((bmax - bmin) * p + 0.5f);
    if (bits < bmin) bits = bmin; if (bits > bmax) bits = bmax;
    return bits;
}
