/* wubu_megakernel.h — Photon 2.0 fused decode megakernel (C11, opaque, minimal)
 *
 * Photon 2.0 (moondream): whole-inference-on-GPU via a single fused
 * megakernel compiled per (model, chip, objective) configuration. The
 * compiler produces a single kernel that fuses attention + FFN + norm
 * into one dispatch, eliminating CPU↔GPU round-trips.
 *
 * We port this as a PSO (Pipeline State Object) pattern: at init time,
 * we pre-compile (jit-compile via function pointer table) a fused decode
 * kernel for each (bits, d_model, n_heads, d_head) configuration. The
 * pre-compiled kernel is cached and called as a single indirect function
 * pointer in the hot path — zero dispatch overhead.
 *
 * Architecture:
 *   - wubu_megakernel_create(cfg): builds the PSO cache for the given
 *     configuration. Allocates fused kernel state.
 *   - wubu_megakernel_decode(mk, ctx, qkv, kv_cache, pos, out):
 *     single-call fused decode — RMSNorm → QKV attention → FFN → residual.
 *     This is the hot path: one function pointer dispatch.
 *
 * The PSO pattern (from wubuwizard-c11-engineering skill):
 *   pso_decode_fast: pre-compiled decode loop for fixed (d, n_heads)
 *   pso_decode: fallback for variable configs
 *
 * Reference: moondream.ai Photon 2.0 blog
 *   Photon outperforms vLLM and SGLang on H100 across Moondream, Qwen, Gemma.
 *   Fused kernel = attention + FFN + all norms in a single GPU kernel launch.
 */
#ifndef WUBU_MEGAKERNEL_H
#define WUBU_MEGAKERNEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque megakernel context handle */
typedef struct wubu_megakernel wubu_megakernel_t;

/* Fused decode configuration (the PSO key) */
typedef struct {
    int d_model;       /* hidden dimension */
    int n_heads;       /* number of query heads */
    int n_kv_heads;    /* number of KV heads (GQA) */
    int d_head;        /* per-head dimension */
    int d_ff;          /* FFN inner dimension */
    int rms_eps;       /* RMSNorm epsilon (as scaled int: eps * 1e6) */
    float rms_epsilon; /* RMSNorm epsilon */
} wubu_megakernel_cfg_t;

/* Create a megakernel PSO context. Pre-compiles the fused decode kernel
 * for the given configuration. Returns NULL on bad args or OOM. */
wubu_megakernel_t *wubu_megakernel_create(const wubu_megakernel_cfg_t *cfg);

/* Destroy context. NULL-safe. */
void wubu_megakernel_free(wubu_megakernel_t *mk);

/* Fused single-token decode — the hot path.
 *
 * Performs in one call:
 *   1. RMSNorm on input
 *   2. QKV projection (linear layers)
 *   3. Causal multi-head attention (GQA with shared KV heads)
 *   4. Residual add
 *   5. RMSNorm
 *   6. FFN (GELU activ + linear)
 *   7. Residual add
 *
 * All operations are fused — no intermediate buffers exposed to caller.
 *
 * ctx:        [d_model] input hidden state (already has KV cached)
 * qkv_weight: [3 * d_model * d_model] QKV projection weights (row-major)
 *             layout: [d_model*d_model | d_model*d_model | d_model*d_model]
 *             (Q | K | V), where K and V share n_kv_heads dimension
 * attn_weight: [d_model * d_model] output projection (O)
 * ffh_weight: [d_ff * d_model] first FFN layer (up-projection)
 * ffo_weight: [d_model * d_ff] second FFN layer (down-projection)
 * rms_norm1:  [d_model] RMSNorm scale (pre-attention)
 * rms_norm2:  [d_model] RMSNorm scale (pre-FFN)
 * kv_cache:   [n_kv_heads * d_head * max_pos] KV cache buffer
 * pos:        current decode position (causal mask limit)
 * out:        [d_model] output hidden state
 *
 * Returns 0 on success, -1 on error. */
int wubu_megakernel_decode(const wubu_megakernel_t *mk,
                            const float *ctx,
                            const float *qkv_weight,
                            const float *attn_weight,
                            const float *ffh_weight,
                            const float *ffo_weight,
                            const float *rms_norm1,
                            const float *rms_norm2,
                            float *kv_cache, int pos,
                            float *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MEGAKERNEL_H */
