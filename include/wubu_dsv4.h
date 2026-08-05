/* wubu_dsv4.h — DeepSeek-V4 Flash layer port (C11, opaque, minimal includes)
 *
 * DeepSeek-V4-Flash-0731: 24B activated params, 256 experts / 6 active,
 * MXFP4 native expert storage, hash routing, hyper-connections,
 * sinkhorn-normalized routing weights, lightning coarse-to-fine KV indexing.
 *
 * Reuses existing modules: wubu_hashrouter (hash routing), wubu_mxfp4
 * (MXFP4 pack/unpack), wubu_dsa (lightning coarse-to-fine indexer).
 *
 * Design: the DSV4 layer is a MoE transformer block with:
 *   1. Hyper-connection: gated residual x = x + gate_scale * FFN(x)
 *      (DeepSeek-V4 replaces plain residual with a learned scalar gate)
 *   2. Hash routing: token_id + position → top-k experts via splitmix64
 *      (reuses wubu_hashrouter; no learned router weights)
 *   3. MXFP4 experts: experts stored in MXFP4 natively (E2M1 + E8M0)
 *      (reuses wubu_mxfp4_pack/unpack)
 *   4. Sinkhorn normalization: normalize the expert routing logits via
 *      2-3 iterations of Sinkhorn balancing for load distribution
 *   5. Lightning indexer: coarse-to-fine top-k block selection
 *      (extends wubu_dsa with a second-pass refinement)
 *
 * Reference: AtomicChat/DeepSeek-V4-Flash-0731-GGUF (huggingface)
 */
#ifndef WUBU_DSV4_H
#define WUBU_DSV4_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque DeepSeek-V4 layer handle */
typedef struct wubu_dsv4 wubu_dsv4_t;

/* Configuration for a DSV4 layer */
typedef struct {
    int d_model;       /* hidden dimension */
    int n_heads;       /* number of attention heads */
    int n_experts;     /* total number of MoE experts (256) */
    int n_active;      /* active experts per token (6) */
    int n_layers;      /* number of layers */
} wubu_dsv4_cfg_t;

/* Create a DSV4 layer context. Returns NULL on bad args or OOM. */
wubu_dsv4_t *wubu_dsv4_create(const wubu_dsv4_cfg_t *cfg);

/* Destroy the layer context. NULL-safe. */
void wubu_dsv4_free(wubu_dsv4_t *ds);

/* Hyper-connection gated residual:
 *   out = x + gate_scale * FFN(x)
 * where gate_scale is a per-layer learned scalar (here, passed in).
 * This replaces the plain residual x = x + FFN(x) from vanilla transformers.
 * x: [d_model], ffn: [d_model], gate_scale: scalar
 * Returns 0 on success, -1 on error. */
int wubu_dsv4_hyper_residual(const float *x, const float *ffn_out,
                             float gate_scale, int d_model,
                             float *out);

/* Sinkhorn normalization of a routing weight matrix.
 * Performs `iters` iterations of Sinkhorn-Knopp balancing on the
 * [n_tokens x n_experts] matrix w in-place. Each iteration scales
 * rows and columns to sum to 1. Stabilized via log-sum-exp.
 * Returns 0 on success, -1 on error. */
int wubu_dsv4_sinkhorn_norm(float *w, int n_tokens, int n_experts, int iters);

/* Hash-route a token to top-k experts.
 * Uses the internal hashrouter (splitmix64 with per-slot salts).
 * token_id: the token identifier
 * pos: position in sequence
 * out_experts: [n_active] expert ids (guaranteed distinct)
 * Returns n_active on success, -1 on error. */
int wubu_dsv4_route(const wubu_dsv4_t *ds, uint32_t token_id,
                    uint32_t pos, int *out_experts);

/* Pack expert weights into MXFP4 (native expert storage).
 * experts: [n_experts * expert_dim] float weights
 * out:    [n_experts * (mx4_packed_size(expert_dim))] packed MXFP4
 * Returns 0 on success, -1 on error. */
int wubu_dsv4_pack_experts_mxfp4(const float *experts, int n_experts,
                                  int expert_dim, uint8_t *out);

/* Unpack MXFP4 experts back to float.
 * in:  [n_experts * mx4_packed_size(expert_dim)]
 * out: [n_experts * expert_dim] float weights
 * Returns 0 on success, -1 on error. */
int wubu_dsv4_unpack_experts_mxfp4(const uint8_t *in, int n_experts,
                                   int expert_dim, float *out);

/* Lightning indexer: coarse-to-fine top-k KV block selection.
 * Coarse stage scores all blocks by block-mean similarity (dot product
 * with query), then fine stage runs full attention over top-k blocks.
 * query: [d], block_means: [n_blocks][d], block_vals: [n_blocks][d_v]
 * out: [d_v] weighted sum
 * Returns 0 on success, -1 on error. */
int wubu_dsv4_lightning_indexer(const float *query, int d,
                                const float *const *block_means,
                                const float *const *block_vals,
                                int n_blocks, int top_k, int d_v,
                                float *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_DSV4_H */
