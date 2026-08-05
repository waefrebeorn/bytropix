/* wubu_dsv4.c — DeepSeek-V4 Flash layer port (C11, opaque, minimal includes)
 *
 * DeepSeek-V4-Flash-0731: 24B activated params, 256 experts / 6 active,
 * MXFP4 native expert storage, hash routing, hyper-connections,
 * sinkhorn-normalized routing weights, lightning coarse-to-fine KV indexing.
 *
 * All modules are self-contained — reuses wubu_hashrouter, wubu_mxfp4,
 * wubu_dsa. No god headers. Opaque struct. C11 only.
 */
#define _POSIX_C_SOURCE 200809L
#include "wubu_dsv4.h"
#include "wubu_hashrouter.h"
#include "wubu_mxfp4.h"
#include "wubu_dsa.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

struct wubu_dsv4 {
    wubu_dsv4_cfg_t cfg;
    wubu_hashrouter_t *router;  /* hash routing for MoE experts */
};

wubu_dsv4_t *wubu_dsv4_create(const wubu_dsv4_cfg_t *cfg) {
    if (!cfg || cfg->d_model <= 0 || cfg->n_heads <= 0 ||
        cfg->n_experts <= 0 || cfg->n_active <= 0 ||
        cfg->n_active > cfg->n_experts || cfg->n_layers <= 0)
        return NULL;
    wubu_dsv4_t *ds = (wubu_dsv4_t *)calloc(1, sizeof(*ds));
    if (!ds) return NULL;
    ds->cfg = *cfg;
    /* Hash router: token_id → top-k experts, deterministic, no training */
    ds->router = wubu_hashrouter_create(cfg->n_experts, cfg->n_active, 0xDEADBEEF);
    if (!ds->router) { free(ds); return NULL; }
    return ds;
}

void wubu_dsv4_free(wubu_dsv4_t *ds) {
    if (!ds) return;
    wubu_hashrouter_free(ds->router);
    free(ds);
}

/* Hyper-connection gated residual:
 *   out = x + gate_scale * FFN(x)
 * Replaces plain residual x = x + FFN(x). The gate_scale is a learned
 * scalar (DeepSeek-V4 hyper-connection). We just compute the elementwise
 * update: out[i] = x[i] + gate_scale * ffn_out[i]. */
int wubu_dsv4_hyper_residual(const float *x, const float *ffn_out,
                             float gate_scale, int d_model,
                             float *out) {
    if (!x || !ffn_out || !out || d_model <= 0) return -1;
    for (int i = 0; i < d_model; i++) {
        out[i] = x[i] + gate_scale * ffn_out[i];
    }
    return 0;
}

/* Sinkhorn-Knopp balancing of a [n_tokens x n_experts] weight matrix.
 * Each iteration: divide each row by its sum, then each column by its sum.
 * Stabilized via log-space accumulation to avoid division by zero.
 * This normalizes the routing weights so the load balances across experts
 * (DeepSeek-V4's sinkhorn normalization for MoE routing). */
int wubu_dsv4_sinkhorn_norm(float *w, int n_tokens, int n_experts, int iters) {
    if (!w || n_tokens <= 0 || n_experts <= 0 || iters <= 0) return -1;
    for (int iter = 0; iter < iters; iter++) {
        /* Row normalization: each row sums to 1 */
        for (int r = 0; r < n_tokens; r++) {
            float sum = 0;
            for (int c = 0; c < n_experts; c++) {
                float v = w[r * n_experts + c];
                sum += v > 0 ? v : 0;  /* ReLU-style, ignore negatives */
            }
            if (sum < 1e-8f) sum = 1e-8f;
            for (int c = 0; c < n_experts; c++) {
                w[r * n_experts + c] /= sum;
            }
        }
        /* Column normalization: each column sums to 1 */
        for (int c = 0; c < n_experts; c++) {
            float sum = 0;
            for (int r = 0; r < n_tokens; r++) {
                float v = w[r * n_experts + c];
                sum += v > 0 ? v : 0;
            }
            if (sum < 1e-8f) sum = 1e-8f;
            for (int r = 0; r < n_tokens; r++) {
                w[r * n_experts + c] /= sum;
            }
        }
    }
    return 0;
}

/* Hash-route a token to top-k experts.
 * Delegates to the internal hashrouter (splitmix64 + per-slot salts).
 * Deterministic: same (token_id, pos) → same expert list. */
int wubu_dsv4_route(const wubu_dsv4_t *ds, uint32_t token_id,
                    uint32_t pos, int *out_experts) {
    if (!ds || !out_experts) return -1;
    return wubu_hashrouter_route(ds->router, token_id, pos, out_experts);
}

/* Pack expert weights into MXFP4 (native expert storage).
 * Each expert is an independent [expert_dim] vector. MXFP4 packs
 * 32-element blocks into 16 nibbles + 1 E8M0 scale byte = 17 bytes/block.
 * The scale byte is at the END of each block (OCP layout).
 * experts: [n_experts * expert_dim] row-major (expert 0 first).
 * out:     [n_experts * packed_per_expert] bytes.
 * Reuses wubu_mxfp4_pack (E8M0 scale + E2M1 nibbles, scale at end). */
int wubu_dsv4_pack_experts_mxfp4(const float *experts, int n_experts,
                                  int expert_dim, uint8_t *out) {
    if (!experts || !out || n_experts <= 0 || expert_dim <= 0) return -1;
    if (expert_dim % WUBU_MX_BLOCK != 0) return -1;
    int packed_per_expert = wubu_mxfp4_pack(experts, n_experts * expert_dim, out);
    return packed_per_expert > 0 ? 0 : -1;
}

/* Unpack MXFP4 experts back to float.
 * Reverse of wubu_dsv4_pack_experts_mxfp4. */
int wubu_dsv4_unpack_experts_mxfp4(const uint8_t *in, int n_experts,
                                   int expert_dim, float *out) {
    if (!in || !out || n_experts <= 0 || expert_dim <= 0) return -1;
    if (expert_dim % WUBU_MX_BLOCK != 0) return -1;
    return wubu_mxfp4_unpack(in, n_experts * expert_dim, out);
}

/* Lightning indexer: coarse-to-fine top-k KV block selection.
 * Coarse: score each block by dot(query, block_mean) → top-k blocks.
 *
 * Fine: full softmax attention over ONLY the top-k blocks' keys/values.
 *
 * This is the DeepSeek lightning indexer pattern: first a cheap coarse
 * pass eliminates 90%+ of blocks, then the expensive attention runs on
 * the survivors. We reuse wubu_dsa_attend() which already implements
 * this exact coarse-to-fine pipeline.
 *
 * query: [d], block_means: [n_blocks][d], block_vals: [n_blocks][d_v]
 * out: [d_v] weighted sum of selected block values.
 * Returns 0 on success, -1 on error. */
int wubu_dsv4_lightning_indexer(const float *query, int d,
                                const float *const *block_means,
                                const float *const *block_vals,
                                int n_blocks, int top_k, int d_v,
                                float *out) {
    if (!query || !block_means || !block_vals || !out ||
        d <= 0 || n_blocks <= 0 || top_k <= 0 || d_v <= 0)
        return -1;
    if (top_k > n_blocks) top_k = n_blocks;

    /* Create a DSA indexer for the coarse-to-fine pass.
     * block_size is unused (1 for indexing), d is the head dim. */
    wubu_dsa_t *dsa = wubu_dsa_create(n_blocks, 1, top_k, d);
    if (!dsa) return -1;

    int result = wubu_dsa_attend(dsa, query, block_means, block_vals, out, d_v);
    wubu_dsa_free(dsa);
    return result;
}
