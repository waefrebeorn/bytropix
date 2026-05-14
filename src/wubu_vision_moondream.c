/**
 * wubu_vision_moondream.c — Moondream3 SigLIP-style ViT forward pass in C
 *
 * Loads dumped f32 weights from moondream3_vision_weights.bin and runs:
 *   patch_embed → 27× ViT block → post_ln → proj_mlp → exp_map → Poincaré
 *
 * Each ViT block:
 *   x = x + attention(layer_norm(x))
 *   x = x + mlp(layer_norm(x))
 *
 * MLP: fc1 → GELU(tanh) → fc2
 * Attention: QKV fused linear → SDPA → proj
 *
 * Weight loading: index.json maps tensor names → offset/size in .bin
 * Weight layout mirrors build_vision_model() in vision.py
 */

#include "wubu_vision_moondream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <json-c/json.h>  /* or manual JSON parser */

// ── Weight Loading ─────────────────────────────────────────────────────

static bool load_json_index(const char *path, void **entries, int *n) {
    // TODO: parse moondream3_vision_index.json
    // For each entry: read offset, size_bytes, shape from JSON
    // Map to vm_weights_t fields by tensor key name
    fprintf(stderr, "[vm] load_json_index: %s\n", path);
    return false;  // stub
}

bool vm_init(vm_state_t *state, const char *bin_path, const char *index_path) {
    memset(state, 0, sizeof(*state));

    // 1. Load index
    // 2. Open binary file
    // 3. For each tensor key:
    //    - Parse model.vision.blocks.N.{submodule}.{param}
    //    - Seek to offset, read size_bytes into the correct vm_weights_t field
    // 4. Allocate scratch buffer for max intermediate tensor

    fprintf(stderr, "[vm] init: weights=%s index=%s\n", bin_path, index_path);
    state->loaded = false;
    return state->loaded;
}

void vm_free(vm_state_t *state) {
    if (!state) return;
    // Free all weight pointers allocated in vm_init
    // Free scratch buffer
    memset(state, 0, sizeof(*state));
}

// ── Core Ops ───────────────────────────────────────────────────────────

void vm_create_patches(const float *pixels, int H, int W, float *patches) {
    // Equivalent to vision.create_patches()
    // Input:  pixels[3][H][W], float32, normalized [-1, 1]
    // Output: patches[729][588]
    //
    // Algorithm:
    //   for each 14×14 patch in the 27×27 grid:
    //     extract [3, 14, 14] → flatten to [588]
    //     store in patches[py * 27 + px]

    const int ps = VM_PATCH_SIZE;
    const int gs = VM_GRID_SIZE;

    for (int py = 0; py < gs; py++) {
        for (int px = 0; px < gs; px++) {
            int patch_idx = py * gs + px;
            float *out = &patches[patch_idx * VM_PATCH_DIM];
            int off = 0;
            for (int c = 0; c < 3; c++) {
                for (int j = 0; j < ps; j++) {
                    for (int i = 0; i < ps; i++) {
                        int y = py * ps + j;
                        int x = px * ps + i;
                        out[off++] = pixels[c * H * W + y * W + x];
                    }
                }
            }
        }
    }
}

void vm_layer_norm(float *x, const float *weight, const float *bias,
                   int n, float eps) {
    // Standard LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= n;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++) {
        x[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

void vm_attention(const float *q, const float *k, const float *v,
                  float *output, int n, int n_heads) {
    // Scaled dot-product attention (single token, all heads)
    // q, k, v: [n] where n = n_heads * head_dim
    // output: [n]
    // Equivalent to F.scaled_dot_product_attention for B=1, T=1

    int head_dim = n / n_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < n_heads; h++) {
        int base = h * head_dim;
        const float *qh = &q[base];
        const float *kh = &k[base];
        const float *vh = &v[base];
        float *out_h = &output[base];

        // For single-query attention: score = q · k / sqrt(d)
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++)
            score += qh[i] * kh[i];
        score *= scale;

        // Softmax (single score → 1.0)
        float attn = 1.0f / (1.0f + expf(-score));  // sigmoid for binary attention
        // In practice with >1 tokens: full softmax over all KVs

        // Weighted sum (simplified for T=1 case)
        for (int i = 0; i < head_dim; i++)
            out_h[i] = vh[i];  // placeholder until multi-token support
    }
}

void vm_linear(const float *x, const float *w, const float *bias,
               float *out, int in_dim, int out_dim) {
    // y = x @ W^T + bias
    // x: [in_dim], w: [out_dim, in_dim] row-major
    for (int i = 0; i < out_dim; i++) {
        float sum = bias ? bias[i] : 0.0f;
        for (int j = 0; j < in_dim; j++) {
            sum += x[j] * w[i * in_dim + j];
        }
        out[i] = sum;
    }
}

void vm_exp_map(const float *vec, float *out, int n, float R) {
    // Exponential map: project Euclidean vector to Poincaré ball
    // exp_0(v) = tanh(||v||/R) * v / (R * ||v||)
    // For ||v|| = 0, returns 0

    float norm = 0.0f;
    for (int i = 0; i < n; i++) norm += vec[i] * vec[i];
    norm = sqrtf(norm);

    if (norm < 1e-8f) {
        for (int i = 0; i < n; i++) out[i] = 0.0f;
        return;
    }

    float factor = tanhf(norm / R) / (R * norm);
    for (int i = 0; i < n; i++) out[i] = vec[i] * factor;
}

// ── Forward Pass ───────────────────────────────────────────────────────

void vm_forward(vm_state_t *state, const float *pixels, int H, int W,
                float *output) {
    if (!state->loaded) {
        fprintf(stderr, "[vm] ERROR: not initialized\n");
        return;
    }

    vm_weights_t *w = &state->w;
    float *scratch = state->scratch;  // large enough for max intermediate

    // Dimension pointers for readability
    int N = VM_N_PATCHES;    // 729
    int D = VM_ENC_DIM;      // 1152
    int F = VM_ENC_FF_DIM;   // 4304
    int Pd = VM_PATCH_DIM;   // 588

    // 1. Create patches: [N, Pd]
    //    patches_ptr starts at scratch[0]
    float *patches = scratch;
    vm_create_patches(pixels, H, W, patches);

    // 2. Patch embedding: Linear(Pd → D) → [N, D]
    //    x starts after patches
    float *x = patches + N * Pd;
    // x = patches @ W.T + bias  for each patch
    for (int i = 0; i < N; i++) {
        vm_linear(&patches[i * Pd], w->patch_emb_weight, w->patch_emb_bias,
                  &x[i * D], Pd, D);
    }

    // 3. Add position embeddings
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            x[i * D + j] += w->pos_emb[i * D + j];
        }
    }

    // 4. Transformer blocks
    float *residual = scratch + (N * Pd + N * D) / sizeof(float);  // temp
    for (int l = 0; l < VM_ENC_N_LAYERS; l++) {
        // Block: x = x + attention(layer_norm(x))
        //        x = x + mlp(layer_norm(x))

        // --- Attention sub-block ---
        memcpy(residual, x, N * D * sizeof(float));

        // LayerNorm
        for (int i = 0; i < N; i++) {
            vm_layer_norm(&x[i * D], w->blocks[l].ln1_weight,
                          w->blocks[l].ln1_bias, D, 1e-5f);
        }

        // QKV projection: [N, D] → [N, 3456] → split into Q, K, V each [N, 1152]
        float *qkv = scratch;  // reuse
        for (int i = 0; i < N; i++) {
            vm_linear(&x[i * D], w->blocks[l].attn_qkv_weight,
                      w->blocks[l].attn_qkv_bias, &qkv[i * 3 * D], D, 3 * D);
        }

        // Split QKV and do attention
        float *q = scratch;
        float *k = scratch + N * D;
        float *v = scratch + 2 * N * D;
        // In-place reshape: qkv [N, 3D] → q [N, D], k [N, D], v [N, D]
        // Already laid out this way since we write to qkv linearly

        // Attention per head (simplified full SDPA)
        float *attn_out = scratch + 3 * N * D;
        for (int i = 0; i < N; i++) {
            vm_attention(&q[i * D], &k[i * D], &v[i * D],
                         &attn_out[i * D], D, VM_ENC_N_HEADS);
        }

        // Output projection
        float *proj_out = scratch;  // reuse qkv space
        for (int i = 0; i < N; i++) {
            vm_linear(&attn_out[i * D], w->blocks[l].attn_proj_weight,
                      w->blocks[l].attn_proj_bias, &proj_out[i * D], D, D);
        }

        // Residual connection
        for (int i = 0; i < N * D; i++)
            x[i] = residual[i] + proj_out[i];

        // --- MLP sub-block ---
        memcpy(residual, x, N * D * sizeof(float));

        // LayerNorm
        for (int i = 0; i < N; i++) {
            vm_layer_norm(&x[i * D], w->blocks[l].ln2_weight,
                          w->blocks[l].ln2_bias, D, 1e-5f);
        }

        // MLP: fc1 [D → F] → GELU → fc2 [F → D]
        float *mlp_hidden = scratch;  // [N, F]
        for (int i = 0; i < N; i++) {
            vm_linear(&x[i * D], w->blocks[l].mlp_fc1_weight,
                      w->blocks[l].mlp_fc1_bias, &mlp_hidden[i * F], D, F);
            // GELU in-place
            for (int j = 0; j < F; j++)
                mlp_hidden[i * F + j] = vm_gelu(mlp_hidden[i * F + j]);
        }

        float *mlp_out = scratch + N * F;  // [N, D]
        for (int i = 0; i < N; i++) {
            vm_linear(&mlp_hidden[i * F], w->blocks[l].mlp_fc2_weight,
                      w->blocks[l].mlp_fc2_bias, &mlp_out[i * D], F, D);
        }

        // Residual connection
        for (int i = 0; i < N * D; i++)
            x[i] = residual[i] + mlp_out[i];
    }

    // 5. Post-layer norm
    for (int i = 0; i < N; i++) {
        vm_layer_norm(&x[i * D], w->post_ln_weight, w->post_ln_bias, D, 1e-5f);
    }

    // 6. Projection MLP: [N, 1152] → concat? → [N, 2304] → fc1 → GELU → fc2 → [N, 2048]
    //    NOTE: The full projection concatenates global_features with "reconstructed"
    //    (from a decoder). For standalone vision encoder, we just project global_features.
    //    The proj_mlp.fc1 expects [2304] input = 2 × 1152 (global + reconstructed).
    //    Since we don't have reconstructed features in standalone mode,
    //    we pad with zeros or use a simpler projection.
    //
    //    For now: feed [global, zeros] through proj_mlp.
    int Pout = VM_PROJ_OUT;
    float *proj_input = scratch;  // [N, 2304]
    for (int i = 0; i < N; i++) {
        memcpy(&proj_input[i * 2 * D], &x[i * D], D * sizeof(float));
        memset(&proj_input[i * 2 * D + D], 0, D * sizeof(float));  // zeros for reconstructed
    }

    float *proj_hidden = scratch + N * 2 * D;  // [N, 8192]
    for (int i = 0; i < N; i++) {
        vm_linear(&proj_input[i * 2 * D], w->proj_mlp_fc1_weight,
                  w->proj_mlp_fc1_bias, &proj_hidden[i * VM_PROJ_INNER],
                  2 * D, VM_PROJ_INNER);
        for (int j = 0; j < VM_PROJ_INNER; j++)
            proj_hidden[i * VM_PROJ_INNER + j] = vm_gelu(proj_hidden[i * VM_PROJ_INNER + j]);
    }

    // Final projection → output
    for (int i = 0; i < N; i++) {
        vm_linear(&proj_hidden[i * VM_PROJ_INNER], w->proj_mlp_fc2_weight,
                  w->proj_mlp_fc2_bias, &output[i * Pout],
                  VM_PROJ_INNER, Pout);
    }

    // 7. Exponential map → Poincaré ball
    float R = 0.956f;  // Default curvature radius
    float *poincare = scratch;  // temp
    for (int i = 0; i < N; i++) {
        vm_exp_map(&output[i * Pout], &poincare[i * Pout], Pout, R);
    }
    memcpy(output, poincare, N * Pout * sizeof(float));
}
