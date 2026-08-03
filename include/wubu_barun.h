/*
 * wubu_barun.h -- BarunLM-35M, our base model, ported to C11. THE MUSTARD SEED.
 *
 * BarunLM-35M (Apache-2.0, (c) 2026 Harshal Singh) is a 35,072,768-parameter
 * decoder-only base language model. We port it into the wubuwizard engine as
 * our own second brain: the seed that grows via the AGI brain-cluster loop --
 * more tokens, more parameters, more knowledge, all designed and trained
 * in-house (the "there is no third party" doctrine).
 *
 * Architecture (faithful to the reference):
 *   - 12 layers, dim 448, 7 query heads, 1 KV head (GQA 7:1)
 *   - hybrid attention: 3 local (256-token window) layers then 1 full layer
 *   - 50% partial RoPE (rotary on the first rope_dim=32 of head_dim=64)
 *   - QK RMSNorm, gated attention outputs (sigmoid gate), bounded SwiGLU
 *     (activation clip 10), residual selectors every 4 layers
 *   - tied embeddings, vocab 16,384, context 2,048
 *   - Muon-optimizer training (the reference recipe: lr 1e-4, wd 0.1)
 *
 * Pure C11, opaque struct, no third-party deps. The forward pass runs on
 * CPU (hosted) and will run on metal via the same kernel-dispatch path.
 */
#ifndef WUBU_BARUN_H
#define WUBU_BARUN_H

#include <stdint.h>
#include <stddef.h>

/* The released configuration (barun_config.json). */
#define BARUN_VOCAB       16384
#define BARUN_DIM         448
#define BARUN_LAYERS      12
#define BARUN_HEADS       7
#define BARUN_KV_HEADS    1
#define BARUN_HEAD_DIM    64
#define BARUN_ROPE_DIM    32
#define BARUN_FFN_DIM     1228
#define BARUN_MAX_SEQ     2048
#define BARUN_LOCAL_WIN   256
#define BARUN_FULL_EVERY  4
#define BARUN_SELECT_EVERY 4
#define BARUN_CLIP        10.0f
#define BARUN_EPS         1e-6f
#define BARUN_SELECTORS   3   /* 12 / 4 */

/* the exact released parameter count */
#define BARUN_PARAMS      35072768

/* A single transformer block's weights (local OR full attention). */
typedef struct {
    /* attention */
    float *q_proj;   /* [dim, heads*head_dim] = [448, 448] */
    float *k_proj;   /* [dim, kv_heads*head_dim] = [448, 64] */
    float *v_proj;   /* [448, 64] */
    float *o_proj;   /* [448, 448] */
    float *g_proj;   /* [448, 448] (attention gate) */
    float *q_norm;   /* [64] */
    float *k_norm;   /* [64] */
    float *attn_norm;/* [448] */
    /* ffn (bounded SwiGLU) */
    float *gate_up;  /* [448, 2*1228] = [448, 2456] */
    float *down;     /* [1228, 448] */
    float *ffn_norm; /* [448] */
} barun_block_t;

/* The full model. */
typedef struct {
    float *embedding;       /* [16384, 448] (tied with lm_head) */
    float *final_norm;      /* [448] */
    barun_block_t blocks[BARUN_LAYERS];
    float *selectors[BARUN_SELECTORS];  /* [448] each (score weight) */
    int    is_full[BARUN_LAYERS];       /* attention rhythm */
} barun_model_t;

/* An inference buffer (all the working memory, allocated once). */
typedef struct {
    float *x;        /* [seq, 448] the hidden stream */
    float *x2;       /* [seq, 448] scratch */
    float *q;        /* [seq, heads*head_dim] */
    float *k;        /* [seq, kv*head_dim] */
    float *v;        /* [seq, kv*head_dim] */
    float *attn_out; /* [seq, 448] */
    float *gate;     /* [seq, 448] */
    float *ffn_gate; /* [seq, ffn_dim] */
    float *ffn_up;   /* [seq, ffn_dim] */
    float *ffn_out;  /* [seq, 448] */
    float *logits;   /* [seq, vocab] */
    float *cos_tbl;  /* [max_seq, rope_dim] */
    float *sin_tbl;  /* [max_seq, rope_dim] */
    float *cache_k;  /* [layers][max_seq, 64] */
    float *cache_v;  /* [layers][max_seq, 64] */
    size_t seq_alloc;
} barun_buf_t;

/* B1: init the model from raw weight buffers (the safetensors loader
 * fills them; the model takes ownership of the pointers). */
int barun_model_init(barun_model_t *m, float *embedding, float *final_norm,
                     barun_block_t *blocks, float **selectors);

/* B2: load the released checkpoint from a safetensors file. */
int barun_load(barun_model_t *m, const char *safetensors_path);

/* B3: allocate the inference buffer for a given max sequence. */
int barun_buf_alloc(barun_buf_t *b, size_t max_seq);

/* B4: the forward pass -- full sequence, causal. */
int barun_forward(barun_model_t *m, barun_buf_t *b,
                  const uint16_t *tokens, size_t n_tokens);

/* B5: greedy + temperature generation. */
size_t barun_generate(barun_model_t *m, barun_buf_t *b,
                      uint16_t *tokens, size_t n_prompt, size_t max_new,
                      float temperature, uint32_t seed);

/* B6: the logits for the last token (the caller samples). */
float *barun_last_logits(barun_buf_t *b);

/* B7: the cross-entropy loss + the Muon optimizer step (training). */
float barun_loss(barun_buf_t *b, const uint16_t *tokens, size_t n_tokens);
int  barun_muon_step(barun_model_t *m, float lr, float weight_decay);

/* B8: the parameter count sanity check (must be 35,072,768). */
long barun_parameter_count(const barun_model_t *m);

/* B9: free everything. */
void barun_free(barun_model_t *m, barun_buf_t *b);

#endif