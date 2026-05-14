/**
 * hashmind_model.h — HashMind Model Architecture
 *
 * Transformer with dual-source embedding (char + rolling hash).
 * Supports both forward pass (inference) and backward pass (training).
 *
 * Architecture:
 *   Embedding → N Transformer blocks (Self-Attention + FFN) → Output projection
 *   Dual-source: char_embed[D_MODEL] + hash_embed[D_MODEL]
 *   n_heads=4, d_model=64, d_ff=256, d_head=16
 */
#ifndef HASHMIND_MODEL_H
#define HASHMIND_MODEL_H

#include "tokenizer.h"
#include "nn_ops.h"

/* Model hyperparameters */
#define D_MODEL     64
#define N_HEADS     4
#define D_HEAD      (D_MODEL / N_HEADS)  /* 16 */
#define D_FF        (D_MODEL * 4)         /* 256 */
#define N_LAYERS    4
#define CONTEXT_LEN 16
#define HASH_WINDOW 3
#define VOCAB       VOCAB_SIZE

/* ─── HashMind Model Parameters ─── */
typedef struct {
    /* Embeddings */
    float token_embed[VOCAB][D_MODEL];
    float hash_projector[D_MODEL];

    /* Transformer blocks: attention + FFN + layer norms */
    struct {
        /* Attention QKV projection: d_model → 3*d_model */
        float qkv_w[D_MODEL][D_MODEL * 3];
        float out_w[D_MODEL][D_MODEL];

        /* FFN */
        float ffn1_w[D_MODEL][D_FF];
        float ffn2_w[D_FF][D_MODEL];

        /* Layer norms */
        float ln1_gamma[D_MODEL], ln1_beta[D_MODEL];
        float ln2_gamma[D_MODEL], ln2_beta[D_MODEL];
    } blocks[N_LAYERS];

    /* Output projection */
    float out_w[D_MODEL][VOCAB];
} HashMindModel;

/* ─── Gradient buffers (same shape as model) ─── */
typedef struct {
    float token_embed[VOCAB][D_MODEL];
    float hash_projector[D_MODEL];
    struct {
        float qkv_w[D_MODEL][D_MODEL * 3];
        float out_w[D_MODEL][D_MODEL];
        float ffn1_w[D_MODEL][D_FF];
        float ffn2_w[D_FF][D_MODEL];
        float ln1_gamma[D_MODEL], ln1_beta[D_MODEL];
        float ln2_gamma[D_MODEL], ln2_beta[D_MODEL];
    } blocks[N_LAYERS];
    float out_w[D_MODEL][VOCAB];
} HashMindGrad;

/* ─── Optimizer state (momentum for SGD) ─── */
typedef struct {
    float token_embed[VOCAB][D_MODEL];
    float hash_projector[D_MODEL];
    struct {
        float qkv_w[D_MODEL][D_MODEL * 3];
        float out_w[D_MODEL][D_MODEL];
        float ffn1_w[D_MODEL][D_FF];
        float ffn2_w[D_FF][D_MODEL];
        float ln1_gamma[D_MODEL], ln1_beta[D_MODEL];
        float ln2_gamma[D_MODEL], ln2_beta[D_MODEL];
    } blocks[N_LAYERS];
    float out_w[D_MODEL][VOCAB];
} HashMindMomentum;

/* ─── Activation buffers (for backward pass) ─── */
typedef struct {
    /* Input embedding */
    float x[D_MODEL];

    /* After layer-norm 1 */
    float ln1_out[D_MODEL];

    /* QKV */
    float q[N_HEADS][D_HEAD];
    float k[N_HEADS][D_HEAD];
    float v[N_HEADS][D_HEAD];

    /* Attention scores and weights */
    float attn_scores[N_HEADS];
    float attn_weights[N_HEADS];

    /* Attention output (before out_proj) */
    float attn_concat[D_MODEL];

    /* After attention + residual */
    float residual1[D_MODEL];

    /* After layer-norm 2 */
    float ln2_out[D_MODEL];

    /* FFN hidden */
    float ffn_hidden[D_FF];
    float ffn_relu[D_FF];  /* after ReLU */
} BlockActs;

/* ─── Training context ─── */
typedef struct {
    float lr;
    float weight_decay;
    int step;
    HashMindModel* model;
    HashMindGrad* grad;
    HashMindMomentum* vel;
} TrainCtx;

/* ─── API ─── */

/* Initialize model weights (small random) */
void hashmind_model_init(HashMindModel* model);

/* Forward pass for a single token within a sequence context.
 * context_indices[0] = current token, context_indices[1..ctx_len-1] = past tokens.
 * Uses dual-source: char_embed + hash_embed.
 * logits_out[VOCAB] is set. */
void hashmind_forward(HashMindModel* model,
                      const uint32_t* context_hashes, int num_hashes,
                      const int* context_indices, int ctx_len,
                      float* logits_out, BlockActs* acts);

/* Backward pass: given dL/dlogits, accumulate gradients into grad.
 * Must have called hashmind_forward first with same context + non-NULL acts. */
void hashmind_backward(HashMindModel* model, HashMindGrad* grad,
                       const float* dlogits,
                       const uint32_t* context_hashes, int num_hashes,
                       const int* context_indices, int ctx_len,
                       const BlockActs* acts);

/* Apply gradients (SGD with Nesterov momentum + weight decay) */
void hashmind_apply_gradients(TrainCtx* ctx);

/* Zero out gradient buffers */
void hashmind_zero_grad(HashMindGrad* grad);

/* Generate next token (temperature sampling) */
int hashmind_generate(HashMindModel* model,
                      const int* indices, int len,
                      const uint32_t* hashes, float temperature);

/* Save model to binary file */
int hashmind_save(const HashMindModel* model, const char* path);

/* Load model from binary file */
int hashmind_load(HashMindModel* model, const char* path);

/* Count total parameters */
long hashmind_param_count(void);

#endif /* HASHMIND_MODEL_H */
