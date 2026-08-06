/* wubu_kv_embedding.h — the KV-FS embedding bridge
 *
 * The file system IS the model's context. This module is the bridge:
 * it takes file content (bytes or token IDs), places it into the KV
 * namespace at a resolved path, and provides the coherence signal
 * that tells us whether the model "understands" the file.
 *
 * Three operations:
 *   - ENCODE: file content → /kv/in/<path>  (the model reads this as context)
 *   - DECODE: /kv/synth/<path> → model output (the model writes here)
 *   - COHERENCE: score how well the model attended to a file (the
 *     reward signal for training-on-the-filesystem).
 *
 * The coherence score is computed from the attention weights the model
 * produced when reading tokens from /kv/in/<path>:
 *   - attention_mass: fraction of total attention that stays within
 *     the file's KV region (cross-file attention should be low for
 *     well-understood isolated files, high for related files)
 *   - attention_entropy: how spread the attention is (low = focused =
 *     the model knows what to look at)
 *   - attention_consistency: the attention pattern should be stable
 *     across query reformulations of the same question
 *
 * C11, opaque structs, no third-party deps.
 *
 * Design: docs/wubu1-hive-mind-plan.md §4 Phase 1 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_KV_EMBEDDING_H
#define WUBU_KV_EMBEDDING_H

#include <stddef.h>
#include <stdint.h>
#include "wubu_kvfs.h"  /* the namespace layer G1 */

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct wubu_kv_embedding wubu_kv_embedding_t;

/* A coherence measurement result */
typedef struct {
    float attention_mass;      /* [0, 1]: frac of attention within the file region */
    float attention_entropy;   /* bits: 0 = fully focused, log(n) = uniform */
    float consistency;         /* [0, 1]: stability across queries (1 = perfect) */
    float score;               /* composite: 0.4*mass + 0.3*(1-entropy/ln) + 0.3*consistency */
    int   n_tokens;           /* tokens the file occupied in the namespace */
} wubu_coherence_t;

/* Create an embedding bridge over an existing KV namespace.
 * fs must outlive the embedding object. Returns NULL on failure. */
wubu_kv_embedding_t *wubu_kv_embedding_create(wubu_kvfs_t *fs,
                                               uint32_t block_size);

/* ENCODE: write file content into /kv/in/<path>.
 * The content is byte-level encoded (each byte → token via the simple
 * byte tokenizer). The path is created via wubu_kvfs_mount so the
 * model can read it through the namespace.
 * Returns 0 on success, -1 on namespace failure.
 * Sets *out_n_tokens to the number of tokens written. */
int wubu_kv_embedding_encode(wubu_kv_embedding_t *kv,
                              const char *path,
                              const void *content, size_t content_bytes,
                              size_t *out_n_tokens);

/* ENCODE_TOKENS: write pre-tokenized content into /kv/in/<path>.
 * The caller is responsible for tokenization (using wubu_tokenc).
 * This avoids double-encoding when the pipeline already has tokens.
 * Returns 0 on success, -1 on failure. */
int wubu_kv_embedding_encode_tokens(wubu_kv_embedding_t *kv,
                                     const char *path,
                                     const uint16_t *tokens, size_t n_tokens);

/* DECODE: write model output (tokens) into /kv/synth/<path>.
 * The model synthesizes content and writes it back to the namespace
 * as its "thought" or "plan" about the file at /kv/in/<path>.
 * Returns 0 on success, -1 on failure. */
int wubu_kv_embedding_decode(wubu_kv_embedding_t *kv,
                              const char *path,
                              const uint16_t *tokens, size_t n_tokens);

/* COHERENCE: measure how well the model understood the file at /kv/in/<path>.
 * The attention_weights array is [n_query_tokens][n_context_tokens],
 * the raw attention matrix from the model's forward pass.
 * context_start / context_len locate the file's region in the context.
 * query_start / query_len locate the query tokens that read the file.
 *
 * Returns 0 on success with results in *out, -1 on invalid args.
 * The score is a composite metric [0, 1] (higher = more coherent). */
int wubu_kv_embedding_coherence(const wubu_kv_embedding_t *kv,
                                 const char *path,
                                 const float *attention_weights,
                                 size_t n_query_tokens,
                                 size_t n_context_tokens,
                                 size_t context_start, size_t context_len,
                                 size_t query_start, size_t query_len,
                                 wubu_coherence_t *out);

/* Convenience: look up the KV region for a path.
 * Returns 0 on success (out_block, out_offset filled), -1 if not mounted. */
int wubu_kv_embedding_region(const wubu_kv_embedding_t *kv,
                              const char *path,
                              uint32_t *out_block, size_t *out_offset,
                              size_t *out_n_floats);

/* Free the embedding bridge. Does NOT free fs. */
void wubu_kv_embedding_free(wubu_kv_embedding_t *kv);

/* Returns the number of files in the path registry. */
size_t wubu_kv_embedding_file_count(const wubu_kv_embedding_t *kv);

/* Returns the full KV path for the i-th encoded file (e.g. "/kv/in/doc1.txt").
 * Returns NULL if i is out of range. */
const char *wubu_kv_embedding_get_path(const wubu_kv_embedding_t *kv, size_t i);

/* Returns the number of tokens for the i-th file, or 0 if out of range. */
size_t wubu_kv_embedding_get_n_tokens(const wubu_kv_embedding_t *kv, size_t i);

/* Returns the underlying KV namespace (for shell/direct I/O). */
wubu_kvfs_t *wubu_kv_embedding_get_fs(wubu_kv_embedding_t *kv);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_KV_EMBEDDING_H */
