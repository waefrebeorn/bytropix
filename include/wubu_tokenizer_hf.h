/*
 * wubu_tokenizer_hf.h -- HuggingFace tokenizer.json loader for wubuwizard.
 *
 * Loads a standard HF BPE tokenizer (Qwen3.6 / Agents-A1 family): NFC
 * normalizer, ByteLevel pre-tokenizer + post-processor, vocab + merges +
 * added_tokens. Self-contained; embeds a tiny JSON scanner (no cJSON dep).
 */
#ifndef WUBU_TOKENIZER_HF_H
#define WUBU_TOKENIZER_HF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_tok_hf wubu_tok_hf_t;

/* Load an HF tokenizer.json. Returns NULL on failure. */
wubu_tok_hf_t *wubu_tok_hf_load(const char *tokenizer_json_path);

/* Free. */
void wubu_tok_hf_free(wubu_tok_hf_t *t);

/* bos/eos ids (from added_tokens or -1 if absent). */
int wubu_tok_hf_bos_id(const wubu_tok_hf_t *t);
int wubu_tok_hf_eos_id(const wubu_tok_hf_t *t);

/* vocab size / id->string (read-only). */
int wubu_tok_hf_vocab_size(const wubu_tok_hf_t *t);
const char *wubu_tok_hf_id_to_str(const wubu_tok_hf_t *t, int id);

/* Encode text -> token ids. Returns count (>=0), or -1 on error.
 * out must hold at least out_cap ints. */
int wubu_tok_hf_encode(const wubu_tok_hf_t *t, const char *text,
                        int *out, int out_cap);

/* Decode token ids -> UTF-8 text (malloc'd, NUL-terminated). Caller frees. */
char *wubu_tok_hf_decode(const wubu_tok_hf_t *t, const int *ids, int n);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_TOKENIZER_HF_H */
