/* wubu_fs_dataset.h — file-system dataset for training on the KV namespace
 *
 * The KV cache is a file system. This module walks a directory of files,
 * tokenizes them, encodes them into the KV namespace at /kv/in/<path>,
 * and produces batches of token IDs that the trainer feeds to the model.
 *
 * The dataset IS the filesystem: any file inserted into the watched
 * directory appears in /kv/in/ on the next batch. The model's coherence
 * score over /kv/in/ is the training reward (see wubu_coherence_reward).
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 3 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_FS_DATASET_H
#define WUBU_FS_DATASET_H

#include <stddef.h>
#include <stdint.h>
#include "wubu_kv_embedding.h"
#include "wubu_tokenizer_hf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct wubu_fs_dataset wubu_fs_dataset_t;

/* A batch of token sequences */
typedef struct {
    uint16_t *tokens;    /* flattened token IDs, total_tokens length */
    int       *lengths;  /* per-sequence length */
    int        n_seqs;
    int        total_tokens;
} wubu_batch_t;

/* A dataset entry (one file) */
typedef struct {
    char    relpath[512];  /* path relative to root, e.g. "src/foo.c" */
    size_t  n_tokens;      /* token count */
    size_t  file_size;     /* original file size in bytes */
} wubu_file_entry_t;

/* Create a dataset over a root directory, using the given tokenizer and
 * KV embedding bridge. The dataset walks the directory tree and encodes
 * each file into /kv/in/<relpath>.
 *
 * block_size: floats per KV block (passed to kv_embedding).
 * Returns NULL on failure. */
wubu_fs_dataset_t *wubu_fs_dataset_create(const char *root_dir,
                                           wubu_tok_hf_t *tok,
                                           wubu_kv_embedding_t *kv,
                                           uint32_t block_size);

/* Re-scan the directory tree. New files are encoded into /kv/in/.
 * Removed files are unmounted. Returns the number of new/missing files. */
int wubu_fs_dataset_rescan(wubu_fs_dataset_t *ds);

/* Get the list of file entries in the dataset.
 * Returns n_entries; caller must NOT free the returned array (owned by ds). */
int wubu_fs_dataset_files(const wubu_fs_dataset_t *ds,
                           const wubu_file_entry_t **out_entries);

/* Produce the next training batch of up to batch_size sequences,
 * each up to seq_len tokens (right-truncated). Tokens are read from
 * the KV namespace via the embedding's encode_tokens path.
 * Returns 0 on success, -1 if no more data (rewinds automatically). */
int wubu_fs_dataset_next_batch(wubu_fs_dataset_t *ds,
                                int batch_size, int seq_len,
                                wubu_batch_t *out);

/* Free a batch (frees the tokens/lengths arrays). */
void wubu_fs_dataset_free_batch(wubu_batch_t *batch);

/* Free the dataset. Does NOT free tok or kv (caller-owned). */
void wubu_fs_dataset_free(wubu_fs_dataset_t *ds);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_FS_DATASET_H */
