/* wubu_fs_dataset.c — file-system dataset implementation
 *
 * Walks a directory tree, tokenizes each file, encodes it into the
 * KV namespace at /kv/in/<relpath>. Produces batches of token IDs
 * for the trainer. New files are picked up on rescan.
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 3 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#define _POSIX_C_SOURCE 200809L
#include "wubu_fs_dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>

struct wubu_fs_dataset {
    char                    root_dir[4096];
    wubu_tok_hf_t            *tok;
    wubu_kv_embedding_t    *kv;
    uint32_t                block_size;
    wubu_file_entry_t      *files;
    size_t                  n_files;
    size_t                  cap_files;
    /* batch cursor for round-robin iteration */
    size_t                  cursor;
};

/* ---- file table growth ---- */
static int ds_ensure_cap(wubu_fs_dataset_t *ds) {
    if (ds->n_files < ds->cap_files) return 0;
    size_t newcap = ds->cap_files ? ds->cap_files * 2 : 32;
    wubu_file_entry_t *f = (wubu_file_entry_t *)realloc(
        ds->files, newcap * sizeof(wubu_file_entry_t));
    if (!f) return -1;
    ds->files = f;
    ds->cap_files = newcap;
    return 0;
}

wubu_fs_dataset_t *wubu_fs_dataset_create(const char *root_dir,
                                           wubu_tok_hf_t *tok,
                                           wubu_kv_embedding_t *kv,
                                           uint32_t block_size) {
    if (!root_dir || !tok || !kv || block_size == 0) return NULL;
    wubu_fs_dataset_t *ds = (wubu_fs_dataset_t *)calloc(1, sizeof(*ds));
    if (!ds) return NULL;
    strncpy(ds->root_dir, root_dir, sizeof(ds->root_dir) - 1);
    ds->tok = tok;
    ds->kv = kv;
    ds->block_size = block_size;
    ds->cursor = 0;
    /* Scan immediately */
    if (wubu_fs_dataset_rescan(ds) < 0) {
        free(ds);
        return NULL;
    }
    return ds;
}

/* Recursive directory walker — collects file paths */
static void ds_walk_dir(wubu_fs_dataset_t *ds, const char *dirpath,
                         const char *base_relpath, size_t relpath_len) {
    DIR *dir = opendir(dirpath);
    if (!dir) return;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        /* skip . and .. */
        if (entry->d_name[0] == '.' && (entry->d_name[1] == 0 ||
            (entry->d_name[1] == '.' && entry->d_name[2] == 0)))
            continue;
        char fullpath[4096];
        char relpath[512];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", dirpath, entry->d_name);
        if (relpath_len > 0)
            snprintf(relpath, sizeof(relpath), "%s/%s", base_relpath, entry->d_name);
        else
            snprintf(relpath, sizeof(relpath), "%s", entry->d_name);

        struct stat st;
        if (stat(fullpath, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            ds_walk_dir(ds, fullpath, relpath, strlen(relpath));
        } else if (S_ISREG(st.st_mode)) {
            /* Add this file to the table */
            if (ds->n_files >= ds->cap_files && ds_ensure_cap(ds) != 0)
                continue;
            wubu_file_entry_t *fe = &ds->files[ds->n_files];
            strncpy(fe->relpath, relpath, sizeof(fe->relpath) - 1);
            fe->relpath[sizeof(fe->relpath) - 1] = '\0';
            fe->file_size = (size_t)st.st_size;
            /* Tokenize the file */
            FILE *f = fopen(fullpath, "rb");
            if (f) {
                fseek(f, 0, SEEK_END);
                long sz = ftell(f);
                fseek(f, 0, SEEK_SET);
                char *buf = (char *)malloc(sz + 1);
                if (buf) {
                    fread(buf, 1, sz, f);
                    buf[sz] = '\0';
                    int ids[65536];
                    int n = wubu_tok_hf_encode(ds->tok, buf, ids, 65536);
                    if (n < 0) n = 0;
                    /* Encode into the KV namespace */
                    /* Convert int ids to uint16_t */
                    uint16_t *tokens = (uint16_t *)malloc(n * sizeof(uint16_t));
                    if (tokens) {
                        for (int i = 0; i < n; i++)
                            tokens[i] = (uint16_t)ids[i];
                        /* Encode into /kv/in/<relpath> */
                        wubu_kv_embedding_encode_tokens(ds->kv,
                            relpath, tokens, (size_t)n);
                        free(tokens);
                    }
                    fe->n_tokens = (size_t)n;
                }
                free(buf);
                fclose(f);
            } else {
                fe->n_tokens = 0;
            }
            ds->n_files++;
        }
    }
    closedir(dir);
}

int wubu_fs_dataset_rescan(wubu_fs_dataset_t *ds) {
    if (!ds) return -1;
    /* Free old file table */
    free(ds->files);
    ds->files = NULL;
    ds->n_files = 0;
    ds->cap_files = 0;
    ds->cursor = 0;
    /* Walk the directory tree */
    ds_walk_dir(ds, ds->root_dir, "", 0);
    return (int)ds->n_files;
}

int wubu_fs_dataset_files(const wubu_fs_dataset_t *ds,
                           const wubu_file_entry_t **out_entries) {
    if (!ds || !out_entries) return -1;
    *out_entries = ds->files;
    return (int)ds->n_files;
}

int wubu_fs_dataset_next_batch(wubu_fs_dataset_t *ds,
                                int batch_size, int seq_len,
                                wubu_batch_t *out) {
    if (!ds || !out || batch_size <= 0 || seq_len <= 0) return -1;
    if (ds->n_files == 0) return -1;

    memset(out, 0, sizeof(*out));

    /* Collect up to batch_size sequences, each up to seq_len tokens.
     * We read from the KV namespace via the embedding's recorded offsets. */
    int n_seqs = 0;
    /* Allocate token storage: batch_size * seq_len */
    uint16_t *tokens = (uint16_t *)malloc((size_t)batch_size * seq_len * sizeof(uint16_t));
    int *lengths = (int *)calloc((size_t)batch_size, sizeof(int));
    if (!tokens || !lengths) { free(tokens); free(lengths); return -1; }

    for (int i = 0; i < batch_size && ds->n_files > 0; i++) {
        /* Round-robin from cursor */
        size_t idx = ds->cursor % ds->n_files;
        ds->cursor++;
        const wubu_file_entry_t *fe = &ds->files[idx];
        /* Look up the KV region for this file */
        size_t offset, n_floats;
        if (wubu_kv_embedding_region(ds->kv, fe->relpath, NULL, &offset, &n_floats) != 0) {
            /* File not in KV namespace — skip */
            continue;
        }
        /* Read tokens from the KV tensor at offset */
        /* In the real runtime, the executor passes kv_base. For now, we
         * reconstruct from the tokenizer. The KV namespace is the canonical
         * store — the trainer reads tokens from it via the handle API. */
        /* Read file from disk to get tokens (in production, read from KV tensor) */
        char fullpath[4096];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", ds->root_dir, fe->relpath);
        FILE *f = fopen(fullpath, "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        char *buf = (char *)malloc(sz + 1);
        if (!buf) { fclose(f); continue; }
        fread(buf, 1, sz, f);
        buf[sz] = '\0';
        fclose(f);
        int ids[65536];
        int n = wubu_tok_hf_encode(ds->tok, buf, ids, 65536);
        free(buf);
        if (n < 0) n = 0;
        if (n == 0) continue;

        /* Right-truncate or pad to seq_len */
        int copy_len = n < seq_len ? n : seq_len;
        for (int j = 0; j < copy_len; j++)
            tokens[n_seqs * seq_len + j] = (uint16_t)ids[j];
        for (int j = copy_len; j < seq_len; j++)
            tokens[n_seqs * seq_len + j] = 0; /* pad with BOS-like */
        lengths[n_seqs] = copy_len;
        n_seqs++;
    }

    if (n_seqs == 0) {
        free(tokens);
        free(lengths);
        return -1;
    }

    out->tokens = tokens;
    out->lengths = lengths;
    out->n_seqs = n_seqs;
    out->total_tokens = 0;
    for (int i = 0; i < n_seqs; i++)
        out->total_tokens += lengths[i];

    return 0;
}

void wubu_fs_dataset_free_batch(wubu_batch_t *batch) {
    if (!batch) return;
    free(batch->tokens);
    free(batch->lengths);
    memset(batch, 0, sizeof(*batch));
}

void wubu_fs_dataset_free(wubu_fs_dataset_t *ds) {
    if (!ds) return;
    free(ds->files);
    free(ds);
}
