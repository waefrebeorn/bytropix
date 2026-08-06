/* wubu_kv_shell.c — shell command routing to KV/FS
 *
 * Exposes the KV filesystem as a shell-accessible namespace.
 * Routes `ls`, `cat`, `stat` commands through the KV embedding layer
 * so the shell sees the same context the model sees during forward.
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 11 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_kv_shell.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

struct wubu_kv_shell {
    wubu_kv_embedding_t *kv;
    wubu_fs_dataset_t   *dataset;  /* may be NULL */
    float               *kv_base;  /* flat KV tensor (may be NULL) */
};

wubu_kv_shell_t *wubu_kv_shell_create(wubu_kv_embedding_t *kv,
                                       wubu_fs_dataset_t *dataset,
                                       float *kv_base) {
    if (!kv) return NULL;
    wubu_kv_shell_t *s = (wubu_kv_shell_t *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->kv = kv;
    s->dataset = dataset;
    s->kv_base = kv_base;
    return s;
}

void wubu_kv_shell_free(wubu_kv_shell_t *s) {
    if (!s) return;
    free(s);
}

/* List files in a KV directory */
static int shell_ls(wubu_kv_shell_t *shell, const char *path,
                     char *out, size_t out_cap) {
    const char *prefix = "/kv/in/";
    const char *rel = path;
    if (strncmp(path, prefix, strlen(prefix)) == 0)
        rel = path + strlen(prefix);

    size_t n_files = wubu_kv_embedding_file_count(shell->kv);
    int out_len = 0;
    for (size_t i = 0; i < n_files; i++) {
        const char *fname = wubu_kv_embedding_get_path(shell->kv, i);
        if (!fname) continue;
        const char *frel = fname;
        if (strncmp(fname, prefix, strlen(prefix)) == 0)
            frel = fname + strlen(prefix);

        int match = 0;
        if (rel[0] == '\0' || strcmp(rel, "/") == 0) {
            if (strchr(frel, '/') == NULL)
                match = 1;
        } else {
            size_t rel_len = strlen(rel);
            if (rel[rel_len - 1] == '/') rel_len--;
            if (strncmp(frel, rel, rel_len) == 0) {
                const char *after = frel + rel_len;
                if (*after == '\0' || *after == '/') match = 1;
            }
        }
        if (match) {
            int slen = (int)strlen(frel);
            if (out_len + slen + 1 >= (int)out_cap) break;
            memcpy(out + out_len, frel, (size_t)slen);
            out_len += slen;
            out[out_len++] = '\n';
        }
    }
    if (out_len < (int)out_cap) out[out_len] = '\0';
    else if (out_cap > 0) out[out_cap - 1] = '\0';
    return 0;
}

/* Print file content as decoded bytes.
 * Reads token IDs from the KV tensor (via kvfs_handle_read) and
 * decodes them back to bytes using the byte-level tokenizer:
 * token = BYTE_VOCAB_BASE + byte → byte = token - BYTE_VOCAB_BASE */
static int shell_cat(wubu_kv_shell_t *shell, const char *path,
                      char *out, size_t out_cap) {
    if (!shell->kv_base) {
        snprintf(out, out_cap, "error: cat requires kv_base (no KV tensor mounted)");
        return -1;
    }

    uint32_t blk;
    size_t off, nfloats;
    int rc = wubu_kv_embedding_region(shell->kv, path, &blk, &off, &nfloats);
    if (rc != 0) {
        if (out_cap > 0) out[0] = '\0';
        return -1;
    }

    /* Read token floats from the KV tensor */
    float *tokens = (float *)malloc(nfloats * sizeof(float));
    if (!tokens) {
        if (out_cap > 0) out[0] = '\0';
        return -1;
    }

    wubu_kvfs_t *fs = wubu_kv_embedding_get_fs(shell->kv);
    rc = wubu_kvfs_read(fs, path, shell->kv_base, tokens, nfloats);
    if (rc != 0) {
        free(tokens);
        if (out_cap > 0) out[0] = '\0';
        return -1;
    }

    /* Decode bytes: each float is a token (uint16_t value cast to float)
     * Byte-level: token = 16384 - 256 + byte, so byte = (uint16_t)token - 16128 */
    int out_len = 0;
    for (size_t i = 0; i < nfloats && out_len < (int)out_cap - 1; i++) {
        uint16_t token = (uint16_t)tokens[i];
        /* BYTE_VOCAB_BASE = 16384 - 256 = 16128 */
        if (token >= 16128 && token < 16384) {
            char byte = (char)(token - 16128);
            out[out_len++] = byte;
        }
    }
    out[out_len] = '\0';
    free(tokens);
    return 0;
}

/* Print KV metadata for a file */
static int shell_stat(wubu_kv_shell_t *shell, const char *path,
                       char *out, size_t out_cap) {
    uint32_t blk;
    size_t off, nfloats;
    int rc = wubu_kv_embedding_region(shell->kv, path, &blk, &off, &nfloats);
    if (rc != 0) {
        if (out_cap > 0) out[0] = '\0';
        return -1;
    }
    size_t n_tokens = 0;
    /* Find the file to get its token count */
    size_t n_files = wubu_kv_embedding_file_count(shell->kv);
    for (size_t i = 0; i < n_files; i++) {
        const char *p = wubu_kv_embedding_get_path(shell->kv, i);
        if (p && strcmp(p, path) == 0) {
            n_tokens = wubu_kv_embedding_get_n_tokens(shell->kv, i);
            break;
        }
    }
    snprintf(out, out_cap,
             "path: %s\n"
             "start_block: %u\n"
             "float_offset: %zu\n"
             "n_floats: %zu\n"
             "n_tokens: %zu\n",
             path, blk, off, nfloats, n_tokens);
    return 0;
}

int wubu_kv_shell_exec(wubu_kv_shell_t *shell,
                        const char *command, const char *path,
                        char *out, size_t out_cap) {
    if (!shell || !command || !path || !out) return -1;
    if (out_cap == 0) return -1;
    out[0] = '\0';

    if (strcmp(command, "ls") == 0)
        return shell_ls(shell, path, out, out_cap);
    else if (strcmp(command, "cat") == 0)
        return shell_cat(shell, path, out, out_cap);
    else if (strcmp(command, "stat") == 0)
        return shell_stat(shell, path, out, out_cap);
    else {
        snprintf(out, out_cap, "unknown command: %s\n", command);
        return -1;
    }
}
