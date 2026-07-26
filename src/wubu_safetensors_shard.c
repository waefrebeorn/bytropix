/*
 * wubu_safetensors_shard.c -- multi-shard safetensors loader.
 * See wubu_safetensors_shard.h. Self-contained; uses safetensors_reader.
 */
#include "wubu_safetensors_shard.h"
#include "safetensors_reader.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define SHARD_MAX 1024

struct wubu_shard_ctx {
    st_ctx *shards[SHARD_MAX];
    int     n;
};

/* Scan a directory for model-NNNNN-of-NNNNN.safetensors files. */
static int scan_shards(const char *dir, char paths[SHARD_MAX][1024]) {
    /* Use a glob via opendir would be cleaner, but to stay dependency-free
     * we shell out to a tiny helper: list files, keep the shard pattern. */
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             "ls %s/model-*-of-*.safetensors 2>/dev/null | sort", dir);
    FILE *p = popen(cmd, "r");
    int n = 0;
    if (p) {
        char line[2048];
        while (fgets(line, sizeof(line), p) && n < SHARD_MAX) {
            size_t L = strlen(line);
            while (L && (line[L-1]=='\n'||line[L-1]=='\r')) line[--L]=0;
            if (L) { snprintf(paths[n], 1024, "%s", line); n++; }
        }
        pclose(p);
    }
    return n;
}

/* Derive the directory + base name from a single shard path. */
static void dir_of(const char *path, char *dir, size_t dsize) {
    const char *sl = strrchr(path, '/');
    if (sl) { size_t n = (size_t)(sl - path); if (n >= dsize) n = dsize-1;
              memcpy(dir, path, n); dir[n]=0; }
    else { dir[0]=0; }
}

wubu_shard_ctx_t *wubu_shard_open(const char *path_or_dir) {
    char dir[1024]; dir_of(path_or_dir, dir, sizeof(dir));
    if (dir[0]==0) snprintf(dir, sizeof(dir), ".");

    char paths[SHARD_MAX][1024];
    int n = 0;
    /* If the given path is itself a shard, scan its directory. */
    if (strstr(path_or_dir, "model-") && strstr(path_or_dir, "-of-")) {
        n = scan_shards(dir, paths);
    } else {
        /* Treat as directory; also check for a lone file. */
        n = scan_shards(path_or_dir, paths);
    }
    if (n == 0) return NULL;

    wubu_shard_ctx_t *sc = (wubu_shard_ctx_t *)calloc(1, sizeof(*sc));
    if (!sc) return NULL;
    sc->n = 0;
    for (int i = 0; i < n && i < SHARD_MAX; i++) {
        st_ctx *s = st_open(paths[i]);
        if (s) sc->shards[sc->n++] = s;
    }
    if (sc->n == 0) { free(sc); return NULL; }
    return sc;
}

int wubu_shard_count(const wubu_shard_ctx_t *sc) { return sc ? sc->n : 0; }

int64_t wubu_shard_n_tensors(const wubu_shard_ctx_t *sc) {
    if (!sc) return 0;
    int64_t t = 0;
    for (int i = 0; i < sc->n; i++) t += st_n_tensors(sc->shards[i]);
    return t;
}

static const st_tensor_info *find_across(const wubu_shard_ctx_t *sc,
                                         const char *name, int *shard_idx) {
    for (int i = 0; i < sc->n; i++) {
        const st_tensor_info *t = st_find_tensor(sc->shards[i], name);
        if (t) { if (shard_idx) *shard_idx = i; return t; }
    }
    return NULL;
}

float *wubu_shard_load_f32(const wubu_shard_ctx_t *sc, const char *name,
                           int64_t *n_elems_out) {
    if (!sc) return NULL;
    int si = 0;
    const st_tensor_info *t = find_across(sc, name, &si);
    if (!t) return NULL;
    int64_t n = t->n_elems;
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (!buf) return NULL;
    if (st_read_tensor_f32(sc->shards[si], t, buf, n) != n) { free(buf); return NULL; }
    if (n_elems_out) *n_elems_out = n;
    return buf;
}

float *wubu_shard_load_f32_t(const wubu_shard_ctx_t *sc, const char *name,
                             int rows, int cols) {
    if (!sc) return NULL;
    int si = 0;
    const st_tensor_info *t = find_across(sc, name, &si);
    if (!t) return NULL;
    int64_t need = (int64_t)rows * cols;
    if (t->n_elems < need) return NULL;
    float *tmp = (float *)malloc((size_t)need * sizeof(float));
    if (!tmp) return NULL;
    if (st_read_tensor_f32(sc->shards[si], t, tmp, need) != need) { free(tmp); return NULL; }
    float *out = (float *)malloc((size_t)need * sizeof(float));
    if (!out) { free(tmp); return NULL; }
    /* transpose [rows,cols] -> [cols,rows] */
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            out[(size_t)c * rows + r] = tmp[(size_t)r * cols + c];
    free(tmp);
    return out;
}

const float *wubu_shard_data_f32(const wubu_shard_ctx_t *sc, const char *name,
                                 int64_t *n_elems_out) {
    if (!sc) return NULL;
    int si = 0;
    const st_tensor_info *t = find_across(sc, name, &si);
    if (!t) return NULL;
    /* st_read_tensor_f32 writes into a caller buffer; to avoid a copy we
     * re-read into a persistent (malloc'd) buffer owned by this call's
     * caller is not possible here, so we return NULL and require the
     * load_f32 variants. Kept for API completeness. */
    (void)n_elems_out;
    return NULL;
}

void wubu_shard_close(wubu_shard_ctx_t *sc) {
    if (!sc) return;
    for (int i = 0; i < sc->n; i++) st_close(sc->shards[i]);
    free(sc);
}
