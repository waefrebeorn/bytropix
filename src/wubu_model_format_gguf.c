/*
 * wubu_model_format_gguf.c — GGUF adapter for the format-agnostic
 * model loader (ADR-002). Wraps gguf_reader.h into the
 * wubu_model_format_t vtable.
 *
 * Strangler Fig: wubu_model_init() still calls gguf_open() directly
 * (backward compatible); new code uses wubu_model_open() which dispatches
 * through the vtable. This adapter proves the pattern for a second format.
 */
#include "wubu_model_format.h"
#include "gguf_reader.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* The GGUF adapter wraps gguf_ctx; its wubu_format_ctx_t carries both
 * the vtable pointer and the wrapped context. The vtable pointer is
 * stored first so wubu_model_close() can find the right close fn. */
typedef struct {
    wubu_model_format_t *fmt;
    gguf_ctx *gguf;
} wubu_gguf_ctx_t;

static int fmt_gguf_probe(const char *path) {
    const char *dot = strrchr(path, '.');
    if (!dot) return 0;
    return strcasecmp(dot, ".gguf") == 0;
}

static wubu_format_ctx_t *fmt_gguf_open(const char *path) {
    gguf_ctx *g = gguf_open(path);
    if (!g) return NULL;
    wubu_gguf_ctx_t *ctx = calloc(1, sizeof(wubu_gguf_ctx_t));
    if (!ctx) { gguf_close(g); return NULL; }
    ctx->fmt = &wubu_format_gguf;
    ctx->gguf = g;
    return (wubu_format_ctx_t *)ctx;
}

static void fmt_gguf_close(wubu_format_ctx_t *ctx) {
    wubu_gguf_ctx_t *g = (wubu_gguf_ctx_t *)ctx;
    if (!g) return;
    if (g->gguf) gguf_close(g->gguf);
    free(g);
}

static int fmt_gguf_get_int(wubu_format_ctx_t *ctx, const char *key, int64_t *val) {
    wubu_gguf_ctx_t *g = (wubu_gguf_ctx_t *)ctx;
    if (!g || !g->gguf) return -1;
    return gguf_meta_get_int(g->gguf, key, val);
}

static int fmt_gguf_get_str(wubu_format_ctx_t *ctx, const char *key, const char **val) {
    wubu_gguf_ctx_t *g = (wubu_gguf_ctx_t *)ctx;
    if (!g || !g->gguf) return -1;
    return gguf_meta_get_str(g->gguf, key, val);
}

static const char *fmt_gguf_tensor_name(wubu_format_ctx_t *ctx, int idx) {
    wubu_gguf_ctx_t *g = (wubu_gguf_ctx_t *)ctx;
    if (!g || !g->gguf || idx < 0 || idx >= (int)g->gguf->n_tensors) return NULL;
    return g->gguf->tensors[idx].name;
}

static int fmt_gguf_tensor_count(wubu_format_ctx_t *ctx) {
    wubu_gguf_ctx_t *g = (wubu_gguf_ctx_t *)ctx;
    if (!g || !g->gguf) return 0;
    return (int)g->gguf->n_tensors;
}

wubu_model_format_t wubu_format_gguf = {
    .name        = "gguf",
    .extension   = ".gguf",
    .probe       = fmt_gguf_probe,
    .open        = fmt_gguf_open,
    .close       = fmt_gguf_close,
    .get_tensor  = NULL,  /* TODO: wire through gguf_tensor lookup */
    .tensor_name = fmt_gguf_tensor_name,
    .tensor_count = fmt_gguf_tensor_count,
    .get_int     = fmt_gguf_get_int,
    .get_str     = fmt_gguf_get_str,
};
