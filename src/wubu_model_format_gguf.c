/*
 * wubu_model_format_gguf.c — GGUF adapter for the format-agnostic
 * model loader (ADR-002). Wraps gguf_reader.h into the
 * wubu_model_format_t vtable.
 *
 * Strangler Fig: wubu_model_init() still calls gguf_open() directly
 * (backward compatible); new code uses wubu_model_open() which dispatches
 * through the vtable.
 */
#include "wubu_model_format.h"
#include "gguf_reader.h"
#include "wubu_tokens.h"   /* green accent (shared CLI+GUI tokens) */
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* The GGUF adapter wraps gguf_ctx; its wubu_format_ctx_t carries both
 * the vtable pointer and the wrapped context. */
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

/* NOTE: get_int/get_str are left NULL — the current gguf_reader.h API
 * does not expose arbitrary metadata key/value access. When that is
 * added, wire these in. */

wubu_model_format_t wubu_format_gguf = {
    .name        = "gguf",
    .extension   = ".gguf",
    .probe       = fmt_gguf_probe,
    .open        = fmt_gguf_open,
    .close       = fmt_gguf_close,
    .get_tensor  = NULL,
    .tensor_name = fmt_gguf_tensor_name,
    .tensor_count = fmt_gguf_tensor_count,
    .get_int     = NULL,
    .get_str     = NULL,
};
