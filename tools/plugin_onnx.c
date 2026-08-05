/*
 * plugin_onnx.c — Example dlopen plugin: ONNX format adapter.
 *
 * Demonstrates the plugin architecture (research 066-G1) + the format
 * adapter vtable (research 066-B1 / ADR-002). This is a STUB plugin —
 * the actual ONNX parsing is tracked as a gap in the ADR. The plugin
 * compiles as a standalone .so and is loaded at runtime via
 * wubu_plugin_load("plugins/libonnx_format.so").
 *
 * Build:  make plugin_onnx
 *   which runs: gcc -shared -fPIC -DWUBU_PLUGIN_API=\"wubu_plugin_init\" \
 *       -I include -o plugins/libonnx_format.so tools/plugin_onnx.c
 */
#define _POSIX_C_SOURCE 200809L
#include "wubu_plugin.h"
#include "wubu_model_format.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ONNX adapter implements the wubu_model_format_t vtable.
 * For now, probe() returns 1 for .onnx files, open() returns a
 * placeholder context. The real implementation will wire up onnxruntime
 * C API (tracked as gap G1-ONNX-RUNTIME in ADR-002). */

typedef struct {
    const char *path;  /* path passed to open() */
} wubu_onnx_ctx_t;

static int onnx_probe(const char *path) {
    const char *dot = strrchr(path, '.');
    if (!dot) return 0;
    return strcasecmp(dot, ".onnx") == 0;
}

static wubu_format_ctx_t *onnx_open(const char *path) {
    wubu_onnx_ctx_t *ctx = calloc(1, sizeof(wubu_onnx_ctx_t));
    if (!ctx) return NULL;
    ctx->path = path;
    fprintf(stderr, "[onnx plugin] would load ONNX: %s (STUB)\n", path);
    return (wubu_format_ctx_t *)ctx;
}

static void onnx_close(wubu_format_ctx_t *ctx) {
    free(ctx);
}

/* The vtable — one adapter definition */
static wubu_model_format_t wubu_format_onnx = {
    .name        = "onnx",
    .extension   = ".onnx",
    .probe       = onnx_probe,
    .open        = onnx_open,
    .close       = onnx_close,
    .get_tensor  = NULL,
    .tensor_name = NULL,
    .tensor_count = NULL,
    .get_int     = NULL,
    .get_str     = NULL,
};

/* Plugin entry point — fills the wubu_plugin_api_t struct */
int wubu_plugin_init(wubu_plugin_api_t *api) {
    if (!api) return -1;
    api->name        = "onnx-format";
    api->version     = "0.1.0";
    api->type        = WUBU_PLUGIN_FORMAT;
    api->format_vtable = &wubu_format_onnx;
    api->cleanup      = NULL;
    return 0;
}
