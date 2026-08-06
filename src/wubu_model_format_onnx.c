/*
 * wubu_model_format_onnx.c — ONNX format adapter (stub).
 *
 * ONNX models are not yet implemented in wubuwizard. This file provides
 * a probe-only stub so the format vtable resolves at link time for
 * CPU builds (the real implementation will provide a full wubu_model_format_t
 * with open/get_tensor/close). Until then, .onnx files probe as "unclaimed"
 * and fall through to whatever adapter matches (typically GGUF).
 *
 * ADR-002: format adapters are standalone TUs — the engine links only
 * the ones it needs. This stub exists so the registration symbol is not
 * a dangling extern in wubu_model_format.c.
 */
#include "wubu_model_format.h"

static int onnx_probe(const char *path) {
    (void)path;
    return 0;  /* probe-only: never claims a file */
}

/* The engine calls wubu_model_format_register_onnx() at startup; this stub
 * is intentionally NOT registered (probe returns 0 = no match) so ONNX
 * files are left to the real adapter when it is built. */
wubu_model_format_t wubu_format_onnx_stub = {
    .name      = "onnx-stub",
    .extension = ".onnx",
    .probe     = onnx_probe,
    .open      = NULL,
    .close     = NULL,
    .get_tensor = NULL,
    .tensor_name = NULL,
    .tensor_count = NULL,
    .get_int   = NULL,
    .get_str   = NULL,
    .meta_key  = NULL,
    .meta_count = NULL,
};
