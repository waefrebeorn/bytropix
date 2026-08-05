/*
 * wubu_model_format.h — Format-agnostic model loading interface (ADR-002).
 *
 * The engine must not be coupled to a single weight format. Each format
 * (GGUF, safetensors, ONNX) provides an adapter implementing this vtable;
 * wubu_model_init() dispatches on the file extension to pick the right
 * adapter. New formats are added by writing one adapter + registering it
 * via wubu_model_format_register().
 *
 * Convergence from research 066-B1: "Abstract formats behind a vtable so
 * the engine never knows or cares which serializer produced a checkpoint."
 */
#ifndef WUBU_MODEL_FORMAT_H
#define WUBU_MODEL_FORMAT_H

#include "wubu_model_fwd.h"
#include "gguf_reader.h"   /* for gguf_ctx, used by the GGUF adapter */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Per-format adapter vtable ---- */

/* Opaque handle — each adapter owns its own context state. */
typedef struct wubu_format_ctx wubu_format_ctx_t;

/* The format adapter vtable. */
typedef struct wubu_model_format wubu_model_format_t;

struct wubu_model_format {
    const char *name;          /* "gguf", "safetensors", "onnx" */
    const char *extension;     /* ".gguf", ".safetensors", ".onnx" */
    int (*probe)(const char *path); /* returns 1 if this adapter owns the file */
    wubu_format_ctx_t *(*open)(const char *path); /* NULL on failure */
    void (*close)(wubu_format_ctx_t *ctx);

    /* Tensor access: name → data pointer + shape. Caller does NOT own data. */
    int (*get_tensor)(wubu_format_ctx_t *ctx, const char *name,
                      const void **data, int *n_dims, const int64_t **shape);
    const char *(*tensor_name)(wubu_format_ctx_t *ctx, int idx);
    int (*tensor_count)(wubu_format_ctx_t *ctx);

    /* Metadata (scalar) access. */
    int (*get_int)(wubu_format_ctx_t *ctx, const char *key, int64_t *val);
    int (*get_str)(wubu_format_ctx_t *ctx, const char *key,
                   const char **val);
    const char *(*meta_key)(wubu_format_ctx_t *ctx, int idx);
    int (*meta_count)(wubu_format_ctx_t *ctx);
};

/* ---- Registration / dispatch ---- */

/* Register a format adapter. Called at program init by each adapter's
 * _register() function. Returns 0 on success, -1 if already registered. */
int wubu_model_format_register(const wubu_model_format_t *fmt);

/* Dispatch: pick the adapter whose probe() returns true for path.
 * Returns NULL if no adapter matches. */
const wubu_model_format_t *wubu_model_format_for(const char *path);

/* Convenience: open a file via the right adapter. */
wubu_format_ctx_t *wubu_model_open(const char *path);
void wubu_model_close(wubu_format_ctx_t *ctx);

/* ---- Built-in adapters ---- */
/* Each lives in its own file; the engine only links the ones it needs. */

void wubu_model_format_register_gguf(void);   /* wubu_model_format_gguf.c */
void wubu_model_format_register_safetensors(void); /* wubu_model_format_st.c */
void wubu_model_format_register_onnx(void);   /* wubu_model_format_onnx.c (stub) */

/* Call once at startup to register all built-in adapters. */
void wubu_model_format_register_all(void);

/* ---- Adapter-specific: gguf_ctx is the GGUF adapter's internal type ---- */
/* The GGUF adapter wraps gguf_ctx; other adapters wrap their own types.
 * wubu_model_init() currently calls gguf_open() directly — it should
 * call wubu_model_open() instead, then use the vtable. */

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MODEL_FORMAT_H */
