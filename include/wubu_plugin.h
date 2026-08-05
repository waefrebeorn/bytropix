/*
 * wubu_plugin.h — Dynamic plugin interface for wubuwizard backends.
 *
 * Implements research 066-G1 (plugin architecture): the engine loads
 * backend kernels (CUDA kernels, quantized matmul implementations,
 * tokenizers, full model formats) via dlopen/dlsym. This keeps the
 * core engine free of hard dependencies on optional libraries.
 *
 * Plugin contract:
 *   A .so file exports one symbol: the plugin's entry point, whose
 *   name is the value of the "WUBU_PLUGIN_API" macro (see below).
 *   The entry point returns int and takes a wubu_plugin_api_t*, which
 *   the plugin fills with its function pointers + metadata.
 *
 * ADR-002: plugins are the dynamic form of the backend vtable pattern.
 * Where wubu_model_format_t is statically linked, wubu_plugin_t is
 * loaded at runtime via dlopen.
 */
#ifndef WUBU_PLUGIN_H
#define WUBU_PLUGIN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The plugin must export this symbol. */
#define WUBU_PLUGIN_API  "wubu_plugin_init"

/* Plugin types — each maps to a backend domain. */
typedef enum {
    WUBU_PLUGIN_UNKNOWN     = 0,
    WUBU_PLUGIN_KERNEL      = 1,  /* compute kernels (matmul, attention) */
    WUBU_PLUGIN_FORMAT      = 2,  /* weight format adapter */
    WUBU_PLUGIN_TOKENIZER   = 3,  /* tokenizer implementation */
    WUBU_PLUGIN_SCHEDULER   = 4,  /* KV cache scheduling policy */
} wubu_plugin_type_t;

/*
 * Plugin API struct — filled in by the plugin's wubu_plugin_init().
 * Each domain uses a different subset of these fields.
 */
typedef struct {
    const char *name;        /* human-readable plugin name */
    const char *version;     /* plugin version string */
    wubu_plugin_type_t type; /* domain this plugin serves */

    /* Format adapter vtable (for WUBU_PLUGIN_FORMAT):
     * If the plugin provides a format, it can fill in this vtable.
     * The engine will also call wubu_model_format_register() to
     * register it with the core adapter registry. */
    void *format_vtable;     /* optional: wubu_model_format_t* */

    /* Kernel backend (for WUBU_PLUGIN_KERNEL):
     * Opaque handle to the plugin's kernel dispatch table. */
    void *kernel_dispatch;   /* optional: plugin-owned kernel table */

    /* Opaque plugin state — freed by calling plugin_cleanup */
    void *state;             /* plugin-owned context */

    /* Called by the engine when unloading: free state, release resources. */
    void (*cleanup)(struct wubu_plugin_api *api);
} wubu_plugin_api_t;

/*
 * Plugin entry point signature.
 * Returns 0 on success, nonzero on failure.
 * The plugin fills in `api` and the engine takes ownership.
 */
typedef int (*wubu_plugin_init_fn)(wubu_plugin_api_t *api);

#ifdef __cplusplus
}
#endif

/* ---- Engine-side loader (implemented in wubu_plugin.c) ---- */

#ifdef WUBU_PLUGIN_IMPL
#include <dlfcn.h>
#endif

/*
 * Load a plugin from a shared object file.
 * The .so must export WUBU_PLUGIN_API (wubu_plugin_init).
 * Returns the filled api on success, NULL on failure.
 *
 * Note: this is a proof-of-concept stub. The full implementation
 * (with reference-counting, unload hooks, and error reporting)
 * is tracked as a gap in the ADR.
 */
wubu_plugin_api_t *wubu_plugin_load(const char *path);

/* Unload a plugin loaded by wubu_plugin_load(). */
void wubu_plugin_unload(wubu_plugin_api_t *api);

#endif /* WUBU_PLUGIN_H */
