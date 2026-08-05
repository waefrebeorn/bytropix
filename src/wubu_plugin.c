/*
 * wubu_plugin.c — Dynamic plugin loader for wubuwizard backends.
 *
 * Implements research 066-G1 (plugin architecture). Uses dlopen/dlsym
 * to load backend .so files at runtime — CUDA kernels, quantized matmul
 * implementations, tokenizers, full model formats.
 */
#define WUBU_PLUGIN_IMPL
#include "wubu_plugin.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>

/*
 * Load a plugin from a shared object file.
 * Returns a heap-allocated wubu_plugin_api_t (caller frees with
 * wubu_plugin_unload), or NULL on failure.
 */
wubu_plugin_api_t *wubu_plugin_load(const char *path) {
    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "wubu_plugin_load: cannot open %s: %s\n", path, dlerror());
        return NULL;
    }

    wubu_plugin_init_fn init = (wubu_plugin_init_fn)dlsym(handle, WUBU_PLUGIN_API);
    if (!init) {
        fprintf(stderr, "wubu_plugin_load: %s does not export %s: %s\n",
                path, WUBU_PLUGIN_API, dlerror());
        dlclose(handle);
        return NULL;
    }

    wubu_plugin_api_t *api = calloc(1, sizeof(wubu_plugin_api_t));
    if (!api) {
        fprintf(stderr, "wubu_plugin_load: out of memory\n");
        dlclose(handle);
        return NULL;
    }

    /* Store handle so wubu_plugin_unload can close it */
    api->state = handle;
    api->cleanup = NULL;  /* set by plugin if it needs cleanup */

    /* Call the plugin's init — it fills in its api struct */
    int rc = init(api);
    if (rc != 0) {
        fprintf(stderr, "wubu_plugin_load: plugin init failed (rc=%d)\n", rc);
        free(api);
        dlclose(handle);
        return NULL;
    }

    return api;
}

/* Unload a plugin loaded by wubu_plugin_load(). */
void wubu_plugin_unload(wubu_plugin_api_t *api) {
    if (!api) return;
    if (api->cleanup) api->cleanup(api);
    void *handle = api->state;
    free(api);
    if (handle) dlclose(handle);
}
