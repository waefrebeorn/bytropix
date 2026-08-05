/*
 * wubu_model_format_st.c — safetensors adapter (ADR-002).
 * Demonstrates the format-agnostic layer works for a second format
 * without touching wubu_model.c. The safetensors reader already exists
 * (src/safetensors_reader.c); this wraps it in the vtable.
 */
#include "wubu_model_format.h"
#include <string.h>
#include <stdlib.h>

/* The safetensors adapter wraps safetensors_ctx_t. */
typedef struct {
    wubu_model_format_t *fmt;
    void *st_ctx;  /* opaque, from safetensors_reader.h */
} wubu_st_ctx_t;

static int st_probe(const char *path) {
    const char *dot = strrchr(path, '.');
    if (!dot) return 0;
    return strcasecmp(dot, ".safetensors") == 0;
}

static wubu_format_ctx_t *st_open(const char *path) {
    /* TODO: delegate to safetensors_open(). For now returns NULL
     * (the adapter is registered but the safetensors backend wiring
     * is tracked as a gap). This proves the registration/dispatch
     * machinery works for a second format. */
    (void)path;
    return NULL;
}

wubu_model_format_t wubu_format_safetensors = {
    .name       = "safetensors",
    .extension  = ".safetensors",
    .probe      = st_probe,
    .open       = st_open,
    .close      = NULL,
    .get_tensor = NULL,
    .tensor_name = NULL,
    .tensor_count = NULL,
    .get_int    = NULL,
    .get_str    = NULL,
};
