/* wubu_model_scalable.h — fractal self-scaling foundation model (AN23)
 *
 * The model is a self-similar tree of layers. Layer N has 3× the
 * parameters of layer N-1. On a weak machine, only the trunk (layer 0)
 * loads. On a powerful machine, all layers load.
 *
 * Parameters are stored in importance order (most critical first).
 * The loader mmap's from the front of the file and stops when the
 * memory budget is exhausted. The coherence diagnostic + grow/shrink
 * operators then manage which weight regions stay active.
 *
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_MODEL_SCALABLE_H
#define WUBU_MODEL_SCALABLE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Precision tier for weight storage (same cascade as KV tiering) */
typedef enum {
    WUBU_WT_F32  = 0,  /* 4 bytes/element */
    WUBU_WT_F16  = 1,  /* 2 bytes/element */
    WUBU_WT_Q8_K = 2,  /* ~0.5 bytes/element */
    WUBU_WT_Q4_K = 3,  /* ~0.25 bytes/element */
    WUBU_WT_Q2_K = 4,  /* ~0.125 bytes/element (esoteric tails) */
} wubu_weight_format_t;

/* A weight region: a contiguous span of the parameter file.
 * The file is laid out in importance order (most critical at offset 0).
 * Each region has a name (e.g. "layer.0.attn.qkv"), a format (precision
 * tier), and a byte span [offset, offset+n_bytes). */
typedef struct {
    char                  name[128];
    uint64_t              offset;     /* byte offset in the parameter file */
    uint64_t              n_bytes;     /* byte size at file precision */
    wubu_weight_format_t  fmt;         /* storage precision in the file */
    int                   layer;       /* which fractal layer (0 = trunk) */
    int                   priority;     /* importance rank (0 = highest) */
    int                   active;      /* 1 = currently loaded into RAM */
} wubu_weight_region_t;

/* The fractal model configuration. */
typedef struct {
    int     trunk_params;    /* layer 0 param count (e.g. 12M) */
    int     branch_factor;   /* layer N = branch_factor × layer N-1 (3) */
    int     max_layers;      /* max fractal depth (e.g. 6 = 12M→324M) */
    size_t  ram_budget;      /* total bytes available for loaded weights */
    size_t  kv_cache_bytes;  /* KV cache memory budget (subtracted from ram) */
    int     min_precision;   /* deepest weight format allowed (Q4_K=3) */
} wubu_scalable_cfg_t;

/* Default config: 12M trunk, 3× branch, 6 layers, 256MB budget. */
wubu_scalable_cfg_t wubu_scalable_default_cfg(void);

/* The fractal model state. */
typedef struct wubu_scalable_model wubu_scalable_model_t;

/* Create the fractal model. The parameter file (param_path) is mmap'd
 * lazily — only regions that fit in the budget are paged in.
 *
 * cfg->ram_budget is the total RAM budget. The loader computes how
 * many layers/depth of the fractal tree fit, then mmap's that region.
 *
 * Returns NULL on failure. */
wubu_scalable_model_t *wubu_scalable_model_create(const char *param_path,
                                                    const wubu_scalable_cfg_t *cfg);

/* Returns the number of weight regions registered. */
size_t wubu_scalable_region_count(wubu_scalable_model_t *m);

/* Returns the region descriptor at index i (read-only). */
const wubu_weight_region_t *
wubu_scalable_get_region(wubu_scalable_model_t *m, size_t i);

/* Determine how many fractal layers fit in the budget.
 * Returns the depth (0 = trunk only, 1 = trunk+first branch, etc.). */
int wubu_scalable_budget_depth(wubu_scalable_model_t *m);

/* Get the actual bytes of a weight region as F32 (lazy dequantize).
 * Only works for regions that are active (within the budget).
 * Returns 0 on success, -1 if region not loaded.
 * *out_n_elems gets the number of float32 elements. */
int wubu_scalable_get_f32(wubu_scalable_model_t *m,
                           const char *name,
                           float **out_data, size_t *out_n_elems);

/* Mark a region as "hot" (recently attended to). The grow operator
 * uses this to decide which regions to promote. Returns 0 on success. */
int wubu_scalable_mark_hot(wubu_scalable_model_t *m, const char *name);

/* Prune cold weight regions — returns their memory to the pool.
 * Returns the number of regions pruned. */
int wubu_scalable_prune_cold(wubu_scalable_model_t *m);

/* Report current memory usage. */
void wubu_scalable_memory_stats(wubu_scalable_model_t *m,
                                 size_t *out_active_bytes,
                                 size_t *out_total_bytes,
                                 int *out_active_regions,
                                 int *out_total_regions);

/* Free the model. Does NOT close the mmap (that's the OS's job). */
void wubu_scalable_model_free(wubu_scalable_model_t *m);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_MODEL_SCALABLE_H */
