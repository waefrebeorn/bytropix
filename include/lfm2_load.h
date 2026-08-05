#ifndef LFM2_LOAD_H
#define LFM2_LOAD_H

#include "wubu_lfm2.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Loads a LFM2.5 safetensors checkpoint dir into lfm2_model_t.
 * Self-contained loader: scans model-NNN-of-MMM shards, parses config.json,
 * binds per-layer weights (conv or GQA variants) as F32 (BF16 dequantized
 * by the safetensors reader). Returns true on success. */

/* internal: open all shards in model_dir; returns count (0 on failure) */
int lfm2_open_shards(const char *model_dir);

#ifdef __cplusplus
}
#endif

#endif /* LFM2_LOAD_H */
