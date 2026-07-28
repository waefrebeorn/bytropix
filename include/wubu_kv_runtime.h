/*
 * wubu_kv_runtime.h -- runtime KV-cache scheme selection + dispatch support.
 * The KV cache functions in wubu_model.h dispatch on the global g_kv_scheme
 * (set here) so the engine picks precision per-model at load time.
 */
#ifndef WUBU_KV_RUNTIME_H
#define WUBU_KV_RUNTIME_H

#include "wubu_kv_select.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Global selected scheme (set at model load, read by kv_cache_* in wubu_model.h). */
extern int g_kv_scheme;
void wubu_kv_set_scheme(int scheme);
int  wubu_kv_get_scheme(void);

/* Pick + apply the KV scheme from real model params + detected bandwidth.
 * P_params absolute param count (e.g. 27e9); s = target context.
 * Returns the chosen scheme. */
int wubu_kv_autoselect(double P_params, int n_layers, int n_kv_heads,
                        int head_dim, double beta_eff_tb_s, int s);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_KV_RUNTIME_H */
