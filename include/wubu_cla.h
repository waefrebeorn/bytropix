#ifndef WUBU_CLA_H
#define WUBU_CLA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_cla wubu_cla_t;

/* Plan cross-layer KV sharing. type[i]: 0=sliding, 1=global (type-matched). */
wubu_cla_t *wubu_cla_plan(int n_layers, int share_k, const int *type);
void wubu_cla_free(wubu_cla_t *c);
int  wubu_cla_kv_owner(const wubu_cla_t *c, int layer);
double wubu_cla_unique_kv_frac(const wubu_cla_t *c);
double wubu_cla_kv_reduction(const wubu_cla_t *c, double kv_bytes);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_CLA_H */
