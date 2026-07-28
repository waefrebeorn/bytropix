#ifndef WUBU_LATENTMOE_H
#define WUBU_LATENTMOE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_latentmoe wubu_latentmoe_t;

/* Stable LatentMoE: n_routed experts, top_k active, optional always-on shared. */
wubu_latentmoe_t *wubu_latentmoe_create(int n_routed, int top_k, int shared);
void wubu_latentmoe_free(wubu_latentmoe_t *m);
void wubu_latentmoe_route(const wubu_latentmoe_t *m, const float *scores, int *idx);
int  wubu_latentmoe_active_count(const wubu_latentmoe_t *m);
float wubu_latentmoe_entropy(const wubu_latentmoe_t *m, const float *scores);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_LATENTMOE_H */
