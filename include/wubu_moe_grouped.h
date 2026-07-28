#ifndef WUBU_MOE_GROUPED_H
#define WUBU_MOE_GROUPED_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_moe_router wubu_moe_router_t;

wubu_moe_router_t *wubu_moe_router_create(int n_experts);
void wubu_moe_router_free(wubu_moe_router_t *r);
void wubu_moe_router_assign(wubu_moe_router_t *r, const int *routes, int n_tokens);
void wubu_moe_router_top_hot(wubu_moe_router_t *r, int k, int *out_experts);

/* Per-expert token list accessor (for grouped-GEMM caller). */
int wubu_moe_router_count(wubu_moe_router_t *r, int e);
int *wubu_moe_router_idx(wubu_moe_router_t *r, int e);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MOE_GROUPED_H */
