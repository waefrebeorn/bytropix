#ifndef WUBU_PD_SPLIT_H
#define WUBU_PD_SPLIT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_pd_split wubu_pd_split_t;

wubu_pd_split_t *wubu_pd_split_create(int np, int nd, double rdma_tb_s);
void wubu_pd_split_free(wubu_pd_split_t *s);

double wubu_pd_kv_transfer_ms(const wubu_pd_split_t *s, int s_tokens,
                             int layers, int kv_heads, int head_dim, int kv_bits);
int wubu_pd_route_decode(const wubu_pd_split_t *s, const int *queues);
int wubu_pd_transfer_mode(const wubu_pd_split_t *s, double prefill_load,
                          double decode_load);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_PD_SPLIT_H */
