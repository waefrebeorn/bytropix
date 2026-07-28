#ifndef WUBU_CUDA_GRAPH_H
#define WUBU_CUDA_GRAPH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_CG_MAX_NODES 64

typedef struct wubu_cuda_graph wubu_cuda_graph_t;

wubu_cuda_graph_t *wubu_cuda_graph_create(int max_seq);
void wubu_cuda_graph_free(wubu_cuda_graph_t *g);

int wubu_cuda_graph_plan(wubu_cuda_graph_t *g, int seq);
int wubu_cuda_graph_update_kv(wubu_cuda_graph_t *g, int step, void *kv_dev_ptr);
int wubu_cuda_graph_replay(wubu_cuda_graph_t *g);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_CUDA_GRAPH_H */
