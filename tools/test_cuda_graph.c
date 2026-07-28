/* Test: wubu_cuda_graph (Area E — CUDA-graph decode planning/update).
 * Exercises the CPU-side planning + partial-KV-update logic without a GPU. */
#include "wubu_cuda_graph.h"
#include <stdio.h>
#include <assert.h>

int main(void) {
    wubu_cuda_graph_t *g = wubu_cuda_graph_create(128);
    assert(g != NULL);

    /* Plan capture at seq len 128. */
    assert(wubu_cuda_graph_plan(g, 128) == 0);

    /* Simulate 10 decode steps, each updating only the KV node param. */
    int dummy_kv[10];
    for (int t = 0; t < 10; t++) {
        assert(wubu_cuda_graph_update_kv(g, t, &dummy_kv[t]) == 0);
        assert(wubu_cuda_graph_replay(g) == 0);
    }
    printf("cuda-graph: planned + 10 partial-KV updates + replays OK\n");

    /* Out-of-range guard. */
    assert(wubu_cuda_graph_update_kv(g, 999, NULL) == -1);

    wubu_cuda_graph_free(g);
    printf("ALL CUDA-GRAPH TESTS PASSED\n");
    return 0;
}
