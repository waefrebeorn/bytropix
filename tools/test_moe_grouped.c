/* Test: wubu_moe_grouped (Area D — MoE expert grouping). */
#include "wubu_moe_grouped.h"
#include <stdio.h>
#include <assert.h>

int main(void) {
    int n_experts = 8;
    wubu_moe_router_t *r = wubu_moe_router_create(n_experts);
    assert(r != NULL);
    /* 12 tokens routed round-robin across 8 experts. */
    int routes[12];
    for (int t = 0; t < 12; t++) routes[t] = t % n_experts;
    wubu_moe_router_assign(r, routes, 12);

    int total = 0;
    for (int e = 0; e < n_experts; e++) total += wubu_moe_router_count(r, e);
    printf("grouped %d tokens across %d experts (expect 12)\n", total, n_experts);
    assert(total == 12);
    /* expert 0 got tokens 0 and 8 */
    assert(wubu_moe_router_count(r, 0) == 2);
    printf("expert0 token idx[0]=%d idx[1]=%d (expect 0,8)\n",
           wubu_moe_router_idx(r,0)[0], wubu_moe_router_idx(r,0)[1]);
    assert(wubu_moe_router_idx(r,0)[0] == 0 && wubu_moe_router_idx(r,0)[1] == 8);

    int hot[3];
    wubu_moe_router_top_hot(r, 3, hot);
    printf("top-hot experts: %d %d %d\n", hot[0], hot[1], hot[2]);
    assert(hot[0] >= 0);

    wubu_moe_router_free(r);
    printf("ALL MOE-GROUPED TESTS PASSED\n");
    return 0;
}
