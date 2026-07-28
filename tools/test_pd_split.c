/* Test: wubu_pd_split (Round-2 #131 — PD disaggregation planner). */
#include "wubu_pd_split.h"
#include <stdio.h>
#include <assert.h>

int main(void) {
    /* 2 prefill GPUs, 4 decode GPUs, InfiniBand-class RDMA (0.05 TB/s). */
    wubu_pd_split_t *s = wubu_pd_split_create(2, 4, 0.05);
    assert(s != NULL);
    /* KV transfer for 4096-token prefix, 80 layers, 8 GQA heads, d128, FP16. */
    double ms = wubu_pd_kv_transfer_ms(s, 4096, 80, 8, 128, 16);
    printf("KV transfer(4k,80L,8kv,d128,FP16) = %.1f ms (InfiniBand)\n", ms);
    assert(ms > 0 && ms < 5000);

    /* NVLink-class (0.9 TB/s) should be ~18x faster. */
    wubu_pd_split_t *nv = wubu_pd_split_create(2, 4, 0.9);
    double ms_nv = wubu_pd_kv_transfer_ms(nv, 4096, 80, 8, 128, 16);
    printf("KV transfer NVLink = %.1f ms (expect ~18x faster)\n", ms_nv);
    assert(ms_nv * 15 < ms);

    /* Routing: pick least-loaded decode GPU. */
    int q[4] = {10, 3, 25, 7};
    int r = wubu_pd_route_decode(s, q);
    printf("routed to decode gpu=%d (expect 1, queue=3)\n", r);
    assert(r == 1);

    /* Transfer mode: decode-heavy -> read-mode (1). */
    assert(wubu_pd_transfer_mode(s, 0.3, 0.9) == 1);
    assert(wubu_pd_transfer_mode(s, 0.9, 0.3) == 0);

    wubu_pd_split_free(s); wubu_pd_split_free(nv);
    printf("ALL PD-SPLIT TESTS PASSED\n");
    return 0;
}
