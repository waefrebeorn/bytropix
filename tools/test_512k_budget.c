/* Test: 512K-context OOM-safe memory budget (doc: 512K cycles, no OOM).
 * Verifies the budget calculator keeps the KV cache within available RAM at
 * 512K context and that layer-streaming engages when KV would exceed RAM. */
#include "wubu_mem_budget.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

int main(void) {
    /* Simulate a 24-layer GQA model, kv_dim=128, F16 (2 bytes/element). */
    int n_gqa = 24;
    int kv_dim = 128;
    int bpe = 2;            /* F16 */
    int req_ctx = 512 * 1024;   /* 512K */
    size_t model_w = 4ULL * 1024 * 1024 * 1024;  /* 4GB weights (mmap) */

    /* --- RAM-TIGHT (3 GB): 512K KV = 6 GB > RAM -> must stream + shrink --- */
    size_t ram_tight = 3ULL * 1024 * 1024 * 1024;
    wubu_mem_budget_t *bt = wubu_mem_budget_create(ram_tight, model_w, n_gqa, 0, NULL, kv_dim, bpe);
    assert(bt);
    wubu_mem_budget_info_t it = wubu_mem_budget_compute(bt, req_ctx, 64ULL<<20, 512ULL<<20, 0, 0);
    printf("[512K tight] max_ctx=%d kv_bytes=%zu ram=%zuMB stream=%d\n",
           it.max_kv_ctx, it.kv_cache_bytes, ram_tight/(1024*1024), it.use_layer_stream);

    assert(it.kv_cache_bytes <= ram_tight);     /* never exceeds RAM */
    assert(it.use_layer_stream == 1);           /* 6GB KV > 3GB RAM -> stream */
    assert(it.max_kv_ctx < req_ctx);            /* RAM-bound shrink */

    /* --- Generous RAM (12 GB, like this host): 512K KV (6 GB) fits -> no stream */
    size_t ram_ok = 12ULL * 1024 * 1024 * 1024;
    wubu_mem_budget_t *bo = wubu_mem_budget_create(ram_ok, model_w, n_gqa, 0, NULL, kv_dim, bpe);
    assert(bo);
    wubu_mem_budget_info_t io = wubu_mem_budget_compute(bo, req_ctx, 64ULL<<20, 512ULL<<20, 0, 0);
    printf("[512K ok]   max_ctx=%d kv_bytes=%zu ram=%zuMB stream=%d\n",
           io.max_kv_ctx, io.kv_cache_bytes, ram_ok/(1024*1024), io.use_layer_stream);
    assert(io.kv_cache_bytes <= ram_ok);
    assert(io.max_kv_ctx == req_ctx);           /* fits, full context */
    assert(io.use_layer_stream == 0);           /* no streaming needed */

    /* --- Huge RAM (1 TB): full 512K context, no stream --- */
    wubu_mem_budget_t *bh = wubu_mem_budget_create(1024ULL<<30, 100ULL<<20, n_gqa, 0, NULL, kv_dim, bpe);
    wubu_mem_budget_info_t ih = wubu_mem_budget_compute(bh, req_ctx, 64ULL<<20, 512ULL<<20, 0, 0);
    printf("[512K huge]  max_ctx=%d stream=%d\n", ih.max_kv_ctx, ih.use_layer_stream);
    assert(ih.max_kv_ctx == req_ctx);
    assert(ih.use_layer_stream == 0);

    wubu_mem_budget_destroy(bt);
    wubu_mem_budget_destroy(bo);
    wubu_mem_budget_destroy(bh);
    printf("ALL 512K-CONTEXT OOM-SAFETY TESTS PASSED\n");
    return 0;
}
