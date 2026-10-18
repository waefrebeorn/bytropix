/* test_backend_dispatch.c — the backend vtable seam (ADR-002/003).
 *
 * The whole point of wubu_backend.h is that model code never needs
 * #ifdef GPU_SUPPORT: every GPU path goes through a vtable and every
 * dispatch helper must degrade cleanly to "use the CPU path" when the
 * backend is CPU-only or absent.
 *
 * This test pins that contract:
 *   - CPU backend advertises exactly WUBU_BACKEND_CPU and no CUDA caps
 *   - every GPU-only vtable slot is NULL on the CPU backend
 *   - with a NULL model, every helper takes the CPU-fallback branch
 *   - with a CPU-backed model, every helper still takes the CPU branch
 *   - the MoE no-op must not touch the caller's expert buffer
 *   - KV namespace I/O returns -1 so callers fall back to flat tensors
 *
 * Links only src/wubu_backend.o — the CPU vtable is the whole subject.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_backend.h"
#include "wubu_model.h"
#include "wubu_moe.h"

static int g_fail = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); \
        g_fail++; \
    } else { \
        printf("  ok: "); printf(__VA_ARGS__); printf("\n"); \
    } \
} while (0)

int main(void)
{
    printf("=== test_backend_dispatch (one vtable, CPU fallback always honest) ===\n");

    /* ---- 1. CPU backend is present and claims exactly one capability ---- */
    wubu_backend_t *cpu = wubu_backend_cpu_get();
    CHECK(cpu != NULL, "wubu_backend_cpu_get() returns a vtable");
    if (!cpu) { printf("FAILED\n"); return 1; }

    CHECK(cpu->capabilities == WUBU_BACKEND_CPU,
          "CPU capabilities == WUBU_BACKEND_CPU (0x%x)", cpu->capabilities);
    CHECK(wubu_backend_has(cpu, WUBU_BACKEND_CPU), "CPU backend reports CPU cap");
    CHECK(!wubu_backend_has(cpu, WUBU_BACKEND_CUDA_GEMV), "CPU backend does not claim CUDA_GEMV");
    CHECK(!wubu_backend_has(cpu, WUBU_BACKEND_CUDA_SSM),  "CPU backend does not claim CUDA_SSM");
    CHECK(!wubu_backend_has(cpu, WUBU_BACKEND_CUDA_MOE),  "CPU backend does not claim CUDA_MOE");
    CHECK(!wubu_backend_has(cpu, WUBU_BACKEND_CUDA_ATTN), "CPU backend does not claim CUDA_ATTN");

    /* Every GPU acceleration slot must be NULL — a non-NULL slot with no
     * capability would still be skipped, but NULL is the honest CPU shape. */
    CHECK(cpu->init == NULL && cpu->free == NULL &&
          cpu->gqa_forward == NULL && cpu->ssm_forward_prefill == NULL &&
          cpu->ssm_project == NULL && cpu->moe_experts == NULL &&
          cpu->quant_matmul == NULL && cpu->set_ssm_hybrid == NULL &&
          cpu->sync_ssm_state_to_gpu == NULL && cpu->chunk_size == NULL &&
          cpu->kvfs_read == NULL && cpu->kvfs_write == NULL &&
          cpu->kvfs_snapshot == NULL,
          "CPU vtable has no GPU function pointers");

    /* ---- 2. NULL model: helpers must signal CPU fallback, never crash ---- */
    CHECK(wubu_backend_get(NULL) == NULL, "wubu_backend_get(NULL) == NULL");
    CHECK(!wubu_backend_has(NULL, WUBU_BACKEND_CPU), "wubu_backend_has(NULL,...) == false");

    float buf[8];
    for (int i = 0; i < 8; i++) buf[i] = (float)i;

    CHECK(wubu_backend_chunk_size(NULL) == 0, "chunk_size(NULL) == 0 (no GPU chunking)");
    CHECK(wubu_backend_gqa_forward(NULL, 0, buf, 8, buf) == -1,
          "gqa_forward(NULL) == -1 (caller uses CPU)");
    CHECK(wubu_backend_ssm_forward_prefill(NULL, 0, buf, 8, buf) == -1,
          "ssm_forward_prefill(NULL) == -1 (caller uses CPU)");
    CHECK(wubu_backend_ssm_project(NULL, 0, buf, 8, buf, buf, buf) == -1,
          "ssm_project(NULL) == -1 (caller uses CPU)");
    CHECK(wubu_backend_kvfs_read(NULL, "/kv/x", buf, 8) == -1,
          "kvfs_read(NULL) == -1 (flat-tensor fallback)");
    CHECK(wubu_backend_kvfs_write(NULL, "/kv/x", buf, 8) == -1,
          "kvfs_write(NULL) == -1 (flat-tensor fallback)");
    CHECK(wubu_backend_kvfs_handle_read(NULL, NULL, buf, 8) == -1,
          "kvfs_handle_read(NULL) == -1 (handle fallback)");
    CHECK(wubu_backend_kvfs_handle_write(NULL, NULL, buf, 8) == -1,
          "kvfs_handle_write(NULL) == -1 (handle fallback)");

    /* void helpers must be safe no-ops on a NULL model */
    wubu_backend_set_ssm_hybrid(NULL, 0, NULL);
    wubu_backend_sync_ssm_state_to_gpu(NULL, 0, NULL, NULL);
    wubu_backend_moe_experts(NULL, buf, NULL, NULL, NULL, NULL);
    CHECK(1, "void helpers are no-ops on NULL model (no crash)");

    /* ---- 3. CPU-backed model: still the CPU branch, every time ---- */
    wubu_model_t *m = (wubu_model_t *)calloc(1, sizeof(*m));
    if (!m) { printf("alloc failed\n"); return 1; }
    m->backend = cpu;

    CHECK(wubu_backend_get(m) == cpu, "model->backend round-trips through wubu_backend_get");
    CHECK(wubu_backend_chunk_size(m) == 0, "chunk_size(CPU model) == 0");
    CHECK(wubu_backend_gqa_forward(m, 0, buf, 8, buf) == -1,
          "gqa_forward(CPU model) == -1");
    CHECK(wubu_backend_ssm_forward_prefill(m, 0, buf, 8, buf) == -1,
          "ssm_forward_prefill(CPU model) == -1");
    CHECK(wubu_backend_ssm_project(m, 0, buf, 8, buf, buf, buf) == -1,
          "ssm_project(CPU model) == -1");
    CHECK(wubu_backend_kvfs_read(m, "/kv/x", buf, 8) == -1,
          "kvfs_read(CPU model) == -1");
    CHECK(wubu_backend_kvfs_write(m, "/kv/x", buf, 8) == -1,
          "kvfs_write(CPU model) == -1");

    /* MoE no-op must leave the caller's expert buffer untouched: the CPU
     * expert math lives in wubu_moe_forward, not in the backend. */
    float contrib[8];
    for (int i = 0; i < 8; i++) contrib[i] = -12345.0f;
    int idx[4] = {0, 1, 2, 3};
    float wgt[4] = {0.25f, 0.25f, 0.25f, 0.25f};
    wubu_backend_moe_experts(NULL, buf, idx, wgt, contrib, m);
    int untouched = 1;
    for (int i = 0; i < 8; i++)
        if (contrib[i] != -12345.0f) untouched = 0;
    CHECK(untouched, "MoE dispatch on CPU backend leaves expert buffer untouched");

    wubu_backend_set_ssm_hybrid(m, 0, NULL);
    wubu_backend_sync_ssm_state_to_gpu(m, 0, buf, buf);
    CHECK(1, "SSM hybrid helpers are no-ops on CPU model (no crash)");

    free(m);

    if (g_fail) {
        printf("=== %d BACKEND-DISPATCH CHECK(S) FAILED ===\n", g_fail);
        return 1;
    }
    printf("=== ALL BACKEND-DISPATCH TESTS PASSED ===\n");
    return 0;
}
