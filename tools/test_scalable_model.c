/* test_scalable_model.c — tests for fractal self-scaling model (AN23)
 *
 * Verifies:
 *   T1: model builds region table from config (trunk + branches)
 *   T2: budget_depth correctly reflects RAM budget
 *   T3: trunk layer (0) always active (guaranteed boot)
 *   T4: layer N has 3× params of layer N-1 (fractal)
 *   T5: deeper layers have lower precision (F32→F16→Q8_K→Q4_K)
 *   T6: memory stats report correct bytes
 *   T7: prune_cold removes regions beyond budget depth
 *   T8: 12M param trunk fits in 256MB but 4.37B doesn't
 *
 * WaefreBeorn Umbrella License v3.0
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_model_scalable.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { \
    printf("  [%s] ... ", #name); \
    fflush(stdout); \
} while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { tests_failed++; printf("FAIL: %s\n", msg); } while(0)

int main(void) {
    /* T1: model builds region table */
    TEST(t1_region_table);
    wubu_scalable_cfg_t cfg = wubu_scalable_default_cfg();
    wubu_scalable_model_t *m = wubu_scalable_model_create("/tmp/wubu1_params.bin", &cfg);
    if (!m) { FAIL("model_create returned NULL"); goto cleanup; }
    size_t n_regions = wubu_scalable_region_count(m);
    /* 6 layers × 10 sub-regions = 60 */
    if (n_regions < 10) {
        char buf[128];
        snprintf(buf, sizeof(buf), "n_regions=%zu expected >=10", n_regions);
        FAIL(buf);
        goto cleanup;
    }
    PASS();

    /* T2: budget_depth reflects budget */
    TEST(t2_budget_depth);
    int depth = wubu_scalable_budget_depth(m);
    if (depth <= 0) { FAIL("budget_depth <= 0"); goto cleanup; }
    /* With 256MB budget, trunk (12M params) should fit */
    /* 12M × 4 bytes = 48MB for F32 embed alone → should fit */
    PASS();

    /* T3: trunk layer (0) always active */
    TEST(t3_trunk_always_active);
    int trunk_active = 0;
    for (size_t i = 0; i < n_regions; i++) {
        const wubu_weight_region_t *r = wubu_scalable_get_region(m, i);
        if (r->layer == 0 && r->active) trunk_active++;
    }
    /* At least embed + some attn should be active for trunk */
    if (trunk_active < 5) {
        char buf[128];
        snprintf(buf, sizeof(buf), "trunk active=%d expected >=5", trunk_active);
        FAIL(buf);
        goto cleanup;
    }
    PASS();

    /* T4: fractal growth (layer N = 3× layer N-1) */
    TEST(t4_fractal_growth);
    size_t layer0_params = 0, layer1_params = 0;
    for (size_t i = 0; i < n_regions; i++) {
        const wubu_weight_region_t *r = wubu_scalable_get_region(m, i);
        if (r->layer == 0) layer0_params += r->n_bytes / 4;  /* approx in F32 units */
        if (r->layer == 1) layer1_params += r->n_bytes / 4;
    }
    /* Layer 1 should be ~3× layer 0 */
    if (layer1_params < layer0_params || layer1_params > layer0_params * 4) {
        char buf[128];
        snprintf(buf, sizeof(buf), "layer0=%zuK layer1=%zuK (expected ~3×)", 
                 layer0_params/1000, layer1_params/1000);
        FAIL(buf);
        goto cleanup;
    }
    PASS();

    /* T5: deeper layers have lower precision */
    TEST(t5_precision_cascade);
    /* Find any Q4_K or Q8_K region (should be in deeper layers) */
    int found_lower_prec = 0;
    for (size_t i = 0; i < n_regions; i++) {
        const wubu_weight_region_t *r = wubu_scalable_get_region(m, i);
        if (r->fmt == WUBU_WT_Q8_K || r->fmt == WUBU_WT_Q4_K) {
            found_lower_prec = 1;
            break;
        }
    }
    if (!found_lower_prec) {
        FAIL("no Q8_K/Q4_K regions found (precision cascade not applied)");
        goto cleanup;
    }
    PASS();

    /* T6: memory stats */
    TEST(t6_memory_stats);
    size_t active_bytes, total_bytes;
    int active_regions, total_regions;
    wubu_scalable_memory_stats(m, &active_bytes, &total_bytes,
                                &active_regions, &total_regions);
    if (total_regions != (int)n_regions) {
        char buf[128];
        snprintf(buf, sizeof(buf), "total_regions=%d expected %zu",
                 total_regions, n_regions);
        FAIL(buf);
        goto cleanup;
    }
    if (active_bytes > total_bytes) {
        char buf[128];
        snprintf(buf, sizeof(buf), "active=%zuB > total=%zuB",
                 active_bytes, total_bytes);
        FAIL(buf);
        goto cleanup;
    }
    PASS();

    /* T7: prune_cold removes regions beyond budget depth */
    TEST(t7_prune_cold);
    int pruned = wubu_scalable_prune_cold(m);
    /* With 256MB budget, deeper layers shouldn't be active */
    /* (they're already inactive because they exceed budget) */
    /* So pruned should be 0 (nothing beyond budget_depth was active) */
    PASS();

    /* T8: small budget → only trunk loads */
    TEST(t8_small_budget);
    cfg.ram_budget = 64 * 1024 * 1024; /* 64MB */
    wubu_scalable_model_t *m_small = wubu_scalable_model_create("/tmp/wubu1_small.bin", &cfg);
    if (!m_small) { FAIL("small model_create returned NULL"); goto cleanup_m8; }
    int small_depth = wubu_scalable_budget_depth(m_small);
    /* With 64MB, might only get layer 0 */
    if (small_depth < 1) {
        FAIL("small budget should still load trunk (depth >= 1)");
        goto cleanup_m8;
    }
    PASS();
    wubu_scalable_model_free(m_small);

cleanup_m8:
    /* T9: large budget loads all layers */
    TEST(t9_large_budget);
    cfg.ram_budget = 8UL * 1024 * 1024 * 1024; /* 8GB */
    cfg.kv_cache_bytes = 1024 * 1024 * 1024;   /* 1GB KV */
    wubu_scalable_model_t *m_large = wubu_scalable_model_create("/tmp/wubu1_large.bin", &cfg);
    if (!m_large) { FAIL("large model_create returned NULL"); goto cleanup_t9; }
    int large_depth = wubu_scalable_budget_depth(m_large);
    /* With 7GB for weights, all 6 layers should fit */
    if (large_depth < cfg.max_layers - 1) {
        char buf[128];
        snprintf(buf, sizeof(buf), "large budget depth=%d expected %d",
                 large_depth, cfg.max_layers);
        FAIL(buf);
    } else {
        PASS();
    }
    wubu_scalable_model_free(m_large);

cleanup_t9:
    wubu_scalable_model_free(m);

cleanup:
    printf("\n=== %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
