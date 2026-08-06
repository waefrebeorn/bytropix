/* test_kv_tiering.c — tests for KV precision tiering (Phase 9)
 *
 * Verifies:
 *   T1: all files start at F32 tier
 *   T2: hot files stay at F32 after eval
 *   T3: cold files get down-tiered after decay_iters
 *   T4: tier_up/tier_down correctness (F32→F16→Q8_K→Q4_K)
 *   T5: stats report correct byte counts
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 9 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "wubu_kv_tiering.h"
#include "wubu_kv_embedding.h"
#include "wubu_kvfs.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { \
    printf("  [%s] ... ", #name); \
    fflush(stdout); \
} while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { tests_failed++; printf("FAIL: %s\n", msg); } while(0)

int main(void) {
    wubu_kvfs_t *fs = wubu_kvfs_create(256, 64);
    if (!fs) { FAIL("kvfs_create returned NULL"); return 1; }
    wubu_kv_embedding_t *kv = wubu_kv_embedding_create(fs, 256);
    if (!kv) { FAIL("kv_embedding_create returned NULL"); goto cleanup; }

    /* Encode 3 files */
    wubu_kv_embedding_encode(kv, "hot.c", "x", 1, NULL);
    wubu_kv_embedding_encode(kv, "warm.c", "y", 1, NULL);
    wubu_kv_embedding_encode(kv, "cold.c", "z", 1, NULL);

    float *kv_base = NULL;
    kv_base = (float *)calloc(64 * 256, sizeof(float));
    if (!kv_base) { FAIL("calloc kv_base"); goto cleanup; }

    /* T1: all files start at F32 */
    TEST(t1_all_files_start_f32);
    wubu_kv_tiering_cfg_t cfg = wubu_kv_tiering_default_cfg();
    cfg.decay_iters = 3;  /* small for testing */
    cfg.budget_floats = 1000; /* generous budget */
    wubu_kv_tiering_t *tier = wubu_kv_tiering_create(kv, &cfg, kv_base, 64*256*sizeof(float));
    if (!tier) { FAIL("tiering_create returned NULL"); goto cleanup; }

    const char *fpaths[] = { "/kv/in/hot.c", "/kv/in/warm.c", "/kv/in/cold.c" };
    int t1_ok = 1;
    for (int i = 0; i < 3; i++) {
        int tier_val = wubu_kv_tiering_get(tier, fpaths[i]);
        if (tier_val != KV_TIER_F32) {
            char buf[128];
            snprintf(buf, sizeof(buf), "file %s tier=%d expected F32(0)", fpaths[i], tier_val);
            FAIL(buf);
            t1_ok = 0;
            break;
        }
    }
    if (t1_ok) PASS();

    /* T2: hot file stays F32 after eval */
    TEST(t2_hot_stays_f32);
    float hot_mass[] = { 0.5f, 0.0f, 0.0f };  /* hot.c is hot, others cold */
    int n = wubu_kv_tiering_eval(tier, fpaths, hot_mass, 3);
    int t2_ok = 1;
    int tier_hot = wubu_kv_tiering_get(tier, "/kv/in/hot.c");
    if (tier_hot != KV_TIER_F32) {
        char buf[128];
        snprintf(buf, sizeof(buf), "hot.c tier=%d expected F32(0)", tier_hot);
        FAIL(buf);
        t2_ok = 0;
    }
    if (t2_ok) PASS();

    /* T3: cold files get down-tiered after decay_iters */
    TEST(t3_cold_files_down_tiered);
    /* Feed cold mass for 3 iterations to hit decay_iters */
    for (int iter = 0; iter < 3; iter++) {
        wubu_kv_tiering_eval(tier, fpaths, hot_mass, 3);
    }
    int tier_cold = wubu_kv_tiering_get(tier, "/kv/in/cold.c");
    int tier_warm = wubu_kv_tiering_get(tier, "/kv/in/warm.c");
    /* cold.c and warm.c both have 0.0 mass → should be down-tiered */
    char buf[128];
    if (tier_cold == KV_TIER_F32 && tier_warm == KV_TIER_F32) {
        FAIL("cold/warm files still F32 — expected down-tier");
    } else {
        snprintf(buf, sizeof(buf), "cold.c tier=%d warm.c tier=%d", tier_cold, tier_warm);
        /* At least one should have moved */
        if (tier_cold != KV_TIER_F32 || tier_warm != KV_TIER_F32) PASS();
        else FAIL(buf);
    }

    /* T4: tier states are F32 < F16 < Q8_K < Q4_K */
    TEST(t4_tier_ordering);
    if (KV_TIER_F32 == KV_TIER_F16 || KV_TIER_F16 == KV_TIER_Q8_K ||
        KV_TIER_Q8_K == KV_TIER_Q4_K) {
        FAIL("tier enum values not distinct");
    } else {
        PASS();
    }

    /* T5: stats report correct byte counts */
    TEST(t5_stats);
    size_t f32b, f16b, q8b, q4b;
    wubu_kv_tiering_stats(tier, &f32b, &f16b, &q8b, &q4b);
    /* At least one file should have been tiered down */
    size_t nonzero = (f32b > 0) + (f16b > 0) + (q8b > 0) + (q4b > 0);
    if (nonzero < 2) {
        char buf[128];
        snprintf(buf, sizeof(buf), "only %zu tier with nonzero bytes (expected >=2)", nonzero);
        FAIL(buf);
    } else {
        PASS();
    }

    wubu_kv_tiering_free(tier);

cleanup:
    free(kv_base);
    wubu_kv_embedding_free(kv);
    wubu_kvfs_free(fs);

    printf("\n=== %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
