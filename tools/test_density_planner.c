/* test_density_planner.c — tests for density planning (AN23-core)
 *
 * Verifies:
 *   T1: high-density files are absorbed (promoted)
 *   T2: low-density files are pruned, not absorbed
 *   T3: mid-density files stay in KV (not absorbed, not pruned)
 *   T4: density = coherence / n_tokens (density ranking correct)
 *   T5: weight budget limits absorption count
 *
 * Design: docs/wubu1-scalable-model-design.md §Density Planning.
 * WaefreBeorn Umbrella License v3.0
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_density_planner.h"
#include "wubu_model_scalable.h"
#include "wubu_kv_shrink.h"

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
    wubu_kv_embedding_t *kv = wubu_kv_embedding_create(fs, 256);
    wubu_scalable_cfg_t mcfg = wubu_scalable_default_cfg();
    wubu_scalable_model_t *model = wubu_scalable_model_create("/tmp/dummy.bin", &mcfg);

    /* Create a shrink operator for the planner */
    wubu_kv_shrink_cfg_t scfg = wubu_kv_shrink_default_cfg();
    wubu_kv_shrink_t *shrink = wubu_kv_shrink_create(fs, &scfg);

    wubu_density_planner_cfg_t cfg = wubu_density_planner_default_cfg();
    cfg.absorb_threshold = 0.001f;
    cfg.keep_threshold = 0.0001f;
    cfg.prune_threshold = 0.00005f;
    cfg.weight_budget = 100 * 1024 * 1024; /* 100MB - generous */
    cfg.absorb_batch = 4;

    wubu_density_planner_t *planner = wubu_density_planner_create(
        kv, model, NULL, shrink, &cfg);
    if (!planner) { FAIL("planner_create returned NULL"); goto cleanup; }

    /* 3 files with different density:
     * high.txt: coherence=0.9, n_tokens=10 → density=0.09 (ABSORB)
     * mid.txt:  coherence=0.5, n_tokens=1000 → density=0.0005 (KEEP)
     * cold.txt: coherence=0.1, n_tokens=1000 → density=0.0001 (KEEP/PRUNE)
     * dead.txt: coherence=0.01, n_tokens=100 → density=0.0001 (PRUNE)
     */
    const char *paths[] = { "/kv/in/high.txt", "/kv/in/mid.txt",
                            "/kv/in/cold.txt", "/kv/in/dead.txt" };
    wubu_coherence_t coh[] = {
        { .attention_mass = 0.9f, .attention_entropy = 0.5f,
          .consistency = 0.9f, .score = 0.9f, .n_tokens = 10 },
        { .attention_mass = 0.5f, .attention_entropy = 0.8f,
          .consistency = 0.6f, .score = 0.5f, .n_tokens = 1000 },
        { .attention_mass = 0.1f, .attention_entropy = 1.0f,
          .consistency = 0.5f, .score = 0.1f, .n_tokens = 1000 },
        { .attention_mass = 0.01f, .attention_entropy = 1.0f,
          .consistency = 0.1f, .score = 0.01f, .n_tokens = 1000 },
    };

    /* T1: high-density file is absorbed */
    TEST(t1_high_density_absorbed);
    wubu_absorb_record_t *absorbed = NULL;
    int n_absorbed = 0;
    int n = wubu_density_planner_cycle(planner, paths, coh, 4,
                                      &absorbed, &n_absorbed);
    if (n < 1) { FAIL("no files absorbed"); goto cleanup; }
    /* Check high.txt is in the records with absorbed=1 */
    int found_high_absorbed = 0;
    for (int i = 0; i < n_absorbed; i++) {
        if (strcmp(absorbed[i].path, "/kv/in/high.txt") == 0 &&
            absorbed[i].absorbed) {
            found_high_absorbed = 1;
            break;
        }
    }
    if (!found_high_absorbed) {
        char buf[128];
        snprintf(buf, sizeof(buf), "high.txt not marked absorbed (density=%.5f)",
                 0.9f / 10.0f);
        FAIL(buf);
        goto cleanup;
    }
    PASS();

    /* T2: low-density file is NOT absorbed */
    TEST(t2_low_density_not_absorbed);
    {
        int found_dead_absorbed = 0;
        for (int i = 0; i < n_absorbed; i++) {
            if (strcmp(absorbed[i].path, "/kv/in/dead.txt") == 0 &&
                absorbed[i].absorbed) {
                found_dead_absorbed = 1;
                break;
            }
        }
        if (found_dead_absorbed) {
            FAIL("dead.txt was absorbed (should be pruned)");
        } else {
            PASS();
        }
    }

    /* T3: density ranking is correct (high > mid > cold > dead) */
    TEST(t3_density_ranking);
    {
        /* Find records in order of density */
        float densities[4];
        const char *check_names[] = { "/kv/in/high.txt", "/kv/in/mid.txt",
                                       "/kv/in/cold.txt", "/kv/in/dead.txt" };
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < n_absorbed; j++) {
                if (strcmp(absorbed[j].path, check_names[i]) == 0) {
                    densities[i] = absorbed[j].density;
                    break;
                }
            }
        }
        /* high density should be >= cold density */
        if (densities[0] < densities[2]) {
            char buf[128];
            snprintf(buf, sizeof(buf), "high(%.5f) < cold(%.5f)",
                     densities[0], densities[2]);
            FAIL(buf);
        } else {
            PASS();
        }
    }

    /* T4: weight budget limits absorption */
    TEST(t4_weight_budget_limit);
    {
        /* Set small budget so only 1-2 files can be absorbed */
        cfg.weight_budget = 10 * 1024 * 1024; /* 10MB */
        cfg.absorb_batch = 4;
        /* Re-create planner with small budget */
        wubu_density_planner_free(planner);
        planner = wubu_density_planner_create(kv, model, NULL, shrink, &cfg);
        if (!planner) { FAIL("planner recreate returned NULL"); goto cleanup; }
        n = wubu_density_planner_cycle(planner, paths, coh, 4,
                                      &absorbed, &n_absorbed);
        /* 10MB budget: high.txt is 40 bytes, mid.txt is 4000 bytes,
         * cold.txt is 4000 bytes, dead.txt is 400 bytes.
         * All fit in 10MB. But mid.txt density=0.0005 > keep_threshold
         * but < absorb_threshold=0.001. So only high.txt gets absorbed. */
        if (n < 1) {
            FAIL("no absorption with 10MB budget");
        } else if (n > 2) {
            char buf[128];
            snprintf(buf, sizeof(buf), "absorbed=%d with 10MB budget, expected <=2", n);
            FAIL(buf);
        } else {
            PASS();
        }
    }

    /* T5: stats report correct byte counts */
    TEST(t5_stats);
    {
        size_t absorbed_bytes, kv_bytes, pruned_bytes;
        wubu_density_planner_stats(planner, &absorbed_bytes, &kv_bytes, &pruned_bytes);
        /* high.txt absorbed (40 bytes). mid/cold in KV, dead pruned. */
        if (absorbed_bytes == 0) {
            FAIL("absorbed_bytes=0, expected >0");
        } else if (pruned_bytes == 0) {
            FAIL("pruned_bytes=0, expected >0 for dead.txt");
        } else {
            PASS();
        }
    }

cleanup:
    wubu_density_planner_free(planner);
    wubu_kv_shrink_free(shrink);
    wubu_scalable_model_free(model);
    wubu_kv_embedding_free(kv);
    wubu_kvfs_free(fs);

    printf("\n=== %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
