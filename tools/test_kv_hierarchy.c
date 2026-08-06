/* test_kv_hierarchy.c — tests for hyperbolic KV namespace addressing
 *
 * Verifies:
 *   T1: root maps to center (radius ≈ 0)
 *   T2: deeper paths → larger radius (further from center)
 *   T3: sibling paths (same parent, different leaf) are spread by angle
 *   T4: parent and child are close (parent depth < child depth →
 *       parent closer to center)
 *   T5: root-to-any path distance > sibling-to-sibling distance
 *   T6: routing score: siblings score higher than distant paths
 *   T7: nearest-neighbor finds the correct file
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 6 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_kv_hierarchy.h"
#include "wubu_kv_semantic_router.h"
#include "wubu_kv_shrink.h"
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
    const char *kv_root = "/kv/in";
    wubu_kv_hyperbolic_cfg_t cfg = wubu_kv_hyperbolic_default_cfg();
    float R = cfg.R;

    /* T1: root maps to center */
    TEST(t1_root_center);
    wubu_kv_point_t root_pt = wubu_kv_path_to_point(kv_root, "/kv/in", &cfg);
    if (root_pt.r > 0.01f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "root r=%.4f expected ~0", root_pt.r);
        FAIL(buf);
    } else PASS();

    /* T2: deeper paths → larger radius */
    TEST(t2_deeper_larger_radius);
    wubu_kv_point_t shallow = wubu_kv_path_to_point(kv_root, "/kv/in/foo.c", &cfg);
    wubu_kv_point_t deep = wubu_kv_path_to_point(kv_root, "/kv/in/src/deep/path/bar.c", &cfg);
    if (shallow.r >= deep.r) {
        char buf[128];
        snprintf(buf, sizeof(buf), "shallow r=%.4f >= deep r=%.4f",
                 shallow.r, deep.r);
        FAIL(buf);
    } else PASS();

    /* T3: sibling paths are spread by angle */
    TEST(t3_siblings_different_angles);
    wubu_kv_point_t s1 = wubu_kv_path_to_point(kv_root, "/kv/in/foo.c", &cfg);
    wubu_kv_point_t s2 = wubu_kv_path_to_point(kv_root, "/kv/in/bar.c", &cfg);
    float d_theta = fabsf(s2.theta - s1.theta);
    if (d_theta < 0.1f) {  /* siblings should be spread out */
        char buf[128];
        snprintf(buf, sizeof(buf), "theta diff=%.4f expected >0.1", d_theta);
        FAIL(buf);
    } else PASS();

    /* T4: parent and child — parent closer to center */
    TEST(t4_parent_child);
    wubu_kv_point_t parent = wubu_kv_path_to_point(kv_root, "/kv/in/src", &cfg);
    wubu_kv_point_t child = wubu_kv_path_to_point(kv_root, "/kv/in/src/foo.c", &cfg);
    if (parent.r >= child.r) {
        char buf[128];
        snprintf(buf, sizeof(buf), "parent r=%.4f >= child r=%.4f",
                 parent.r, child.r);
        FAIL(buf);
    } else PASS();

    /* T5: root-to-deep distance > sibling-to-sibling distance */
    TEST(t5_dist_hierarchy_vs_siblings);
    float d_root_deep = wubu_kv_path_distance(kv_root, "/kv/in", "/kv/in/src/foo.c", &cfg);
    float d_sib = wubu_kv_path_distance(kv_root, "/kv/in/foo.c", "/kv/in/bar.c", &cfg);
    if (d_root_deep <= d_sib) {
        char buf[128];
        snprintf(buf, sizeof(buf), "root-deep d=%.4f <= sibling d=%.4f",
                 d_root_deep, d_sib);
        FAIL(buf);
    } else PASS();

    /* T6: routing score — siblings higher than distant */
    TEST(t6_routing_score);
    float score_siblings = wubu_kv_path_routing_score(kv_root, "/kv/in/foo.c", "/kv/in/bar.c", &cfg);
    float score_distant = wubu_kv_path_routing_score(kv_root, "/kv/in", "/kv/in/src/deep/nested/file.txt", &cfg);
    if (score_siblings <= score_distant) {
        char buf[128];
        snprintf(buf, sizeof(buf), "sibling score=%.4f <= distant score=%.4f",
                 score_siblings, score_distant);
        FAIL(buf);
    } else PASS();

    /* T7: nearest neighbor */
    TEST(t7_nearest_neighbor);
    const char *paths[] = { "/kv/in/a.c", "/kv/in/src/b.c", "/kv/in/src/deep/c.c" };
    const char *query = "/kv/in/src/b.c";
    int nn = wubu_kv_path_nearest(kv_root, paths, 3, query, &cfg);
    if (nn != 1) {  /* paths[1] = "/kv/in/src/b.c" should be nearest to itself */
        char buf[128];
        snprintf(buf, sizeof(buf), "nn index=%d expected 1", nn);
        FAIL(buf);
    } else PASS();

    /* T8: semantic router bias — siblings get higher bias than distant */
    TEST(t8_router_bias_siblings);
    const char *router_paths[] = { "/kv/in/foo.c", "/kv/in/bar.c", "/kv/in/src/deep/nested/file.txt" };
    wubu_kv_router_t *rt = wubu_kv_router_create(kv_root, router_paths, 3, &cfg, 2.0f);
    if (!rt) { FAIL("router_create returned NULL"); goto summary; }
    float bias[3];
    wubu_kv_router_bias(rt, "/kv/in/foo.c", bias);
    /* foo.c should have highest bias for itself (distance ~0) */
    if (bias[0] < 1.0f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "self-bias=%.4f expected ~2.0 (scale*dist~0)", bias[0]);
        FAIL(buf);
        goto cleanup_rt;
    }
    /* bar.c is a sibling — should be higher than deep/nested/file.txt */
    if (bias[1] < bias[2]) {
        char buf[128];
        snprintf(buf, sizeof(buf), "sibling bias=%.4f < distant=%.4f", bias[1], bias[2]);
        FAIL(buf);
        goto cleanup_rt;
    }
    PASS();

    /* T9: router pair bias — same path = scale (max), different = < scale */
    TEST(t9_router_pair_bias);
    float self_bias = wubu_kv_router_bias_pair(rt, "/kv/in/foo.c", "/kv/in/foo.c");
    float far_bias = wubu_kv_router_bias_pair(rt, "/kv/in/foo.c", "/kv/in/src/deep/nested/file.txt");
    if (self_bias < 1.9f || self_bias > 2.1f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "self_bias=%.4f expected ~2.0", self_bias);
        FAIL(buf);
        goto cleanup_rt;
    }
    if (far_bias >= 0.5f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "far_bias=%.4f expected < 0.5", far_bias);
        FAIL(buf);
        goto cleanup_rt;
    }
    if (far_bias <= 0.0f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "far_bias=%.4f expected > 0", far_bias);
        FAIL(buf);
        goto cleanup_rt;
    }
    PASS();

cleanup_rt:
    wubu_kv_router_free(rt);

    /* T10: KV shrink prunes cold regions */
    TEST(t10_kv_shrink_prune);
    wubu_kvfs_t *sfs = wubu_kvfs_create(256, 16);
    wubu_kv_shrink_t *sz = wubu_kv_shrink_create(sfs, &(wubu_kv_shrink_cfg_t){
        .util_threshold = 0.01f, .cold_iters = 3, .min_regions = 1
    });
    if (!sz) { FAIL("shrink_create returned NULL"); goto cleanup_s10; }

    /* 3 regions: A and B are cold, C is warm */
    const char *spaths[] = { "/kv/in/a.c", "/kv/in/b.c", "/kv/in/c.c" };
    float sutil[] = { 0.001f, 0.001f, 0.5f };

    /* Feed 3 cold iterations for A and B, warm for C */
    for (int iter = 0; iter < 3; iter++) {
        wubu_kv_shrink_feed(sz, spaths, sutil, 3);
    }

    char **pruned_paths = NULL;
    int pruned = wubu_kv_shrink_sweep(sz, &pruned_paths);
    /* A and B should be pruned (cold_streak >= 3), C stays.
     * min_regions=1 means we stop at 1 remaining. So 2 pruned. */
    /* But we only mounted 3 regions via feed (no actual mount).
     * The shrink operator tracks them but hasn't mounted.
     * unmount on unmounted path should be safe. Let's just check
     * the count of pruned regions in the tracking. */
    if (pruned < 2) {
        char buf[128];
        snprintf(buf, sizeof(buf), "pruned=%d expected >=2", pruned);
        FAIL(buf);
        if (pruned_paths) {
            for (int i = 0; i < pruned; i++) free(pruned_paths[i]);
            free(pruned_paths);
        }
        goto cleanup_s10;
    }
    PASS();

cleanup_s10:
    if (pruned_paths) {
        for (int i = 0; i < pruned; i++) free(pruned_paths[i]);
        free(pruned_paths);
    }
    wubu_kv_shrink_free(sz);
    wubu_kvfs_free(sfs);

summary:
    return tests_failed > 0 ? 1 : 0;
}
