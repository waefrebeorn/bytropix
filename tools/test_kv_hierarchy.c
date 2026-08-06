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

    printf("\n=== %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
