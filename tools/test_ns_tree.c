/* test_ns_tree.c -- WuBuOS /n/* 9P namespace tree (NAMESPACE #178-#189).
 * Proves: is_ns detection, subsystem classification for every /n/* path,
 * dynamic segment capture ({slot},{cell},{metric}), dir vs file, fail-closed
 * on non-/n and unknown. */
#include "wubu_ns_tree.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_ns_resolved_t r;

    /* is_ns */
    CHECK(wubu_ns_tree_is_ns("/n") == 1, "/n is ns");
    CHECK(wubu_ns_tree_is_ns("/n/bonzi") == 1, "/n/bonzi is ns");
    CHECK(wubu_ns_tree_is_ns("/etc/passwd") == 0, "/etc not ns");
    CHECK(wubu_ns_tree_is_ns("/nx") == 0, "/nx not ns");
    CHECK(wubu_ns_tree_is_ns(NULL) == 0, "NULL not ns");

    /* /n root = kernel dir */
    CHECK(wubu_ns_tree_resolve("/n", &r) == 0, "resolve /n");
    CHECK(r.subsys == WUBU_NS_KERNEL && r.is_dir == 1, "/n = kernel dir");

    /* brain internals */
    CHECK(wubu_ns_tree_resolve("/n/brain", &r) == 0, "resolve /n/brain");
    CHECK(r.subsys == WUBU_NS_BRAIN && r.is_dir == 1, "/n/brain = brain dir");
    CHECK(wubu_ns_tree_resolve("/n/brain/weights", &r) == 0, "resolve /n/brain/weights");
    CHECK(r.subsys == WUBU_NS_BRAIN && r.is_dir == 0, "/n/brain/weights = file");

    /* bonzi + bonzi/mem */
    CHECK(wubu_ns_tree_resolve("/n/bonzi", &r) == 0 && r.subsys == WUBU_NS_BONZI && r.is_dir == 1, "/n/bonzi dir");
    CHECK(wubu_ns_tree_resolve("/n/bonzi/mem", &r) == 0 && r.subsys == WUBU_NS_BONZI && r.is_dir == 0, "/n/bonzi/mem file");
    CHECK(wubu_ns_tree_resolve("/n/bonzi/enabled", &r) == 0 && r.subsys == WUBU_NS_BONZI, "/n/bonzi/enabled");

    /* kv, dram, models */
    CHECK(wubu_ns_tree_resolve("/n/kv", &r) == 0 && r.subsys == WUBU_NS_KV, "/n/kv");
    CHECK(wubu_ns_tree_resolve("/n/dram", &r) == 0 && r.subsys == WUBU_NS_DRAM, "/n/dram");
    CHECK(wubu_ns_tree_resolve("/n/models", &r) == 0 && r.subsys == WUBU_NS_MODELS && r.is_dir == 1, "/n/models dir");

    /* experts/{slot} dynamic capture */
    CHECK(wubu_ns_tree_resolve("/n/experts", &r) == 0 && r.subsys == WUBU_NS_EXPERTS && r.is_dir == 1, "/n/experts dir");
    CHECK(wubu_ns_tree_resolve("/n/experts/3", &r) == 0 && r.subsys == WUBU_NS_EXPERTS, "/n/experts/3");
    CHECK(strcmp(r.dynamic, "3") == 0, "expert slot 3 captured");
    CHECK(r.is_dir == 1, "/n/experts/3 is a slot dir");
    CHECK(wubu_ns_tree_resolve("/n/experts/3/weights", &r) == 0 && r.is_dir == 0, "/n/experts/3/weights is a file");

    /* hive/{cell} dynamic capture */
    CHECK(wubu_ns_tree_resolve("/n/hive/cell7", &r) == 0 && r.subsys == WUBU_NS_HIVE, "/n/hive/cell7");
    CHECK(strcmp(r.dynamic, "cell7") == 0, "hive cell captured");

    /* train/{metric} dynamic capture */
    CHECK(wubu_ns_tree_resolve("/n/train/loss", &r) == 0 && r.subsys == WUBU_NS_TRAIN, "/n/train/loss");
    CHECK(strcmp(r.dynamic, "loss") == 0, "train metric captured");

    /* fail-closed: unknown subsystem under /n */
    CHECK(wubu_ns_tree_resolve("/n/bogus", &r) == -1, "unknown /n subsystem -> -1");
    CHECK(r.subsys == WUBU_NS_UNKNOWN, "unknown subsys flag");

    /* fail-closed: non-/n path */
    CHECK(wubu_ns_tree_resolve("/home/wubu/x", &r) == -1, "non-/n path -> -1");

    /* fail-closed: NULL */
    CHECK(wubu_ns_tree_resolve(NULL, &r) == -1, "NULL path -> -1");

    /* subsys names */
    CHECK(strcmp(wubu_ns_tree_subsys_name(WUBU_NS_KV), "kv") == 0, "name kv");
    CHECK(strcmp(wubu_ns_tree_subsys_name(WUBU_NS_HIVE), "hive") == 0, "name hive");

    printf("\n=== test_ns_tree: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
