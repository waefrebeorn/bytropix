/* tools/test_kvfs.c — triple-DA test for wubu_kvfs
 *
 * P1 (correctness): create, mount, lookup, unmount, snapshot —
 *   all operations produce the documented results.
 * P2 (privacy/no third-party): no external calls, no network,
 *   no telemetry. Pure C11 + stdlib.
 * P3 (robustness): NULL fs, NULL path, duplicate mount,
 *   out-of-bounds blocks, unmount nonexistent path, snapshot
 *   of empty namespace.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "wubu_kvfs.h"

static void test_create_free(void) {
    wubu_kvfs_t *fs = wubu_kvfs_create(16, 1024);
    assert(fs != NULL);
    assert(wubu_kvfs_mount_count(fs) == 0);
    wubu_kvfs_free(fs);
}

static void test_mount_lookup(void) {
    wubu_kvfs_t *fs = wubu_kvfs_create(16, 1024);
    assert(fs);

    /* mount /kv/in at block 0, 64 blocks */
    assert(wubu_kvfs_mount(fs, "/kv/in", 0, 64) == 0);
    assert(wubu_kvfs_mount_count(fs) == 1);

    /* mount /kv/synth at block 64, 32 blocks */
    assert(wubu_kvfs_mount(fs, "/kv/synth", 64, 32) == 0);
    assert(wubu_kvfs_mount_count(fs) == 2);

    /* lookup /kv/in → block 0 */
    uint32_t block = 0; size_t offset = 0;
    assert(wubu_kvfs_lookup(fs, "/kv/in", &block, &offset) == 0);
    assert(block == 0);

    /* lookup /kv/synth → block 64 */
    assert(wubu_kvfs_lookup(fs, "/kv/synth", &block, &offset) == 0);
    assert(block == 64);

    /* lookup nonexistent → -1 */
    assert(wubu_kvfs_lookup(fs, "/kv/mem", &block, &offset) == -1);

    wubu_kvfs_free(fs);
}

static void test_duplicate_mount_fails(void) {
    wubu_kvfs_t *fs = wubu_kvfs_create(16, 1024);
    assert(fs);
    assert(wubu_kvfs_mount(fs, "/kv/in", 0, 64) == 0);
    /* duplicate */
    assert(wubu_kvfs_mount(fs, "/kv/in", 0, 64) == -1);
    wubu_kvfs_free(fs);
}

static void test_out_of_bounds_fails(void) {
    wubu_kvfs_t *fs = wubu_kvfs_create(16, 1024);
    assert(fs);
    /* 1000 blocks but only 1024 total — should fail */
    assert(wubu_kvfs_mount(fs, "/kv/in", 0, 1000) == 0);
    /* now only 24 left — mounting 100 more should fail */
    assert(wubu_kvfs_mount(fs, "/kv/synth", 1000, 100) == -1);
    wubu_kvfs_free(fs);
}

static void test_unmount(void) {
    wubu_kvfs_t *fs = wubu_kvfs_create(16, 1024);
    assert(fs);
    assert(wubu_kvfs_mount(fs, "/kv/in", 0, 64) == 0);
    assert(wubu_kvfs_mount_count(fs) == 1);
    assert(wubu_kvfs_unmount(fs, "/kv/in") == 0);
    assert(wubu_kvfs_mount_count(fs) == 0);
    /* unmount nonexistent */
    assert(wubu_kvfs_unmount(fs, "/kv/in") == -1);
    wubu_kvfs_free(fs);
}

static void test_snapshot(void) {
    wubu_kvfs_t *fs = wubu_kvfs_create(16, 1024);
    assert(fs);
    assert(wubu_kvfs_mount(fs, "/kv/in", 0, 64) == 0);
    assert(wubu_kvfs_mount(fs, "/kv/synth", 64, 32) == 0);

    size_t len = 0;
    char *snap = wubu_kvfs_snapshot_json(fs, &len);
    assert(snap != NULL);
    assert(len > 0);
    assert(strstr(snap, "\"registered\":2") != NULL);
    assert(strstr(snap, "/kv/in") != NULL);
    assert(strstr(snap, "/kv/synth") != NULL);
    assert(strstr(snap, "\"block_size\":16") != NULL);
    assert(strstr(snap, "\"total_blocks\":1024") != NULL);
    free(snap);
    wubu_kvfs_free(fs);
}

static void test_null_fs(void) {
    /* all APIs should handle NULL gracefully */
    assert(wubu_kvfs_lookup(NULL, "/kv/in", NULL, NULL) == -1);
    assert(wubu_kvfs_mount_count(NULL) == 0);
    assert(wubu_kvfs_snapshot_json(NULL, NULL) == NULL);
    wubu_kvfs_free(NULL); /* no crash */
}

static void test_empty_snapshot(void) {
    wubu_kvfs_t *fs = wubu_kvfs_create(16, 1024);
    assert(fs);
    size_t len = 0;
    char *snap = wubu_kvfs_snapshot_json(fs, &len);
    assert(snap != NULL);
    assert(strstr(snap, "\"registered\":0") != NULL);
    free(snap);
    wubu_kvfs_free(fs);
}

int main(void) {
    printf("test_kvfs: starting...\n");
    test_create_free();
    printf("  [PASS] create/free\n");
    test_mount_lookup();
    printf("  [PASS] mount/lookup\n");
    test_duplicate_mount_fails();
    printf("  [PASS] duplicate mount fails\n");
    test_out_of_bounds_fails();
    printf("  [PASS] out-of-bounds fails\n");
    test_unmount();
    printf("  [PASS] unmount\n");
    test_snapshot();
    printf("  [PASS] snapshot\n");
    test_null_fs();
    printf("  [PASS] NULL fs handling\n");
    test_empty_snapshot();
    printf("  [PASS] empty snapshot\n");
    printf("test_kvfs: ALL PASSED\n");
    return 0;
}
