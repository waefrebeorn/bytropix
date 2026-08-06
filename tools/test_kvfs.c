/* tools/test_kvfs.c — triple-DA test for wubu_kvfs
 *
 * P1 (correctness): create, mount, lookup, read, write, unmount, snapshot —
 *   all operations produce the documented results.
 * P2 (privacy/no third-party): no external calls, no network,
 *   no telemetry. Pure C11 + stdlib.
 * P3 (robustness): NULL fs, NULL path, duplicate mount,
 *   out-of-bounds blocks, unmount nonexistent path, snapshot
 *   of empty namespace, read/write bounds checking.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include "wubu_kvfs.h"

static double now_s(void);
static void benchmark_hash_lookup(void);
static void benchmark_handle_io(void);

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
    /* 1000 blocks but only 1024 total — should succeed */
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

static void test_read_write(void) {
    wubu_kvfs_t *fs = wubu_kvfs_create(16, 1024);
    assert(fs);

    /* mount a region */
    assert(wubu_kvfs_mount(fs, "/kv/in", 0, 64) == 0);

    /* allocate a flat KV tensor */
    size_t total_floats = 1024 * 16; /* 1024 blocks × 16 floats/block */
    float *kv_tensor = (float *)calloc(total_floats, sizeof(float));
    assert(kv_tensor);

    /* write data to /kv/in */
    float write_buf[16];
    for (int i = 0; i < 16; i++) write_buf[i] = (float)i;
    assert(wubu_kvfs_write(fs, "/kv/in", kv_tensor, write_buf, 16) == 0);

    /* read it back */
    float read_buf[16];
    assert(wubu_kvfs_read(fs, "/kv/in", kv_tensor, read_buf, 16) == 0);
    for (int i = 0; i < 16; i++) {
        assert(read_buf[i] == (float)i);
    }

    /* read/write out of bounds should fail */
    assert(wubu_kvfs_read(fs, "/kv/in", kv_tensor, read_buf, 1025) == -1);
    assert(wubu_kvfs_write(fs, "/kv/in", kv_tensor, write_buf, 1025) == -1);

    /* nonexistent path should fail */
    assert(wubu_kvfs_read(fs, "/kv/mem", kv_tensor, read_buf, 1) == -1);
    assert(wubu_kvfs_write(fs, "/kv/mem", kv_tensor, write_buf, 1) == -1);

    free(kv_tensor);
    wubu_kvfs_free(fs);
}

static void test_prefix_lookup(void) {
    wubu_kvfs_t *fs = wubu_kvfs_create(16, 1024);
    assert(fs);

    assert(wubu_kvfs_mount(fs, "/kv", 0, 256) == 0);

    /* /kv/in is a prefix of /kv/in/layer_00 — should match /kv */
    uint32_t block = 0; size_t offset = 0;
    assert(wubu_kvfs_lookup(fs, "/kv/in/layer_00", &block, &offset) == 0);
    assert(block == 0); /* matches /kv mount */

    wubu_kvfs_free(fs);
}

static void test_handle_roundtrip(void) {
    wubu_kvfs_t *fs = wubu_kvfs_create(16, 1024);
    assert(fs);
    assert(wubu_kvfs_mount(fs, "/kv/layer_00", 0, 64) == 0);

    /* resolve once */
    wubu_kvfs_handle_t *h = wubu_kvfs_open(fs, "/kv/layer_00");
    assert(h != NULL);
    assert(wubu_kvfs_handle_offset(h) == 0);
    assert(wubu_kvfs_handle_capacity(h) == 64 * 16); /* 1024 floats */

    float *kv = (float *)calloc(1024 * 16, sizeof(float));
    assert(kv);

    /* hot write via handle */
    float src[16];
    for (int i = 0; i < 16; i++) src[i] = (float)i * 2.0f;
    assert(wubu_kvfs_handle_write(h, kv, src, 16) == 0);

    /* hot read via handle */
    float dst[16];
    assert(wubu_kvfs_handle_read(h, kv, dst, 16) == 0);
    for (int i = 0; i < 16; i++) assert(dst[i] == (float)i * 2.0f);

    /* handle bound check: exceeding capacity fails */
    assert(wubu_kvfs_handle_write(h, kv, src, 1025) == -1);
    assert(wubu_kvfs_handle_read(h, kv, dst, 1025) == -1);

    /* NULL handle / NULL base */
    assert(wubu_kvfs_handle_read(NULL, kv, dst, 1) == -1);
    assert(wubu_kvfs_handle_write(h, NULL, src, 1) == -1);

    wubu_kvfs_handle_close(h);
    /* closed handle pointer must not be reused by us */
    free(kv);
    wubu_kvfs_free(fs);
}

static void test_handle_unmounted(void) {
    wubu_kvfs_t *fs = wubu_kvfs_create(16, 1024);
    assert(fs);
    assert(wubu_kvfs_open(fs, "/kv/nope") == NULL);
    wubu_kvfs_free(fs);
}

static void test_hash_scale(void) {
    /* Many mounts: hash lookup must stay correct at scale. */
    const int N = 200;
    wubu_kvfs_t *fs = wubu_kvfs_create(16, 1 << 20);
    assert(fs);
    for (int i = 0; i < N; i++) {
        char p[64];
        snprintf(p, sizeof(p), "/kv/layer_%03d", i);
        assert(wubu_kvfs_mount(fs, p, (uint32_t)i, 1) == 0);
    }
    assert(wubu_kvfs_mount_count(fs) == N);

    /* every path resolves, including the last (worst-case old scan) */
    for (int i = 0; i < N; i++) {
        char p[64];
        snprintf(p, sizeof(p), "/kv/layer_%03d", i);
        uint32_t block = 0; size_t off = 0;
        assert(wubu_kvfs_lookup(fs, p, &block, &off) == 0);
        assert(block == (uint32_t)i);
    }

    /* resolve-once handles at scale */
    for (int i = 0; i < N; i++) {
        char p[64];
        snprintf(p, sizeof(p), "/kv/layer_%03d", i);
        wubu_kvfs_handle_t *h = wubu_kvfs_open(fs, p);
        assert(h != NULL);
        assert(wubu_kvfs_handle_offset(h) == (size_t)i * 16);
        wubu_kvfs_handle_close(h);
    }

    /* longest-prefix at scale: the deepest matching mount wins */
    assert(wubu_kvfs_mount(fs, "/kv", 0, 1) == 0);
    uint32_t block = 0; size_t off = 0;
    assert(wubu_kvfs_lookup(fs, "/kv/layer_199/extra/deep", &block, &off) == 0);
    assert(block == 199); /* deepest prefix: /kv/layer_199 */

    /* unmount at scale then verify removal: the leaf mount is gone, so
     * longest-prefix now falls back to the /kv parent (block 0). */
    char last[64];
    snprintf(last, sizeof(last), "/kv/layer_%03d", N - 1);
    assert(wubu_kvfs_unmount(fs, last) == 0);
    assert(wubu_kvfs_lookup(fs, last, &block, &off) == 0);
    assert(block == 0); /* parent /kv prefix resolves now, not layer_199 */
    assert(wubu_kvfs_mount_count(fs) == N); /* N layers + /kv - unmounted */

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
    test_read_write();
    printf("  [PASS] read/write\n");
    test_prefix_lookup();
    printf("  [PASS] prefix lookup\n");
    test_handle_roundtrip();
    printf("  [PASS] handle round-trip\n");
    test_handle_unmounted();
    printf("  [PASS] handle unmounted\n");
    test_hash_scale();
    printf("  [PASS] hash scale (200 mounts)\n");

    /* ---- speed kernel: numbers, not vibes ---- */
    benchmark_hash_lookup();
    benchmark_handle_io();

    printf("test_kvfs: ALL PASSED\n");
    return 0;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* O(1) hash lookup vs the old O(n) linear scan: 512 mounts, 1M lookups.
 * The old find_mount scanned all mounts with strncmp — this measures
 * the replacement. */
static void benchmark_hash_lookup(void) {
    const int N = 512;
    const int ITERS = 1000000;
    wubu_kvfs_t *fs = wubu_kvfs_create(16, 1 << 20);
    assert(fs);
    for (int i = 0; i < N; i++) {
        char p[64];
        snprintf(p, sizeof(p), "/kv/layer_%04d", i);
        assert(wubu_kvfs_mount(fs, p, (uint32_t)i, 1) == 0);
    }

    /* warmup */
    uint32_t block = 0; size_t off = 0;
    for (int i = 0; i < 1000; i++)
        wubu_kvfs_lookup(fs, "/kv/layer_0511", &block, &off);

    double t0 = now_s();
    for (int i = 0; i < ITERS; i++)
        wubu_kvfs_lookup(fs, "/kv/layer_0511", &block, &off);
    double dt = now_s() - t0;
    double ns = dt * 1e9 / ITERS;
    printf("  [BENCH] lookup /kv/layer_0511 (512 mounts): %.1f ns/op (%.0f M ops/s)\n",
           ns, ITERS / dt / 1e6);
    assert(block == 511);
    wubu_kvfs_free(fs);
}

/* Resolve-once handle I/O: the hot path is bounds check + memcpy.
 * 1M × 64-float writes. */
static void benchmark_handle_io(void) {
    const int ITERS = 1000000;
    wubu_kvfs_t *fs = wubu_kvfs_create(64, 1 << 20);
    assert(fs);
    assert(wubu_kvfs_mount(fs, "/kv/layer_00", 0, 1024) == 0);

    float *kv = (float *)calloc(1024 * 64, sizeof(float));
    assert(kv);
    float src[64];
    for (int i = 0; i < 64; i++) src[i] = (float)i;

    /* resolve ONCE */
    wubu_kvfs_handle_t *h = wubu_kvfs_open(fs, "/kv/layer_00");
    assert(h);

    double t0 = now_s();
    for (int i = 0; i < ITERS; i++)
        assert(wubu_kvfs_handle_write(h, kv, src, 64) == 0);
    double dt = now_s() - t0;
    double ns = dt * 1e9 / ITERS;
    printf("  [BENCH] handle write 64 floats: %.1f ns/op (%.1f GB/s)\n",
           ns, (double)ITERS * 64 * 4 / dt / 1e9);

    t0 = now_s();
    float dst[64];
    for (int i = 0; i < ITERS; i++)
        assert(wubu_kvfs_handle_read(h, kv, dst, 64) == 0);
    dt = now_s() - t0;
    ns = dt * 1e9 / ITERS;
    printf("  [BENCH] handle read 64 floats:  %.1f ns/op (%.1f GB/s)\n",
           ns, (double)ITERS * 64 * 4 / dt / 1e9);

    wubu_kvfs_handle_close(h);
    free(kv);
    wubu_kvfs_free(fs);
}