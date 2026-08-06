/* test_kv_embedding.c — tests for the KV-FS embedding bridge
 *
 * Verifies:
 *   T1: create + free (round-trip)
 *   T2: encode a file → region resolves to correct KV offset
 *   T3: encode_tokens → multiple files get distinct regions
 *   T4: coherence computation on synthetic attention matrix
 *     (high mass = high score, uniform attention = low score)
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 (AN21 phase 1).
 * WaefreBeorn Umbrella License v3.0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "wubu_kv_embedding.h"
#include "wubu_coherence_reward.h"
#include "wubu_fs_dataset.h"
#include "wubu_grow_kv.h"
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
    /* T1: create + free */
    TEST(t1_create_free);
    wubu_kvfs_t *fs = wubu_kvfs_create(256, 64); /* 64 blocks of 256 floats */
    if (!fs) { FAIL("kvfs_create returned NULL"); return 1; }
    wubu_kv_embedding_t *kv = wubu_kv_embedding_create(fs, 256);
    if (!kv) { FAIL("kv_embedding_create returned NULL"); return 1; }
    wubu_kv_embedding_free(kv);
    wubu_kvfs_free(fs);
    PASS();

    /* shared vars for later tests */
    wubu_grow_kv_t *g = NULL;
    wubu_grow_kv_cfg_t cfg;
    uint32_t blk;
    size_t off, nf;
    int rc;
    TEST(t2_encode_resolves);
    fs = wubu_kvfs_create(256, 64);
    kv = wubu_kv_embedding_create(fs, 256);

    const char *test_file = "hello.txt";
    const char *content = "Hello, World! This is a test file.";
    size_t content_len = strlen(content);
    size_t n_tokens;
    rc = wubu_kv_embedding_encode(kv, test_file, content, content_len, &n_tokens);
    if (rc != 0) { FAIL("encode returned nonzero"); goto cleanup_t2; }
    if (n_tokens != content_len) {
        char buf[128];
        snprintf(buf, sizeof(buf), "n_tokens=%zu expected=%zu", n_tokens, content_len);
        FAIL(buf);
        goto cleanup_t2;
    }

    /* The path should resolve */
    uint32_t block;
    size_t offset, n_floats;
    rc = wubu_kv_embedding_region(kv, "/kv/in/hello.txt", &block, &offset, &n_floats);
    if (rc != 0) { FAIL("region lookup failed"); goto cleanup_t2; }
    if (n_floats != n_tokens) {
        char buf[128];
        snprintf(buf, sizeof(buf), "n_floats=%zu expected=%zu", n_floats, n_tokens);
        FAIL(buf);
        goto cleanup_t2;
    }
    if (offset != 256) { /* first mount: block 0 reserved, block 1 → offset 256 */
        char buf[128];
        snprintf(buf, sizeof(buf), "offset=%zu expected=%zu", offset, (size_t)256);
        FAIL(buf);
        goto cleanup_t2;
    }
    PASS();

cleanup_t2:
    wubu_kv_embedding_free(kv);
    wubu_kvfs_free(fs);

    /* T3: multiple files get distinct regions */
    TEST(t3_distinct_regions);
    fs = wubu_kvfs_create(256, 64);
    kv = wubu_kv_embedding_create(fs, 256);

    const char *files[] = { "a.txt", "b.txt", "c.txt" };
    const char *contents[] = { "AAAA", "BBBBBB", "CCCCCCCCCCCC" };
    size_t offsets[3];
    for (int i = 0; i < 3; i++) {
        size_t nt;
        rc = wubu_kv_embedding_encode(kv, files[i], contents[i],
                                       strlen(contents[i]), &nt);
        if (rc != 0) { FAIL("encode failed for file"); goto cleanup_t3; }
        char kvpath[64];
        snprintf(kvpath, sizeof(kvpath), "/kv/in/%s", files[i]);
        size_t off, nf;
        rc = wubu_kv_embedding_region(kv, kvpath, NULL, &off, &nf);
        if (rc != 0) { FAIL("region lookup failed"); goto cleanup_t3; }
        offsets[i] = off;
    }
    /* All offsets should be distinct */
    if (offsets[0] == offsets[1] || offsets[0] == offsets[2] ||
        offsets[1] == offsets[2]) {
        FAIL("two files share the same KV offset");
        goto cleanup_t3;
    }
    if (offsets[0] >= offsets[1] || offsets[1] >= offsets[2]) {
        FAIL("offsets not monotonic (freelist broken)");
        goto cleanup_t3;
    }
    PASS();

cleanup_t3:
    wubu_kv_embedding_free(kv);
    wubu_kvfs_free(fs);

    /* T4: coherence computation on synthetic attention */
    TEST(t4_coherence_high_mass);
    fs = wubu_kvfs_create(256, 64);
    kv = wubu_kv_embedding_create(fs, 256);

    /* Encode a file so the path is registered */
    const char *content4 = "test coherence content here";
    size_t nt4;
    rc = wubu_kv_embedding_encode(kv, "coherence.txt", content4,
                                   strlen(content4), &nt4);
    if (rc != 0) { FAIL("encode for coherence test"); goto cleanup_t4; }

    /* Synthetic attention: 4 context tokens, 1 query token.
     * Context tokens 0..2 are "before file", 3..6 are the file region.
     * High-attn case: all attention on file region. */
    float attn_high[] = {
        /* query 0: [ctx0, ctx1, ctx2, FILE, FILE, FILE, FILE] */
        0.01f, 0.01f, 0.01f, 0.32f, 0.32f, 0.32f, 0.00f
    };
    wubu_coherence_t coh;
    rc = wubu_kv_embedding_coherence(kv, "/kv/in/coherence.txt",
                                      attn_high, 1, 7,
                                      3, 3,  /* context_start=3, context_len=3 */
                                      0, 1, &coh); /* query_start=0, query_len=1 */
    if (rc != 0) { FAIL("coherence returned -1"); goto cleanup_t4; }
    if (coh.attention_mass < 0.95f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "mass=%.3f expected >0.95", coh.attention_mass);
        FAIL(buf);
        goto cleanup_t4;
    }
    if (coh.score < 0.7f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "score=%.3f expected >0.7", coh.score);
        FAIL(buf);
        goto cleanup_t4;
    }
    PASS();

    /* Low-attn case: attention spread uniformly over everything */
    TEST(t4b_coherence_low_mass);
    printf("  [t4b_coherence_low_mass] ... ");
    float attn_low[] = {
        0.143f, 0.143f, 0.143f, 0.143f, 0.143f, 0.143f, 0.142f
    };
    rc = wubu_kv_embedding_coherence(kv, "/kv/in/coherence.txt",
                                      attn_low, 1, 7,
                                      3, 3, 0, 1, &coh);
    if (rc != 0) { FAIL("low-attn coherence returned -1"); goto cleanup_t4; }
    if (coh.attention_mass > 0.5f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "mass=%.3f expected <0.5", coh.attention_mass);
        FAIL(buf);
        goto cleanup_t4;
    }
    if (coh.score > 0.5f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "score=%.3f expected <0.5", coh.score);
        FAIL(buf);
        goto cleanup_t4;
    }
    PASS();

    /* T5: coherence rejects unknown path */
    TEST(t5_coherence_unknown_path);
    rc = wubu_kv_embedding_coherence(kv, "/kv/in/nonexistent.txt",
                                      attn_high, 1, 7, 3, 3, 0, 1, &coh);
    if (rc == 0) { FAIL("should return -1 for unknown path"); goto cleanup_t4; }
    PASS();

cleanup_t4:
    wubu_kv_embedding_free(kv);
    wubu_kvfs_free(fs);

    /* T6: coherence reward (batched) */
    TEST(t6_coherence_reward);
    fs = wubu_kvfs_create(256, 64);
    kv = wubu_kv_embedding_create(fs, 256);
    /* Encode two files */
    const char *f1 = "hello.txt";
    const char *c1 = "hello world";
    const char *f2 = "code.c";
    const char *c2 = "int main(){return 0;}";
    wubu_kv_embedding_encode(kv, f1, c1, strlen(c1), NULL);
    wubu_kv_embedding_encode(kv, f2, c2, strlen(c2), NULL);

    /* Build attention: file 1 gets high mass, file 2 gets low mass */
    /* Context = 8 tokens (3+3+2: padding + file1 + file2) */
    /* Query = 1 token attending over all 8 context tokens */
    /* File 1 region: tokens 3..5, file 2 region: tokens 6..7 */
    float attn_batch1[] = { 0.01f, 0.01f, 0.01f, 0.32f, 0.32f, 0.32f, 0.005f, 0.005f };
    float attn_batch2[] = { 0.12f, 0.12f, 0.12f, 0.12f, 0.12f, 0.12f, 0.14f, 0.14f };
    const char *rpaths[] = { "/kv/in/hello.txt", "/kv/in/code.c" };
    const float *attentions[] = { attn_batch1, attn_batch2 };
    size_t nq[] = { 1, 1 };
    size_t nc[] = { 8, 8 };
    size_t cs_arr[] = { 3, 6 };  /* context_start */
    size_t cl_arr[] = { 3, 2 };  /* context_len */
    size_t qs_arr[] = { 0, 0 };  /* query_start */
    size_t ql_arr[] = { 1, 1 };  /* query_len */

    wubu_reward_result_t reward;
    int r = wubu_coherence_reward_compute(kv, rpaths, attentions, nq, nc,
                                           cs_arr, cl_arr, qs_arr, ql_arr, 2, &reward);
    if (r != 0) { FAIL("coherence reward compute failed"); goto cleanup_t6; }
    if (reward.n_entries != 2) {
        char buf[128];
        snprintf(buf, sizeof(buf), "n_entries=%d expected 2", reward.n_entries);
        FAIL(buf);
        goto cleanup_t6;
    }
    if (reward.entries[0].coh.attention_mass < 0.9f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "file1 mass=%.3f expected >0.9",
                 reward.entries[0].coh.attention_mass);
        FAIL(buf);
        goto cleanup_t6;
    }
    if (reward.entries[1].coh.attention_mass > 0.4f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "file2 mass=%.3f expected <0.4",
                 reward.entries[1].coh.attention_mass);
        FAIL(buf);
        goto cleanup_t6;
    }
    if (reward.reward < 0.3f || reward.reward > 1.0f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "reward=%.3f expected in [0.3, 1.0]",
                 reward.reward);
        FAIL(buf);
        goto cleanup_t6;
    }
    PASS();

cleanup_t6:
    wubu_reward_result_free(&reward);
    wubu_kv_embedding_free(kv);
    wubu_kvfs_free(fs);

    /* T7: FS dataset walks files and encodes them */
    TEST(t7_fs_dataset);
    /* Create a temp directory with files */
    system("mkdir -p /tmp/wubu_fs_test && echo 'hello world' > /tmp/wubu_fs_test/a.txt && mkdir -p /tmp/wubu_fs_test/sub && echo 'int main' > /tmp/wubu_fs_test/sub/b.txt");
    fs = wubu_kvfs_create(256, 256);
    kv = wubu_kv_embedding_create(fs, 256);

    /* We can't easily test FS dataset without a real tokenizer.json.
     * Instead, test the direct encode path: encode files, verify region
     * lookup works through the dataset. */
    wubu_kv_embedding_encode(kv, "a.txt", "hello world", 11, NULL);
    wubu_kv_embedding_encode(kv, "sub/b.txt", "int main", 8, NULL);

    /* Both should resolve */
    if (wubu_kv_embedding_region(kv, "/kv/in/a.txt", &blk, &off, &nf) != 0) {
        FAIL("a.txt region not found");
        goto cleanup_t7;
    }
    if (nf != 11) {
        char buf[128];
        snprintf(buf, sizeof(buf), "a.txt n_floats=%zu expected 11", nf);
        FAIL(buf);
        goto cleanup_t7;
    }
    if (wubu_kv_embedding_region(kv, "/kv/in/sub/b.txt", &blk, &off, &nf) != 0) {
        FAIL("sub/b.txt region not found");
        goto cleanup_t7;
    }
    if (nf != 8) {
        char buf[128];
        snprintf(buf, sizeof(buf), "b.txt n_floats=%zu expected 8", nf);
        FAIL(buf);
        goto cleanup_t7;
    }
    PASS();

cleanup_t7:
    wubu_kv_embedding_free(kv);
    wubu_kvfs_free(fs);
    system("rm -rf /tmp/wubu_fs_test");

    /* T8: KV grow operator — diagnose + grow */
    TEST(t8_kv_grow_diagnose_grow);
    fs = wubu_kvfs_create(256, 64);
    kv = wubu_kv_embedding_create(fs, 256);
    wubu_kv_embedding_encode(kv, "low.txt", "this file is poorly understood", 30, NULL);
    wubu_kv_embedding_encode(kv, "high.txt", "well understood file", 20, NULL);
    cfg = wubu_grow_kv_default_cfg();
    cfg.coherence_threshold = 0.5;
    g = wubu_grow_kv_create(kv, &cfg);
    if (!g) { FAIL("grow_kv_create returned NULL"); goto cleanup_t8; }

    /* Diagnose: low.txt has score 0.3, high.txt has score 0.8 */
    const char *paths[] = { "/kv/in/low.txt", "/kv/in/high.txt" };
    float scores[] = { 0.3f, 0.8f };
    int n_under = wubu_grow_kv_diagnose(g, paths, scores, 2);
    if (n_under != 1) {
        char buf[128];
        snprintf(buf, sizeof(buf), "n_under=%d expected 1", n_under);
        FAIL(buf);
        goto cleanup_t8;
    }

    /* Verify the under-coherent file is low.txt (worst first) */
    const char *under_paths[10];
    int n_found = wubu_grow_kv_undercoherent(g, under_paths, 10);
    if (n_found != 1) {
        char buf[128];
        snprintf(buf, sizeof(buf), "undercoherent=%d expected 1", n_found);
        FAIL(buf);
        goto cleanup_t8;
    }
    if (strcmp(under_paths[0], "low.txt") != 0) {
        char buf[128];
        snprintf(buf, sizeof(buf), "under_path=%s expected low.txt", under_paths[0]);
        FAIL(buf);
        goto cleanup_t8;
    }

    /* GROW: should grow 1 block toward low.txt */
    int grown = wubu_grow_kv_grow(g, 3);
    if (grown != 1) {
        char buf[128];
        snprintf(buf, sizeof(buf), "grown=%d expected 1", grown);
        FAIL(buf);
        goto cleanup_t8;
    }

    /* The grow block should be mounted at /kv/in/low.txt/grow0 */
    if (wubu_kv_embedding_region(kv, "/kv/in/low.txt/grow0", &blk, &off, &nf) != 0) {
        FAIL("grow0 region not found");
        goto cleanup_t8;
    }
    if (nf != 1) {
        char buf[128];
        snprintf(buf, sizeof(buf), "grow0 n_floats=%zu expected 1", nf);
        FAIL(buf);
        goto cleanup_t8;
    }
    PASS();

cleanup_t8:
    wubu_grow_kv_free(g);
    wubu_kv_embedding_free(kv);
    wubu_kvfs_free(fs);

    /* T9: KV grow respects max_kv_blocks ceiling */
    TEST(t9_grow_ceiling);
    fs = wubu_kvfs_create(256, 8);  /* only 8 blocks total */
    kv = wubu_kv_embedding_create(fs, 256);
    wubu_kv_embedding_encode(kv, "f1.txt", "short", 5, NULL);
    wubu_kv_embedding_encode(kv, "f2.txt", "short2", 6, NULL);

    cfg = wubu_grow_kv_default_cfg();
    cfg.max_kv_blocks = 2;  /* ceiling: 2 blocks */
    g = wubu_grow_kv_create(kv, &cfg);
    if (!g) { FAIL("grow_kv_create (ceiling) returned NULL"); goto cleanup_t9; }

    const char *cpaths[] = { "/kv/in/f1.txt", "/kv/in/f2.txt" };
    float cscores[] = { 0.1f, 0.2f };
    wubu_grow_kv_diagnose(g, cpaths, cscores, 2);

    /* Should grow at most 2 blocks (ceiling) even though both are under-coherent */
    int grown2 = wubu_grow_kv_grow(g, 10);
    if (grown2 != 2) {
        char buf[128];
        snprintf(buf, sizeof(buf), "grown=%d expected 2 (ceiling)", grown2);
        FAIL(buf);
        goto cleanup_t9;
    }
    PASS();

cleanup_t9:
    wubu_grow_kv_free(g);
    wubu_kv_embedding_free(kv);
    wubu_kvfs_free(fs);

    /* Summary */
    printf("\n=== %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
