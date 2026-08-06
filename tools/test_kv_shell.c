/* test_kv_shell.c — tests for KV shell command routing (Phase 11)
 *
 * Verifies:
 *   T1: ls /kv/in/ lists encoded files
 *   T2: cat /kv/in/<file> returns decoded content
 *   T3: stat /kv/in/<file> returns KV metadata
 *   T4: unknown command returns error
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 11 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "wubu_kv_shell.h"
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

    /* Allocate a flat KV tensor: 64 blocks × 256 floats */
    float *kv_base = (float *)calloc(64 * 256, sizeof(float));
    if (!kv_base) { FAIL("calloc kv_base"); goto cleanup; }

    /* Encode two files — this mounts regions AND we write tokens */
    size_t n_tokens1;
    wubu_kv_embedding_encode(kv, "doc1.txt", "hello", 5, &n_tokens1);
    /* Write the token data into the KV tensor at the file's region */
    uint32_t blk1; size_t off1, nf1;
    wubu_kv_embedding_region(kv, "/kv/in/doc1.txt", &blk1, &off1, &nf1);
    /* Encode "hello" as byte tokens: BYTE_VOCAB_BASE + byte_value */
    float tokens1[5];
    for (int i = 0; i < 5; i++) tokens1[i] = (float)(16128 + (unsigned char)"hello"[i]);
    wubu_kvfs_write(fs, "/kv/in/doc1.txt", kv_base, tokens1, 5);

    size_t n_tokens2;
    wubu_kv_embedding_encode(kv, "doc2.txt", "world", 5, &n_tokens2);
    uint32_t blk2; size_t off2, nf2;
    wubu_kv_embedding_region(kv, "/kv/in/doc2.txt", &blk2, &off2, &nf2);
    float tokens2[5];
    for (int i = 0; i < 5; i++) tokens2[i] = (float)(16128 + (unsigned char)"world"[i]);
    wubu_kvfs_write(fs, "/kv/in/doc2.txt", kv_base, tokens2, 5);

    char out[256];
    wubu_kv_shell_t *shell = wubu_kv_shell_create(kv, NULL, kv_base);
    if (!shell) { FAIL("shell_create returned NULL"); goto cleanup; }

    /* T1: ls /kv/in/ should list both files */
    TEST(t1_ls_list_files);
    int rc = wubu_kv_shell_exec(shell, "ls", "/kv/in/", out, sizeof(out));
    if (rc != 0) { FAIL("ls returned error"); goto cleanup; }
    if (strstr(out, "doc1.txt") == NULL || strstr(out, "doc2.txt") == NULL) {
        char buf[128];
        snprintf(buf, sizeof(buf), "ls output missing files: '%s'", out);
        FAIL(buf);
        goto cleanup;
    }
    PASS();

    /* T2: cat /kv/in/doc1.txt should return "hello" */
    TEST(t2_cat_returns_content);
    rc = wubu_kv_shell_exec(shell, "cat", "/kv/in/doc1.txt", out, sizeof(out));
    if (rc != 0) {
        char buf[128];
        snprintf(buf, sizeof(buf), "cat returned error rc=%d", rc);
        FAIL(buf);
        goto cleanup;
    }
    if (strncmp(out, "hello", 5) != 0) {
        char buf[128];
        snprintf(buf, sizeof(buf), "cat output='%.10s' expected 'hello'", out);
        FAIL(buf);
        goto cleanup;
    }
    PASS();

    /* T3: stat /kv/in/doc1.txt should include path, block, floats */
    TEST(t3_stat_returns_metadata);
    rc = wubu_kv_shell_exec(shell, "stat", "/kv/in/doc1.txt", out, sizeof(out));
    if (rc != 0) { FAIL("stat returned error"); goto cleanup; }
    if (strstr(out, "/kv/in/doc1.txt") == NULL ||
        strstr(out, "start_block") == NULL ||
        strstr(out, "n_floats") == NULL ||
        strstr(out, "n_tokens") == NULL) {
        char buf[128];
        snprintf(buf, sizeof(buf), "stat output missing fields: '%s'", out);
        FAIL(buf);
        goto cleanup;
    }
    PASS();

    /* T4: unknown command returns error */
    TEST(t4_unknown_command);
    rc = wubu_kv_shell_exec(shell, "rm", "/kv/in/doc1.txt", out, sizeof(out));
    if (rc == 0) { FAIL("unknown command should return -1"); goto cleanup; }
    if (strstr(out, "unknown command") == NULL) {
        char buf[128];
        snprintf(buf, sizeof(buf), "output='%.40s' expected 'unknown command'", out);
        FAIL(buf);
        goto cleanup;
    }
    PASS();

    wubu_kv_shell_free(shell);

cleanup:
    if (kv_base) free(kv_base);
    wubu_kv_embedding_free(kv);
    wubu_kvfs_free(fs);

    printf("\n=== %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
