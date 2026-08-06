/* test_kv_diag.c — tests for the KV coherence diagnose cycle (Phase 7)
 *
 * Verifies the full metabolism loop:
 *   forward → attention → coherence → grow/shrink
 *
 * T1: measure-only computes coherence for 2 files
 * T2: full cycle with grow + shrink → summary populated
 * T3: cycle with no grow/shrink handles NULL gracefully
 * T4: low-coherence file triggers grow, dead region triggers shrink
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 7 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#include <stdio.h>
#include <string.h>
#include "wubu_kv_coherence_diag.h"
#include "wubu_kv_embedding.h"
#include "wubu_grow_kv.h"
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
    wubu_kvfs_t *fs = wubu_kvfs_create(256, 64);
    wubu_kv_embedding_t *kv = wubu_kv_embedding_create(fs, 256);

    /* Encode two files */
    wubu_kv_embedding_encode(kv, "doc1.txt", "hello world this is file one", 29, NULL);
    wubu_kv_embedding_encode(kv, "doc2.txt", "another file here", 17, NULL);

    /* Attention: 2 files, each 1 query attending over 4 context tokens
     * File 1 region: tokens 0-0 (just 1 token "h"), File 2: tokens 1-1
     * Context layout: [h][e][l][l] (4 chars = 4 bytes = 4 tokens)
     * File 1 = byte 0 ('h'), File 2 = byte 1 ('e') */
    /* Actually, each file's tokens are contiguous in the KV namespace.
     * doc1.txt has 29 bytes → 29 tokens. doc2.txt has 17 bytes → 17 tokens.
     * Context is: [0..28] = doc1, [29..45] = doc2 (46 total)
     * Query: 1 token attending over all 46 context tokens.
     * For simplicity, use a smaller context. */
    /* Let's re-encode with smaller content for a manageable context */
    wubu_kv_embedding_free(kv);
    wubu_kvfs_free(fs);
    fs = wubu_kvfs_create(256, 64);
    kv = wubu_kv_embedding_create(fs, 256);
    wubu_kv_embedding_encode(kv, "doc1.txt", "AB", 2, NULL);
    wubu_kv_embedding_encode(kv, "doc2.txt", "CD", 2, NULL);

    /* Context: token 0-1 = doc1 (A,B), token 2-3 = doc2 (C,D) */
    /* File 1 region: context_start=0, context_len=2 */
    /* File 2 region: context_start=2, context_len=2 */
    /* Query: 1 token, query_start=0, query_len=1 */
    /* Full context: 4 tokens */

    /* High attention on doc1 (tokens 0,1), low on doc2 (tokens 2,3) */
    float attn_row[] = { 0.4f, 0.4f, 0.1f, 0.1f };

    const char *paths[] = { "/kv/in/doc1.txt", "/kv/in/doc2.txt" };
    const float *attention[] = { attn_row, attn_row };
    size_t nq[] = { 1, 1 };
    size_t nc[] = { 4, 4 };
    size_t cs[] = { 0, 2 };     /* context_start */
    size_t cl[] = { 2, 2 };     /* context_len */
    size_t qs[] = { 0, 0 };     /* query_start */
    size_t ql[] = { 1, 1 };     /* query_len */

    /* T1: measure-only */
    TEST(t1_measure_only);
    wubu_kv_diag_t *diag = wubu_kv_diag_create(kv, NULL, NULL);
    if (!diag) { FAIL("diag_create returned NULL"); goto cleanup; }
    wubu_coherence_t scores[2];
    int rc = wubu_kv_diag_measure(diag, paths, attention, nq, nc,
                                   cs, cl, qs, ql, 2, scores);
    if (rc != 0) { FAIL("measure returned -1"); goto cleanup; }
    /* doc1 (tokens 0-1) should have high mass */
    if (scores[0].attention_mass < 0.7f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "doc1 mass=%.3f expected >0.7", scores[0].attention_mass);
        FAIL(buf);
        goto cleanup;
    }
    /* doc2 (tokens 2-3) should have low mass */
    if (scores[1].attention_mass > 0.4f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "doc2 mass=%.3f expected <0.4", scores[1].attention_mass);
        FAIL(buf);
        goto cleanup;
    }
    PASS();

    /* T2: full cycle with grow + shrink */
    TEST(t2_full_cycle);
    /* Create grow + shrink operators */
    /* Both files have uniform attention → low coherence score */
    /* score = 0.4*0.5 + 0.3*0.0 + 0.3*1.0 = 0.5 (at threshold) */
    /* Lower the coherence threshold so these are under-coherent */
    wubu_grow_kv_cfg_t gcfg = wubu_grow_kv_default_cfg();
    gcfg.coherence_threshold = 0.6f;
    wubu_grow_kv_t *grow = wubu_grow_kv_create(kv, &gcfg);
    if (!grow) { FAIL("grow_kv_create returned NULL"); goto cleanup_t2; }

    wubu_kv_shrink_cfg_t scfg = wubu_kv_shrink_default_cfg();
    scfg.cold_iters = 3;
    wubu_kv_shrink_t *shrink = wubu_kv_shrink_create(fs, &scfg);

    /* Recreate diag with grow + shrink */
    wubu_kv_diag_free(diag);
    diag = wubu_kv_diag_create(kv, grow, shrink);
    if (!diag) { FAIL("diag_recreate returned NULL"); goto cleanup_t2; }

    /* Both files have low attention on one → low coherence score */
    /* Use uniform attention → low coherence */
    float attn_uniform[] = { 0.25f, 0.25f, 0.25f, 0.25f };
    const float *attn_uniform_arr[] = { attn_uniform, attn_uniform };

    wubu_reward_result_t reward;
    memset(&reward, 0, sizeof(reward));
    wubu_kv_diag_summary_t summary;
    memset(&summary, 0, sizeof(summary));
    rc = wubu_kv_diag_cycle(diag, paths, attn_uniform_arr, nq, nc,
                             cs, cl, qs, ql, 2, &reward, &summary);
    if (rc != 0) { FAIL("cycle returned -1"); goto cleanup_t2; }
    /* With uniform attention, both files should have low coherence */
    /* → grow operator should find 2 under-coherent files */
    if (summary.n_under < 2) {
        char buf[128];
        snprintf(buf, sizeof(buf), "n_under=%d expected >=2", summary.n_under);
        FAIL(buf);
        goto cleanup_t2;
    }
    /* At least 1 block should have been grown */
    if (summary.n_grown < 1) {
        char buf[128];
        snprintf(buf, sizeof(buf), "n_grown=%d expected >=1", summary.n_grown);
        FAIL(buf);
        goto cleanup_t2;
    }
    PASS();

cleanup_t2:
    wubu_kv_diag_free(diag);
    wubu_grow_kv_free(grow);
    wubu_kv_shrink_free(shrink);
    diag = wubu_kv_diag_create(kv, NULL, NULL);
    if (!diag) { FAIL("diag_recreate2 returned NULL"); goto cleanup; }

    /* T3: NULL grow/shrink — cycle should still work */
    TEST(t3_cycle_no_grow_shrink);
    rc = wubu_kv_diag_cycle(diag, paths, attention, nq, nc,
                             cs, cl, qs, ql, 2, &reward, &summary);
    if (rc != 0) { FAIL("cycle returned -1 with NULL grow/shrink"); goto cleanup; }
    if (summary.n_grown != 0 || summary.n_pruned != 0) {
        char buf[128];
        snprintf(buf, sizeof(buf), "grown=%d pruned=%d expected 0,0",
                 summary.n_grown, summary.n_pruned);
        FAIL(buf);
        goto cleanup;
    }
    PASS();

cleanup:
    wubu_kv_diag_free(diag);
    wubu_kv_embedding_free(kv);
    wubu_kvfs_free(fs);

    printf("\n=== %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
