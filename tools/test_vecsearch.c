/*
 * test_vecsearch.c -- AV01-AV08 vector substrate verification.
 */
#include "wubu_vecsearch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_vecsearch (AV01-AV08) ===\n");

    /* AV01: HNSW insert + basic structure */
    wubu_hnsw_t hnsw;
    hnsw.dim = 4; hnsw.max_nodes = 16; hnsw.n_nodes = 0;
    hnsw.max_level = 4; hnsw.rng_state = 1;
    hnsw.vectors = calloc(16 * 4, sizeof(float));
    hnsw.ids = calloc(16, sizeof(uint32_t));
    hnsw.levels = calloc(16, sizeof(int));
    float v0[] = {1,0,0,0}, v1[] = {0,1,0,0}, v2[] = {0,0,1,0};
    CHECK(wubu_hnsw_insert(&hnsw, v0, 100) == 0, "HNSW insert node 0");
    CHECK(wubu_hnsw_insert(&hnsw, v1, 101) == 0, "HNSW insert node 1");
    CHECK(wubu_hnsw_insert(&hnsw, v2, 102) == 0, "HNSW insert node 2");
    CHECK(hnsw.n_nodes == 3, "HNSW 3 nodes inserted");
    free(hnsw.vectors); free(hnsw.ids); free(hnsw.levels);

    /* AV02: RaBitQ quantize + estimate */
    float vec_q[] = {0.8f, -0.3f, 0.5f, 0.1f};
    unsigned char key_q[] = {0xAB, 0xCD};
    unsigned char bits[4] = {0};
    CHECK(wubu_rabitsq_quantize(vec_q, 4, key_q, 2, bits, 4) == 0, "RaBitQ quantize");
    float dist = 0;
    CHECK(wubu_rabitsq_estimate(bits, vec_q, 4, key_q, 2, 4, &dist) == 0, "RaBitQ estimate");
    CHECK(dist >= 0, "RaBitQ distance non-negative");

    /* AV03/AV04: KV cache insert + similarity eviction */
    wubu_kvcache_t kvc;
    kvc.keys = calloc(8, 4 * sizeof(float));
    kvc.vals = calloc(8, 4 * sizeof(float));
    kvc.positions = calloc(8, sizeof(uint64_t));
    kvc.last_access = calloc(8, sizeof(uint64_t));
    kvc.scores = calloc(8, sizeof(float));
    kvc.key_dim = 4; kvc.val_dim = 4; kvc.max_entries = 8; kvc.n_entries = 0;
    float kA[] = {1,0,0,0}, vA[] = {0,1,0,0};
    float kB[] = {0,1,0,0}, vB[] = {0,0,1,0};
    CHECK(wubu_kvcache_insert(&kvc, kA, vA, 4, 4, 0) == 0, "KV insert A");
    CHECK(wubu_kvcache_insert(&kvc, kB, vB, 4, 4, 1) == 0, "KV insert B");
    CHECK(kvc.n_entries == 2, "KV 2 entries");
    float query[] = {1,0,0,0};
    float scores[8];
    wubu_kvcache_score_similarity(&kvc, query, 4, scores);
    CHECK(scores[0] > scores[1], "KV similarity: A closer to query than B");
    CHECK(wubu_kvcache_evict_by_similarity(&kvc, query, 4, 1) == 0, "KV evict to 1");
    CHECK(kvc.n_entries == 1, "KV 1 entry after eviction");
    free(kvc.keys); free(kvc.vals); free(kvc.positions);
    free(kvc.last_access); free(kvc.scores);

    /* AV05: FlashAttention tiling (basic smoke) */
    float Q[4] = {1,0,0,0}, K[4] = {0,1,0,0}, V[4] = {0,0,1,0};
    float out[4] = {0};
    wubu_flash_attn_tile(Q, K, V, 1, 4, 1, out);
    CHECK(out[2] > 0, "FlashAttention tile produces non-zero output");

    /* AV06: MRL truncation */
    float full_emb[] = {0.6f, 0.8f, 0.3f, 0.4f};
    float trunc[2];
    wubu_mrl_truncate(full_emb, 4, 2, trunc);
    CHECK(fabsf(trunc[0] - 0.6f) < 0.01f, "MRL trunc preserves first dims");
    CHECK(fabsf(trunc[1] - 0.8f) < 0.01f, "MRL trunc preserves second dim");
    float nrm = sqrtf(trunc[0]*trunc[0] + trunc[1]*trunc[1]);
    CHECK(fabsf(nrm - 1.0f) < 0.01f, "MRL trunc renormalized");

    /* AV07: On-device vector DB */
    wubu_ondevice_db_t db;
    CHECK(wubu_on_device_db_init(&db, 4, 8) == 0, "on-device DB init");
    CHECK(wubu_on_device_db_add(&db, 100, v0) == 0, "on-device DB add v0");
    CHECK(wubu_on_device_db_add(&db, 101, v1) == 0, "on-device DB add v1");
    uint64_t ids[4]; float dists[4];
    int found = wubu_on_device_db_search(&db, query, 2, ids, dists);
    CHECK(found == 2, "on-device DB search found 2");
    CHECK(ids[0] == 100, "on-device DB top result is v0 (most similar)");
    wubu_on_device_db_free(&db);

    /* AV08: Agentic vector memory */
    wubu_agentic_mem_t am;
    am.embeddings = calloc(8, 4 * sizeof(float));
    am.actions = calloc(8, sizeof(char *));
    for (int i = 0; i < 8; i++) am.actions[i] = calloc(32, 1);
    am.timestamps = calloc(8, sizeof(uint64_t));
    am.rewards = calloc(8, sizeof(double));
    am.dim = 4; am.max = 8; am.n = 0;
    CHECK(wubu_agentic_mem_store(&am, v0, 4, "explore", 100) == 0, "agentic store");
    CHECK(wubu_agentic_mem_store(&am, v1, 4, "retrieve", 200) == 0, "agentic store 2");
    CHECK(am.n == 2, "agentic 2 stored");
    uint64_t a_ids[4]; float a_scores[4];
    int a_found = wubu_agentic_mem_retrieve(&am, query, 4, 2, a_ids, a_scores);
    CHECK(a_found == 2, "agentic retrieve 2");
    CHECK(a_scores[0] > a_scores[1], "agentic top result most similar");
    for (int i = 0; i < 8; i++) free(am.actions[i]);
    free(am.embeddings); free(am.actions); free(am.timestamps); free(am.rewards);

    if (failures == 0) { printf("ALL VECSEARCH TESTS PASSED\n"); return 0; }
    printf("%d VECSEARCH TEST(S) FAILED\n", failures);
    return 1;
}
