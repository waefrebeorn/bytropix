/*
 * wubu_vecsearch.h -- Vector substrate for AGI-OS (AV01-AV08).
 */
#ifndef WUBU_VECSEARCH_H
#define WUBU_VECSEARCH_H

#include <stdint.h>

#define WUBU_HNSW_MAX_NODES 4096
#define WUBU_HNSW_MAX_LEVEL 16
#define WUBU_RABITQ_MAX_BITS 256
#define WUBU_KVCACHE_MAX 1024
#define WUBU_FLASH_TILE 64
#define WUBU_ONDEVICE_MAX 16384
#define WUBU_AGENTIC_MEM_MAX 256

/* AV01: HNSW graph (navigable small-world, O(log N) ANN). */
typedef struct {
    int dim;
    int max_nodes;
    int n_nodes;
    int max_level;
    uint32_t rng_state;
    float *vectors;   /* [n_nodes x dim] */
    uint32_t *ids;
    int   *levels;
} wubu_hnsw_t;

int wubu_hnsw_insert(wubu_hnsw_t *h, const float *vec, uint32_t id);

/* AV02: RaBitQ quantization (1-bit/dim + correction, O(1/sqrt(D)) error). */
int wubu_rabitsq_quantize(const float *vec, int dim, const unsigned char *key,
                                int klen, unsigned char *out_bits, int n_bits);
int wubu_rabitsq_estimate(const unsigned char *bits, const float *vec, int dim,
                                const unsigned char *key, int klen, int n_bits,
                                float *distance);

/* AV03/AV04: KV cache with vector-similarity eviction (cosine-based). */
typedef struct {
    float *keys;       /* [max_entries x key_dim] */
    float *vals;       /* [max_entries x val_dim] */
    uint64_t *positions;
    uint64_t *last_access;
    float  *scores;
    int key_dim;
    int val_dim;
    int max_entries;
    int n_entries;
} wubu_kvcache_t;

int wubu_kvcache_insert(wubu_kvcache_t *c, const float *key, const float *val,
                            int key_dim, int val_dim, uint64_t token_pos);
void wubu_kvcache_score_similarity(wubu_kvcache_t *c, const float *query,
                                         int key_dim, float *out_scores);
int wubu_kvcache_evict_by_similarity(wubu_kvcache_t *c, const float *query,
                                           int key_dim, int target_entries);

/* AV05: FlashAttention-style tiling (never materialize full NxN). */
void wubu_flash_attn_tile(const float *Q, const float *K, const float *V,
                                int N, int d, int tile_size, float *out);

/* AV06: MRL (Matryoshka) flexible-dim truncation + renormalize. */
void wubu_mrl_truncate(const float *emb, int full_dim, int trunc_dim, float *out);

/* AV07: On-device vector DB (embedded, offline, pure C). */
typedef struct {
    float  *vectors;
    uint64_t *ids;
    int dim;
    int max_vecs;
    int n_vecs;
} wubu_ondevice_db_t;

int  wubu_on_device_db_init(wubu_ondevice_db_t *db, int dim, int max_vecs);
void wubu_on_device_db_free(wubu_ondevice_db_t *db);
int  wubu_on_device_db_add(wubu_ondevice_db_t *db, uint64_t id, const float *vec);
int  wubu_on_device_db_search(wubu_ondevice_db_t *db, const float *query,
                                   int top_k, uint64_t *out_ids, float *out_dists);

/* AV08: Agentic vector memory (observe->embed->store->retrieve->decide->act). */
typedef struct {
    float  *embeddings;
    char   **actions;
    uint64_t *timestamps;
    double  *rewards;
    int dim;
    int max;
    int n;
} wubu_agentic_mem_t;

int  wubu_agentic_mem_store(wubu_agentic_mem_t *m, const float *emb, int dim,
                                 const char *action, uint64_t timestamp);
int  wubu_agentic_mem_retrieve(wubu_agentic_mem_t *m, const float *query, int dim,
                                      int top_k, uint64_t *out_ids, float *out_scores);

#endif
