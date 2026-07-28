#ifndef WUBU_PAGED_KV_H
#define WUBU_PAGED_KV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_PAGED_MAX_SEQ 256
#define WUBU_PAGED_MAX_PAGES 1024

typedef struct {
    int pages[WUBU_PAGED_MAX_PAGES];
    int n_pages;
    int n_tokens;
} wubu_kv_seq_t;

typedef struct wubu_paged_kv wubu_paged_kv_t;

wubu_paged_kv_t *wubu_paged_kv_create(int block_size, int n_blocks,
                                      int head_dim, int n_kv_heads);
void wubu_paged_kv_free(wubu_paged_kv_t *m);

int  wubu_paged_kv_new_seq(wubu_paged_kv_t *m);
/* Returns physical block id, or -1 if full (preempt needed). */
int  wubu_paged_kv_ensure(wubu_paged_kv_t *m, int seq, int token_pos);
int  wubu_paged_kv_block_of(wubu_paged_kv_t *m, int seq, int token_pos);
void wubu_paged_kv_free_seq(wubu_paged_kv_t *m, int seq);
int  wubu_paged_kv_free_count(wubu_paged_kv_t *m);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_PAGED_KV_H */
