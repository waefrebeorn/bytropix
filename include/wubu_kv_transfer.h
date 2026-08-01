#ifndef WUBU_KV_TRANSFER_H
#define WUBU_KV_TRANSFER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Localhost KV transfer layer (D05): ship KV blocks from a prefill instance
 * to a decode instance on the same host, NIXL/UCX analog. Backed by an mmap'd
 * temp file (or raw file I/O fallback). */
typedef struct wubu_kv_transfer wubu_kv_transfer_t;

wubu_kv_transfer_t *wubu_kv_transfer_create(const char *shm_path, size_t capacity_bytes);
void wubu_kv_transfer_free(wubu_kv_transfer_t *t);

/* Write/read a fixed-size KV block at slot index (slot * len bytes). */
int wubu_kv_transfer_put(wubu_kv_transfer_t *t, size_t slot, const void *data, size_t len);
int wubu_kv_transfer_get(wubu_kv_transfer_t *t, size_t slot, void *out, size_t len);

size_t wubu_kv_transfer_used(const wubu_kv_transfer_t *t);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_KV_TRANSFER_H */
