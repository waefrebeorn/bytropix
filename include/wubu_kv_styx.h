#ifndef WUBU_KV_STYX_H
#define WUBU_KV_STYX_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kv_meta_slab kv_meta_slab_t;

int  wubu_kv_styx_init(void);
void wubu_kv_styx_shutdown(void);
int  wubu_kv_styx_register(const char *layer_path,
                           void *kv_ptr, size_t kv_bytes);
int  wubu_kv_styx_unregister(const char *layer_path);
const void *wubu_kv_styx_lookup(const char *layer_path, size_t *out_bytes);
int  wubu_kv_styx_registered_count(void);
char *wubu_kv_styx_snapshot_json(size_t *out_len);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_KV_STYX_H */
