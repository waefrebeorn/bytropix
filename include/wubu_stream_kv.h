/*
 * wubu_stream_kv.h -- StreamingLLM attention-sink KV remapping (L01).
 * Opaque struct; access only via the functions below.
 */
#ifndef WUBU_STREAM_KV_H
#define WUBU_STREAM_KV_H

#include <stddef.h>

typedef struct wubu_stream_kv wubu_stream_kv_t;

/* Create a stream-KV remapper. sink = leading tokens kept, window = recent
 * tokens kept. sink==window==0 => disabled (identity remap, no behaviour
 * change to the existing full 512K cache path). */
wubu_stream_kv_t *wubu_stream_kv_create(int sink, int window);
void wubu_stream_kv_destroy(wubu_stream_kv_t *s);
void wubu_stream_kv_set(wubu_stream_kv_t *s, int sink, int window);

int  wubu_stream_kv_enabled(const wubu_stream_kv_t *s);
int  wubu_stream_kv_sink(const wubu_stream_kv_t *s);
int  wubu_stream_kv_window(const wubu_stream_kv_t *s);

/* Physical KV slots needed for a sequence of `len` tokens. */
int  wubu_stream_kv_capacity(const wubu_stream_kv_t *s, int len);

/* Remap logical position -> physical slot, or -1 if evicted. Identity when
 * disabled. */
int  wubu_stream_kv_remap(const wubu_stream_kv_t *s, int len, int pos);

/* Count of live (non-evicted) tokens for a sequence of `len`. */
int  wubu_stream_kv_live_count(const wubu_stream_kv_t *s, int len);

#endif /* WUBU_STREAM_KV_H */
