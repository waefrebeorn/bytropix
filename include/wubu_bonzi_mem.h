#ifndef WUBU_BONZI_MEM_H
#define WUBU_BONZI_MEM_H
#include <stddef.h>
#include <stdint.h>

/* wubu_bonzi_mem.h -- UX #239 companion persistent memory.
 *
 * The companion's persistent memory: a key-value store that survives across
 * sessions (serialized to a byte buffer for persistence, restored exactly).
 * Remembers facts about the user, preferences, and conversation context.
 * Supports set/get/forget, a capacity bound (LRU-ish eviction of oldest when
 * full), and lossless serialize/restore (the "persistent" guarantee).
 * Deterministic, fixed-size, fail-closed. Real work (no stubs). */

#define WUBU_BONZI_MEM_MAX 64
#define WUBU_BONZI_KEYLEN 48
#define WUBU_BONZI_VALLEN 128

typedef struct {
    char key[WUBU_BONZI_KEYLEN];
    char val[WUBU_BONZI_VALLEN];
    uint64_t seq;   /* insertion order for eviction */
} wubu_bonzi_mem_entry_t;

typedef struct {
    wubu_bonzi_mem_entry_t entries[WUBU_BONZI_MEM_MAX];
    int n;
    uint64_t next_seq;
} wubu_bonzi_mem_t;

/* Initialise the memory store. */
int wubu_bonzi_mem_init(wubu_bonzi_mem_t *m);

/* Remember a fact (set/overwrite). Returns 0/-1. Evicts oldest if full. */
int wubu_bonzi_mem_set(wubu_bonzi_mem_t *m, const char *key, const char *val);

/* Recall a fact. Returns the value pointer, or NULL if not remembered. */
const char *wubu_bonzi_mem_get(const wubu_bonzi_mem_t *m, const char *key);

/* Forget a fact. Returns 0 if removed, -1 if not found. */
int wubu_bonzi_mem_forget(wubu_bonzi_mem_t *m, const char *key);

/* Number of facts remembered. */
int wubu_bonzi_mem_count(const wubu_bonzi_mem_t *m);

/* Serialize to a byte buffer (for persistence). Returns bytes or -1. */
int wubu_bonzi_mem_serialize(const wubu_bonzi_mem_t *m, unsigned char *out, size_t outsz);

/* Restore from a serialized buffer (lossless). Returns 0/-1. */
int wubu_bonzi_mem_restore(wubu_bonzi_mem_t *m, const unsigned char *buf, size_t bufsz);

#endif
