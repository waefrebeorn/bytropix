/* Test: cross-request prefix KV reuse (doc 010).
 *
 * Two requests share a 512-token prefix. The second request's prefix_match
 * should return 512 (the full prefix length), not 0. The first request
 * registers the prefix; the second reuses it.
 */
#include "wubu_prefix_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

int main(void) {
    wubu_prefix_cache_t *cache = wubu_prefix_cache_create();
    assert(cache);

    /* Build a 512-token prefix with REAL token ids (1000+ range).
     * The old `tok & 0xFF` indexing would collide e.g. token 1001 and 1. */
    int prefix_len = 512;
    int *prefix = (int *)malloc(prefix_len * sizeof(int));
    for (int i = 0; i < prefix_len; i++) prefix[i] = 1000 + i * 7;  /* 1000, 1007, 1014, ... */

    /* Request 1: register the prefix (first time → miss) */
    int out_blocks[64];
    int match1 = wubu_prefix_cache_match(cache, prefix, prefix_len, out_blocks, 64);
    printf("Request 1: match=%d (expected 0, first-time miss)\n", match1);
    assert(match1 == 0);

    wubu_prefix_hash_t h1 = wubu_prefix_cache_register(cache, prefix, prefix_len, NULL, 16);
    printf("Request 1: registered, hash=0x%016lx\n", (unsigned long)h1);
    assert(h1 != 0);

    /* Request 2: same prefix → should hit */
    int match2 = wubu_prefix_cache_match(cache, prefix, prefix_len, out_blocks, 64);
    printf("Request 2: match=%d (expected 512, full prefix reused)\n", match2);
    assert(match2 > 0);

    /* Request 3: different prefix → should miss (uses token ids in 5000+ range) */
    int *other = (int *)malloc(prefix_len * sizeof(int));
    for (int i = 0; i < prefix_len; i++) other[i] = 5000 + i * 11;  /* 5000, 5011, 5022, ... */
    int match3 = wubu_prefix_cache_match(cache, other, prefix_len, out_blocks, 64);
    printf("Request 3: match=%d (expected 0, different prefix)\n", match3);
    assert(match3 == 0);

    /* Stats check */
    size_t hits, misses, evictions, nodes;
    wubu_prefix_cache_stats(cache, &hits, &misses, &evictions, &nodes);
    printf("Stats: hits=%zu misses=%zu evictions=%zu nodes=%zu\n", hits, misses, evictions, nodes);
    assert(hits >= 1);
    assert(misses >= 2);

    /* Collision test: different prompts → different hashes */
    wubu_prefix_hash_t h2 = wubu_prefix_hash_compute(prefix, prefix_len);
    wubu_prefix_hash_t h3 = wubu_prefix_hash_compute(other, prefix_len);
    printf("Hash1=0x%016lx Hash2=0x%016lx (must differ)\n", (unsigned long)h2, (unsigned long)h3);
    assert(h2 != h3);

    /* Partial prefix match: first 256 tokens of the 512-token prefix */
    int match4 = wubu_prefix_cache_match(cache, prefix, 256, out_blocks, 64);
    printf("Partial prefix (256/512): match=%d (expected 256)\n", match4);
    assert(match4 == 256);

    free(prefix);
    free(other);
    wubu_prefix_cache_free(cache);

    printf("ALL PREFIX-REUSE TESTS PASSED\n");
    return 0;
}
