#include "wubu_kv_tier.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    wubu_kv_tier_t *t = wubu_kv_tier_create(4, 1, 1, "/tmp/kv_warm.bin", "/tmp/kv_cold.bin");
    assert(t != NULL);

    size_t block_bytes = 16;
    wubu_kv_block_t *b = wubu_kv_tier_alloc_block(t, block_bytes);
    assert(b != NULL);
    assert(b->tier == WUBU_KV_TIER_HOT);

    /* Write then read back */
    uint8_t write_buf[16], read_buf[16];
    for (int i = 0; i < 16; i++) write_buf[i] = (uint8_t)i;
    assert(wubu_kv_tier_write_block(t, b, 0, write_buf, 16) == 0);
    assert(wubu_kv_tier_read_block(t, b, 0, read_buf, 16) == 0);
    assert(memcmp(write_buf, read_buf, 16) == 0);

    /* Stats */
    size_t hot, warm, cold;
    wubu_kv_tier_stats(t, &hot, &warm, &cold);
    assert(hot == 1);
    printf("ALL KV-TIER TESTS PASSED (hot=%zu warm=%zu cold=%zu)\n", hot, warm, cold);

    wubu_kv_tier_free(t);
    unlink("/tmp/kv_warm.bin");
    unlink("/tmp/kv_cold.bin");
    return 0;
}
