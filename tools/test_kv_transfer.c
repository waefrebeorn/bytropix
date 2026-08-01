/* Test: D05 localhost KV transfer layer (doc 007/002).
 * Verifies: KV blocks written by a "prefill instance" are bit-identical when
 * read back by a "decode instance" via the transfer buffer. */
#include "wubu_kv_transfer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

int main(void) {
    const char *path = "/tmp/wubu_kv_xfer_test.bin";
    unlink(path);

    wubu_kv_transfer_t *prefill = wubu_kv_transfer_create(path, 1 << 20);
    wubu_kv_transfer_t *decode  = wubu_kv_transfer_create(path, 1 << 20);
    assert(prefill && decode);

    /* Prefill instance writes 4 KV blocks of 256 floats each */
    const int BLK = 256;
    float wbuf[BLK], rbuf[BLK];
    for (int slot = 0; slot < 4; slot++) {
        for (int i = 0; i < BLK; i++) wbuf[i] = (float)(slot * 1000 + i) * 0.01f;
        assert(wubu_kv_transfer_put(prefill, slot, wbuf, sizeof(wbuf)) == 0);
    }

    /* Decode instance reads them back */
    for (int slot = 0; slot < 4; slot++) {
        memset(rbuf, 0, sizeof(rbuf));
        assert(wubu_kv_transfer_get(decode, slot, rbuf, sizeof(rbuf)) == 0);
        for (int i = 0; i < BLK; i++) {
            float expect = (float)(slot * 1000 + i) * 0.01f;
            assert(rbuf[i] == expect);  /* bit-identical */
        }
    }

    printf("KV transfer: 4 blocks × %d floats round-trip bit-identical ✓\n", BLK);
    printf("KV transfer used bytes: %zu\n", wubu_kv_transfer_used(prefill));

    wubu_kv_transfer_free(prefill);
    wubu_kv_transfer_free(decode);
    printf("ALL KV-TRANSFER TESTS PASSED\n");
    return 0;
}
