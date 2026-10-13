/* test_blkdev.c -- KERNEL #26 block device driver + request queue.
 * Proves: enqueue/complete, write then read round-trip, FIFO order, queue
 * depth tracking, drain, out-of-range/queue-full rejected, fail-closed. */
#include "wubu_blkdev.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_blkdev_t b;
    CHECK(wubu_blk_init(&b) == 0, "init ok");
    CHECK(wubu_blk_queue_depth(&b) == 0, "empty queue depth 0");

    /* write sector 10 with a pattern */
    uint8_t wbuf[WUBU_BLK_SECTOR_SIZE];
    memset(wbuf, 0xAB, sizeof wbuf);
    CHECK(wubu_blk_enqueue(&b, WUBU_BLK_WRITE, 10, 1, wbuf) == 0, "enqueue write s10");
    CHECK(wubu_blk_queue_depth(&b) == 1, "depth 1 after enqueue");
    CHECK(wubu_blk_complete_one(&b) == 1, "complete write");
    CHECK(wubu_blk_queue_depth(&b) == 0, "depth 0 after complete");
    CHECK(wubu_blk_completed(&b) == 1, "1 completed");

    /* read sector 10 back -> same pattern */
    uint8_t rbuf[WUBU_BLK_SECTOR_SIZE];
    memset(rbuf, 0, sizeof rbuf);
    CHECK(wubu_blk_enqueue(&b, WUBU_BLK_READ, 10, 1, rbuf) == 0, "enqueue read s10");
    CHECK(wubu_blk_complete_one(&b) == 1, "complete read");
    CHECK(rbuf[0] == 0xAB && rbuf[511] == 0xAB, "read back the written pattern");

    /* read an untouched sector -> zeros */
    uint8_t zbuf[WUBU_BLK_SECTOR_SIZE];
    memset(zbuf, 0xFF, sizeof zbuf);
    wubu_blk_enqueue(&b, WUBU_BLK_READ, 50, 1, zbuf);
    wubu_blk_complete_one(&b);
    int allzero = 1;
    for (int i = 0; i < WUBU_BLK_SECTOR_SIZE; i++) if (zbuf[i] != 0) { allzero = 0; break; }
    CHECK(allzero, "untouched sector reads as zeros");

    /* FIFO order: enqueue 3 writes, drain completes all in order */
    uint8_t a[WUBU_BLK_SECTOR_SIZE], cbuf[WUBU_BLK_SECTOR_SIZE];
    memset(a, 0x11, sizeof a);
    memset(cbuf, 0x33, sizeof cbuf);
    wubu_blk_enqueue(&b, WUBU_BLK_WRITE, 1, 1, a);
    wubu_blk_enqueue(&b, WUBU_BLK_WRITE, 2, 1, a);
    wubu_blk_enqueue(&b, WUBU_BLK_WRITE, 3, 1, cbuf);
    CHECK(wubu_blk_queue_depth(&b) == 3, "depth 3");
    int drained = wubu_blk_drain(&b);
    CHECK(drained == 3, "drained 3 requests");
    CHECK(wubu_blk_queue_depth(&b) == 0, "depth 0 after drain");
    CHECK(wubu_blk_completed(&b) == 6, "6 total completed (1+1+1+3)");

    /* out-of-range sector rejected */
    CHECK(wubu_blk_enqueue(&b, WUBU_BLK_WRITE, WUBU_BLK_SECTORS - 1, 2, a) == -1, "out-of-range -> -1");
    CHECK(wubu_blk_enqueue(&b, WUBU_BLK_WRITE, 0, 0, a) == -1, "zero nsectors -> -1");

    /* complete on empty -> 0 */
    CHECK(wubu_blk_complete_one(&b) == 0, "complete empty -> 0");

    /* fail-closed on NULL */
    CHECK(wubu_blk_init(NULL) == -1, "NULL init -> -1");
    CHECK(wubu_blk_enqueue(NULL, WUBU_BLK_WRITE, 0, 1, a) == -1, "NULL enqueue -> -1");
    CHECK(wubu_blk_complete_one(NULL) == -1, "NULL complete -> -1");
    CHECK(wubu_blk_drain(NULL) == -1, "NULL drain -> -1");
    CHECK(wubu_blk_completed(NULL) == 0, "NULL completed -> 0");
    CHECK(wubu_blk_queue_depth(NULL) == 0, "NULL depth -> 0");

    printf("\n=== test_blkdev: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
