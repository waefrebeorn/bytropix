/* wubu_blkdev.c -- KERNEL #26 block device driver + request queue.
 * See wubu_blkdev.h. */
#include "wubu_blkdev.h"
#include <string.h>

int wubu_blk_init(wubu_blkdev_t *b) {
    if (!b) return -1;
    memset(b->store, 0, sizeof b->store);
    memset(b->queue, 0, sizeof b->queue);
    b->q_head = b->q_tail = b->q_len = 0;
    b->n_completed = 0;
    return 0;
}

int wubu_blk_queue_depth(const wubu_blkdev_t *b) {
    return b ? b->q_len : 0;
}

int wubu_blk_enqueue(wubu_blkdev_t *b, wubu_blk_op_t op, uint32_t sector,
                     uint32_t nsectors, uint8_t *buf) {
    if (!b || !buf || nsectors == 0) return -1;
    if (sector + nsectors > WUBU_BLK_SECTORS) return -1;   /* out of range */
    if (b->q_len >= WUBU_BLK_QUEUE) return -1;             /* queue full */
    wubu_blk_req_t *r = &b->queue[b->q_tail];
    r->op = op; r->sector = sector; r->nsectors = nsectors; r->buf = buf; r->completed = 0;
    b->q_tail = (b->q_tail + 1) % WUBU_BLK_QUEUE;
    b->q_len++;
    return 0;
}

int wubu_blk_complete_one(wubu_blkdev_t *b) {
    if (!b) return -1;   /* fail-closed */
    if (b->q_len == 0) return 0;   /* empty */
    wubu_blk_req_t *r = &b->queue[b->q_head];
    size_t nbytes = (size_t)r->nsectors * WUBU_BLK_SECTOR_SIZE;
    if (r->op == WUBU_BLK_READ) {
        memcpy(r->buf, b->store[r->sector], nbytes);
    } else { /* WRITE */
        memcpy(b->store[r->sector], r->buf, nbytes);
    }
    r->completed = 1;
    b->q_head = (b->q_head + 1) % WUBU_BLK_QUEUE;
    b->q_len--;
    b->n_completed++;
    return 1;
}

int wubu_blk_drain(wubu_blkdev_t *b) {
    if (!b) return -1;
    int n = 0;
    while (b->q_len > 0) { if (wubu_blk_complete_one(b) != 1) break; n++; }
    return n;
}

uint64_t wubu_blk_completed(const wubu_blkdev_t *b) {
    return b ? b->n_completed : 0;
}
