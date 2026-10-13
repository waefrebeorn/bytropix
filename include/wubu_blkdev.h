#ifndef WUBU_BLKDEV_H
#define WUBU_BLKDEV_H
#include <stdint.h>
#include <stddef.h>

/* wubu_blkdev.h -- KERNEL #26 block device driver + request queue.
 *
 * A blk-mq-style block device: requests (read/write of sectors) are queued and
 * completed. The driver owns a backing store (array of sectors); enqueue
 * submits a request; the scheduler completes requests in queue order (FIFO
 * here, extensible to elevator), performing the actual sector I/O. Tracks
 * queue depth and completion. Deterministic, real work (no stubs). */

#define WUBU_BLK_SECTORS 256
#define WUBU_BLK_SECTOR_SIZE 512
#define WUBU_BLK_QUEUE 64

typedef enum { WUBU_BLK_READ=0, WUBU_BLK_WRITE=1 } wubu_blk_op_t;

typedef struct {
    wubu_blk_op_t op;
    uint32_t sector;      /* starting sector */
    uint32_t nsectors;
    uint8_t *buf;         /* data buffer (read: filled, write: source) */
    int completed;
} wubu_blk_req_t;

typedef struct {
    uint8_t store[WUBU_BLK_SECTORS][WUBU_BLK_SECTOR_SIZE];
    wubu_blk_req_t queue[WUBU_BLK_QUEUE];
    int q_head, q_tail, q_len;   /* FIFO ring */
    uint64_t n_completed;
} wubu_blkdev_t;

/* Initialise. */
int wubu_blk_init(wubu_blkdev_t *b);

/* Current queue depth (number of pending requests). */
int wubu_blk_queue_depth(const wubu_blkdev_t *b);

/* Enqueue a request. Returns 0/-1 (queue full -> -1). */
int wubu_blk_enqueue(wubu_blkdev_t *b, wubu_blk_op_t op, uint32_t sector,
                     uint32_t nsectors, uint8_t *buf);

/* Complete one queued request (performs the actual I/O). Returns 1 if a
 * request was completed, 0 if the queue was empty, -1 on error. */
int wubu_blk_complete_one(wubu_blkdev_t *b);

/* Drain the whole queue (complete all pending requests). Returns count. */
int wubu_blk_drain(wubu_blkdev_t *b);

/* Total requests completed. */
uint64_t wubu_blk_completed(const wubu_blkdev_t *b);

#endif
