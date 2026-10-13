#ifndef WUBU_KLOG_H
#define WUBU_KLOG_H
#include <stdint.h>

/* wubu_klog.h -- KERNEL #18 structured kernel log at /n/kernel/klog.
 *
 * A structured kernel log: each entry has a severity, subsystem, and message.
 * Exposed as a queryable 9P file (/n/kernel/klog) that can be READ with a
 * severity filter and subsystem filter -- only entries at/above the severity
 * threshold (and matching the subsystem, if given) are emitted. Ring-buffered
 * (fixed capacity, oldest dropped). Deterministic, real work (no stubs). */

typedef enum {
    WUBU_KLOG_DEBUG=0, WUBU_KLOG_INFO, WUBU_KLOG_WARN, WUBU_KLOG_ERROR, WUBU_KLOG_FATAL,
} wubu_klog_level_t;

#define WUBU_KLOG_CAP 256
#define WUBU_KLOG_MSGLEN 96
#define WUBU_KLOG_SUBLEN 24

typedef struct {
    uint64_t seq;
    wubu_klog_level_t level;
    char subsystem[WUBU_KLOG_SUBLEN];
    char msg[WUBU_KLOG_MSGLEN];
} wubu_klog_entry_t;

typedef struct {
    wubu_klog_entry_t entries[WUBU_KLOG_CAP];
    int n;                 /* number of valid entries */
    int head;              /* ring write position */
    uint64_t next_seq;
} wubu_klog_t;

/* Initialise. */
int wubu_klog_init(wubu_klog_t *k);

/* Append an entry (drops oldest when full). Returns the seq assigned, or 0. */
uint64_t wubu_klog_write(wubu_klog_t *k, wubu_klog_level_t level,
                         const char *subsystem, const char *msg);

/* Query: emit entries at/above min_level, optionally filtered by subsystem
 * (NULL = all). Returns the count emitted; fills an array of pointers (up to
 * max_out). This is what reading /n/kernel/klog returns. */
int wubu_klog_query(const wubu_klog_t *k, wubu_klog_level_t min_level,
                    const char *subsystem,
                    const wubu_klog_entry_t **out, int max_out);

#endif
