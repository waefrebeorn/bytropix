/* wubu_klog.c -- KERNEL #18 structured kernel log at /n/kernel/klog.
 * See wubu_klog.h. */
#include "wubu_klog.h"
#include <string.h>

int wubu_klog_init(wubu_klog_t *k) {
    if (!k) return -1;
    memset(k->entries, 0, sizeof k->entries);
    k->n = 0;
    k->head = 0;
    k->next_seq = 1;
    return 0;
}

uint64_t wubu_klog_write(wubu_klog_t *k, wubu_klog_level_t level,
                         const char *subsystem, const char *msg) {
    if (!k) return 0;
    wubu_klog_entry_t *e = &k->entries[k->head];
    e->seq = k->next_seq++;
    e->level = level;
    if (subsystem) { strncpy(e->subsystem, subsystem, WUBU_KLOG_SUBLEN-1); e->subsystem[WUBU_KLOG_SUBLEN-1]='\0'; }
    else e->subsystem[0] = '\0';
    if (msg) { strncpy(e->msg, msg, WUBU_KLOG_MSGLEN-1); e->msg[WUBU_KLOG_MSGLEN-1]='\0'; }
    else e->msg[0] = '\0';
    k->head = (k->head + 1) % WUBU_KLOG_CAP;
    if (k->n < WUBU_KLOG_CAP) k->n++;
    return e->seq;
}

int wubu_klog_query(const wubu_klog_t *k, wubu_klog_level_t min_level,
                    const char *subsystem,
                    const wubu_klog_entry_t **out, int max_out) {
    if (!k || !out || max_out <= 0) return 0;
    /* iterate oldest-to-newest; the ring is full when n==CAP, else entries are
     * in [0, head) in insertion order */
    int count = 0;
    int start = (k->n == WUBU_KLOG_CAP) ? k->head : 0;
    for (int i = 0; i < k->n; i++) {
        const wubu_klog_entry_t *e = &k->entries[(start + i) % WUBU_KLOG_CAP];
        if ((int)e->level < (int)min_level) continue;
        if (subsystem && strcmp(e->subsystem, subsystem) != 0) continue;
        if (count < max_out) out[count++] = e;
    }
    return count;
}
