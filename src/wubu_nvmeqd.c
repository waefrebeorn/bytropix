/* wubu_nvmeqd.c -- KERNEL #27 NVMe queue depth tunable via /n.
 * See wubu_nvmeqd.h for the contract. Real, deterministic implementation. */

#include "wubu_nvmeqd.h"
#include <string.h>

int wubu_qd_init(wubu_nvmeqd_t *q)
{
    if (!q) return -1;
    memset(q, 0, sizeof(*q));
    return 0;
}

int wubu_qd_add(wubu_nvmeqd_t *q, const char *name, int min_depth,
                int max_depth, int init_depth)
{
    if (!q || !name) return -1;
    if (min_depth < 1 || max_depth < min_depth) return -1;
    if (init_depth < min_depth || init_depth > max_depth) return -1;
    if (wubu_qd_find(q, name) >= 0) return -1;   /* duplicate name */
    if (q->n >= WUBU_QD_MAX_DEVS) return -1;
    wubu_qd_dev_t *d = &q->devs[q->n++];
    memset(d, 0, sizeof(*d));
    strncpy(d->name, name, WUBU_QD_NAME_MAX - 1);
    d->name[WUBU_QD_NAME_MAX - 1] = '\0';
    d->present = 1;
    d->min_depth = min_depth;
    d->max_depth = max_depth;
    d->io_depth = init_depth;
    d->in_flight = 0;
    d->sq_id = 1;                /* admin queue (SQ0/CQ0) reserved */
    d->cq_id = 1;
    d->version = 1;
    return 0;
}

int wubu_qd_find(const wubu_nvmeqd_t *q, const char *name)
{
    if (!q || !name) return -1;
    for (int i = 0; i < q->n; i++)
        if (q->devs[i].present && strcmp(q->devs[i].name, name) == 0)
            return i;
    return -1;
}

int wubu_qd_get(const wubu_nvmeqd_t *q, const char *name)
{
    int i = wubu_qd_find(q, name);
    return i < 0 ? -1 : q->devs[i].io_depth;
}

int wubu_qd_set(wubu_nvmeqd_t *q, const char *name, int depth)
{
    if (!q) return -1;
    int i = wubu_qd_find(q, name);
    if (i < 0) return -1;
    wubu_qd_dev_t *d = &q->devs[i];
    if (depth < d->min_depth || depth > d->max_depth) return -1;
    if (d->in_flight != 0) return -1;   /* must be idle to resize the ring */
    if (depth != d->io_depth) {
        d->io_depth = depth;
        d->version++;
    }
    return 0;
}

int wubu_qd_inflight(wubu_nvmeqd_t *q, const char *name, int delta)
{
    if (!q) return -1;
    int i = wubu_qd_find(q, name);
    if (i < 0) return -1;
    wubu_qd_dev_t *d = &q->devs[i];
    int v = d->in_flight + delta;
    if (v < 0) v = 0;
    if (v > d->io_depth) v = d->io_depth;   /* cannot exceed queue depth */
    d->in_flight = v;
    return v;
}

int wubu_qd_coherent(const wubu_qd_dev_t *d)
{
    if (!d || !d->present) return 0;
    if (d->sq_id != d->cq_id) return 0;            /* I/O pair must match */
    if (d->io_depth < d->min_depth || d->io_depth > d->max_depth) return 0;
    if (d->in_flight < 0 || d->in_flight > d->io_depth) return 0;
    return 1;
}

unsigned long long wubu_qd_version(const wubu_nvmeqd_t *q, const char *name)
{
    int i = wubu_qd_find(q, name);
    return i < 0 ? 0ULL : q->devs[i].version;
}
