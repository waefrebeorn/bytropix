/* wubu_snapshot.c -- OMARCHY #273/#274 btrfs subvolume snapshot + rollback.
 * See wubu_snapshot.h. Fixed-size ring, retention pruning, fail-closed. */
#include "wubu_snapshot.h"
#include <string.h>
#include <stdio.h>

int wubu_snap_init(wubu_snapmgr_t *m, const char *subvol, int retention) {
    if (!m) return -1;
    memset(m->snaps, 0, sizeof m->snaps);
    m->n = 0;
    m->head = 0;
    m->next_id = 1;
    m->active_idx = -1;
    m->retention = retention < 0 ? 0 : retention;
    m->last_timeline_ns = 0;
    if (subvol) { strncpy(m->subvol, subvol, WUBU_SNAP_NAMELEN - 1); m->subvol[WUBU_SNAP_NAMELEN-1]='\0'; }
    else m->subvol[0] = '\0';
    return 0;
}

static const char *type_name(wubu_snap_type_t t) {
    switch (t) {
        case WUBU_SNAP_MANUAL:   return "manual";
        case WUBU_SNAP_PRE:      return "pre";
        case WUBU_SNAP_POST:     return "post";
        case WUBU_SNAP_TIMELINE: return "timeline";
        default:                 return "?";
    }
}

static void prune(wubu_snapmgr_t *m) {
    if (m->retention <= 0) return;
    while (m->n > m->retention) {
        /* oldest slot in ring order */
        int oldest = (m->head - m->n + 2*WUBU_SNAP_MAX) % WUBU_SNAP_MAX;
        if (oldest == m->active_idx) {
            /* never drop the active snapshot; stop pruning */
            break;
        }
        /* compact: shift everything after oldest down by one */
        for (int i = oldest; i != m->head; i = (i + 1) % WUBU_SNAP_MAX) {
            int nxt = (i + 1) % WUBU_SNAP_MAX;
            m->snaps[i] = m->snaps[nxt];
        }
        m->head = (m->head - 1 + WUBU_SNAP_MAX) % WUBU_SNAP_MAX;
        m->n--;
        if (m->active_idx > oldest) m->active_idx--;
    }
}

uint64_t wubu_snap_create(wubu_snapmgr_t *m, const char *name, wubu_snap_type_t type, uint64_t now_ns) {
    if (!m) return 0;
    uint64_t id = m->next_id++;
    wubu_snapshot_t *s = &m->snaps[m->head];
    memset(s, 0, sizeof *s);
    s->id = id;
    s->timestamp_ns = now_ns;
    s->type = type;
    s->active = 0;
    if (name && name[0]) {
        strncpy(s->name, name, WUBU_SNAP_NAMELEN - 1);
    } else {
        snprintf(s->name, WUBU_SNAP_NAMELEN, "%.16s-%s-%llu",
                 m->subvol[0] ? m->subvol : "subvol", type_name(type),
                 (unsigned long long)id);
    }
    s->name[WUBU_SNAP_NAMELEN - 1] = '\0';
    /* first snapshot becomes active by default */
    if (m->active_idx < 0) { s->active = 1; m->active_idx = m->head; }
    m->head = (m->head + 1) % WUBU_SNAP_MAX;
    if (m->n < WUBU_SNAP_MAX) m->n++;
    prune(m);
    return id;
}

int wubu_snap_rollback(wubu_snapmgr_t *m, uint64_t id) {
    if (!m) return -1;
    const wubu_snapshot_t *t = wubu_snap_find(m, id);
    if (!t) return -1;
    /* clear old active, set new */
    for (int i = 0; i < m->n; i++) {
        int idx = (m->head - m->n + i + 2*WUBU_SNAP_MAX) % WUBU_SNAP_MAX;
        m->snaps[idx].active = 0;
    }
    int target = (int)(t - m->snaps);
    m->snaps[target].active = 1;
    m->active_idx = target;
    return 0;
}

const wubu_snapshot_t *wubu_snap_find(const wubu_snapmgr_t *m, uint64_t id) {
    if (!m) return NULL;
    for (int i = 0; i < m->n; i++) {
        int idx = (m->head - m->n + i + 2*WUBU_SNAP_MAX) % WUBU_SNAP_MAX;
        if (m->snaps[idx].id == id) return &m->snaps[idx];
    }
    return NULL;
}

const wubu_snapshot_t *wubu_snap_active(const wubu_snapmgr_t *m) {
    if (!m || m->active_idx < 0) return NULL;
    return &m->snaps[m->active_idx];
}

int wubu_snap_count(const wubu_snapmgr_t *m) {
    return m ? m->n : 0;
}

int wubu_snap_timeline(wubu_snapmgr_t *m, uint64_t now_ns, uint64_t interval_ns) {
    if (!m) return 0;
    if (m->last_timeline_ns != 0 && now_ns < m->last_timeline_ns + interval_ns)
        return 0;   /* not due yet */
    uint64_t id = wubu_snap_create(m, NULL, WUBU_SNAP_TIMELINE, now_ns);
    if (id == 0) return 0;
    m->last_timeline_ns = now_ns;
    return 1;
}
