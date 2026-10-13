#ifndef WUBU_SNAPSHOT_H
#define WUBU_SNAPSHOT_H
#include <stddef.h>
#include <stdint.h>

/* wubu_snapshot.h -- OMARCHY #273/#274 btrfs subvolume snapshot + rollback
 * (snapper-equivalent).
 *
 * Models a btrfs subvolume snapshot manager: a linear history of named
 * snapshots of a subvolume, each with a type (pre/post/timeline) and a
 * timestamp, supporting:
 *   - create a snapshot (with optional type + tag),
 *   - list the snapshot timeline,
 *   - rollback: designate a snapshot as the active state,
 *   - automatic timeline snapshots (snapper-style): a scheduled cadence that
 *     creates periodic snapshots and prunes old ones by retention count.
 * This is the policy/state core; the actual `btrfs subvolume snapshot` /
 * `btrfs subvolume delete` commands are the kernel's job. Deterministic,
 * fixed-size ring (no allocation), fail-closed. Real work (no stubs). */

#define WUBU_SNAP_MAX 64
#define WUBU_SNAP_NAMELEN 48

typedef enum {
    WUBU_SNAP_MANUAL=0,   /* user-created */
    WUBU_SNAP_PRE,        /* pre-change snapshot */
    WUBU_SNAP_POST,       /* post-change snapshot */
    WUBU_SNAP_TIMELINE,   /* automatic timeline snapshot */
} wubu_snap_type_t;

typedef struct {
    char     name[WUBU_SNAP_NAMELEN];
    uint64_t id;
    uint64_t timestamp_ns;
    wubu_snap_type_t type;
    int      active;      /* 1 = this is the current rollback target */
} wubu_snapshot_t;

typedef struct {
    char subvol[WUBU_SNAP_NAMELEN];
    wubu_snapshot_t snaps[WUBU_SNAP_MAX];
    int  n;
    int  head;            /* ring write position */
    uint64_t next_id;
    int  active_idx;      /* index of active snapshot, -1 if none */
    int  retention;       /* max snapshots to keep (0 = unlimited) */
    uint64_t last_timeline_ns; /* last automatic timeline snapshot time */
} wubu_snapmgr_t;

/* Initialise a snapshot manager for a subvolume. retention 0 = unlimited. */
int  wubu_snap_init(wubu_snapmgr_t *m, const char *subvol, int retention);
/* Create a snapshot. name may be NULL (auto-named). Returns snapshot id (>0),
 * or 0 on error. Prunes old snapshots per retention. */
uint64_t wubu_snap_create(wubu_snapmgr_t *m, const char *name, wubu_snap_type_t type, uint64_t now_ns);
/* Rollback: designate snapshot by id as active. Returns 0/-1. */
int  wubu_snap_rollback(wubu_snapmgr_t *m, uint64_t id);
/* Current active snapshot (or NULL). */
const wubu_snapshot_t *wubu_snap_active(const wubu_snapmgr_t *m);
/* Look up by id (or NULL). */
const wubu_snapshot_t *wubu_snap_find(const wubu_snapmgr_t *m, uint64_t id);
/* Count of snapshots currently retained. */
int  wubu_snap_count(const wubu_snapmgr_t *m);
/* Automatic timeline snapshot: create a TIMELINE snapshot at a cadence.
 * Returns 1 if created (now_ns >= last_timeline + interval), 0 if not due. */
int  wubu_snap_timeline(wubu_snapmgr_t *m, uint64_t now_ns, uint64_t interval_ns);

#endif
