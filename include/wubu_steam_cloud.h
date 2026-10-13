#ifndef WUBU_STEAM_CLOUD_H
#define WUBU_STEAM_CLOUD_H

/* wubu_steam_cloud.h -- Steam Cloud Saves sync.
 *
 * Models Steam Cloud save synchronization for a set of games. Each game has a
 * local save and a cloud save with a monotonic version + timestamp; the sync
 * policy resolves conflicts (newer wins, with an explicit "keep local" /
 * "keep cloud" override) and tracks quota. Deterministic, no stubs. */

#include <stddef.h>
#include <stdint.h>

#define WUBU_CLOUD_MAX_GAMES 64
#define WUBU_CLOUD_NAME_MAX 48
#define WUBU_CLOUD_DEFAULT_QUOTA (1024u * 1024u * 1024u)  /* 1 GiB */

typedef enum {
    WUBU_CLOUD_UP_TO_DATE = 0,   /* local == cloud, nothing to do */
    WUBU_CLOUD_UPLOAD,           /* local newer -> push to cloud */
    WUBU_CLOUD_DOWNLOAD,         /* cloud newer -> pull to local */
    WUBU_CLOUD_CONFLICT,         /* both changed since last sync */
    WUBU_CLOUD_MISSING_LOCAL,    /* no local save */
    WUBU_CLOUD_MISSING_CLOUD     /* no cloud save */
} wubu_cloud_state_t;

typedef enum {
    WUBU_CLOUD_RESOLVE_NEWER = 0,  /* newer timestamp wins (default) */
    WUBU_CLOUD_RESOLVE_KEEP_LOCAL,
    WUBU_CLOUD_RESOLVE_KEEP_CLOUD
} wubu_cloud_policy_t;

typedef struct {
    char name[WUBU_CLOUD_NAME_MAX];
    int has_local;
    int has_cloud;
    uint64_t local_version;
    uint64_t cloud_version;
    uint64_t local_ts;      /* last local modification */
    uint64_t cloud_ts;      /* last cloud modification */
    uint64_t base_version;  /* version at last successful sync */
    uint64_t size_bytes;
    int auto_sync;          /* auto-sync enabled for this game */
} wubu_cloud_game_t;

typedef struct {
    wubu_cloud_game_t games[WUBU_CLOUD_MAX_GAMES];
    int n_games;
    wubu_cloud_policy_t policy;
    uint64_t quota_bytes;
    uint64_t used_bytes;
    int online;             /* cloud reachable */
    int n_uploads;
    int n_downloads;
    int n_conflicts;
    int n_resolved;
} wubu_steam_cloud_t;

/* Initialise (online, 1 GiB quota, newer-wins policy). */
int wubu_cl_init(wubu_steam_cloud_t *c);

/* Register a game. Returns index or -1 (dup/full). */
int wubu_cl_add_game(wubu_steam_cloud_t *c, const char *name, uint64_t size_bytes);

/* Mark a local save (bumps local version + timestamp). */
int wubu_cl_save_local(wubu_steam_cloud_t *c, const char *name, uint64_t ts);

/* Mark a cloud save (bumps cloud version + timestamp). */
int wubu_cl_save_cloud(wubu_steam_cloud_t *c, const char *name, uint64_t ts);

/* Evaluate the sync state of a game (what a sync would do). */
wubu_cloud_state_t wubu_cl_state(const wubu_steam_cloud_t *c, const char *name);

/* Perform a sync for one game under the current policy. Returns 0 on
 * success (applies upload/download/conflict resolution), -1 on bad args,
 * 1 if nothing to do (up-to-date). */
int wubu_cl_sync(wubu_steam_cloud_t *c, const char *name);

/* Sync all auto-sync games. Returns the number synced, -1 on bad args. */
int wubu_cl_sync_all(wubu_steam_cloud_t *c);

/* Set per-game auto-sync. */
int wubu_cl_set_auto(wubu_steam_cloud_t *c, const char *name, int auto_sync);

/* Set online/offline. */
void wubu_cl_set_online(wubu_steam_cloud_t *c, int online);

/* Quota check: would adding `bytes` exceed the quota? */
int wubu_cl_quota_exceeded(const wubu_steam_cloud_t *c, uint64_t bytes);

#endif
