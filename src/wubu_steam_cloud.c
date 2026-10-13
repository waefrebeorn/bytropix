/* wubu_steam_cloud.c -- Steam Cloud Saves sync.
 * See wubu_steam_cloud.h. Real, deterministic cloud-save sync engine. */

#include "wubu_steam_cloud.h"
#include <string.h>

int wubu_cl_init(wubu_steam_cloud_t *c) {
    if (!c) return -1;
    memset(c, 0, sizeof(*c));
    c->policy = WUBU_CLOUD_RESOLVE_NEWER;
    c->quota_bytes = WUBU_CLOUD_DEFAULT_QUOTA;
    c->online = 1;
    return 0;
}

static int find(const wubu_steam_cloud_t *c, const char *name) {
    if (!name) return -1;
    for (int i = 0; i < c->n_games; i++)
        if (strcmp(c->games[i].name, name) == 0) return i;
    return -1;
}

int wubu_cl_add_game(wubu_steam_cloud_t *c, const char *name, uint64_t size_bytes) {
    if (!c || !name) return -1;
    if (find(c, name) >= 0) return -1;   /* dup */
    if (c->n_games >= WUBU_CLOUD_MAX_GAMES) return -1;
    wubu_cloud_game_t *g = &c->games[c->n_games];
    memset(g, 0, sizeof(*g));
    strncpy(g->name, name, WUBU_CLOUD_NAME_MAX - 1);
    g->size_bytes = size_bytes;
    g->auto_sync = 1;
    c->used_bytes += size_bytes;
    return c->n_games++;
}

int wubu_cl_save_local(wubu_steam_cloud_t *c, const char *name, uint64_t ts) {
    if (!c) return -1;
    int i = find(c, name);
    if (i < 0) return -1;
    wubu_cloud_game_t *g = &c->games[i];
    if (!g->has_local) { g->has_local = 1; g->local_version = 1; }
    else g->local_version++;
    g->local_ts = ts;
    return 0;
}

int wubu_cl_save_cloud(wubu_steam_cloud_t *c, const char *name, uint64_t ts) {
    if (!c) return -1;
    int i = find(c, name);
    if (i < 0) return -1;
    wubu_cloud_game_t *g = &c->games[i];
    if (!g->has_cloud) { g->has_cloud = 1; g->cloud_version = 1; }
    else g->cloud_version++;
    g->cloud_ts = ts;
    return 0;
}

wubu_cloud_state_t wubu_cl_state(const wubu_steam_cloud_t *c, const char *name) {
    if (!c) return WUBU_CLOUD_MISSING_LOCAL;
    int i = find(c, name);
    if (i < 0) return WUBU_CLOUD_MISSING_LOCAL;   /* unknown game */
    const wubu_cloud_game_t *g = &c->games[i];
    if (!g->has_local && !g->has_cloud) return WUBU_CLOUD_MISSING_LOCAL;
    if (!g->has_local) return WUBU_CLOUD_DOWNLOAD;      /* only cloud */
    if (!g->has_cloud) return WUBU_CLOUD_UPLOAD;        /* only local */
    /* both present: compare against the last-synced base */
    int local_changed = g->local_version != g->base_version;
    int cloud_changed = g->cloud_version != g->base_version;
    if (local_changed && cloud_changed) return WUBU_CLOUD_CONFLICT;
    if (local_changed) return WUBU_CLOUD_UPLOAD;
    if (cloud_changed) return WUBU_CLOUD_DOWNLOAD;
    return WUBU_CLOUD_UP_TO_DATE;
}

int wubu_cl_sync(wubu_steam_cloud_t *c, const char *name) {
    if (!c) return -1;
    int i = find(c, name);
    if (i < 0) return -1;
    if (!c->online) return -1;   /* can't sync offline */
    wubu_cloud_game_t *g = &c->games[i];
    wubu_cloud_state_t st = wubu_cl_state(c, name);

    switch (st) {
    case WUBU_CLOUD_UP_TO_DATE:
        return 1;   /* nothing to do */
    case WUBU_CLOUD_UPLOAD:
        g->has_cloud = 1;
        g->cloud_version = g->local_version;
        g->cloud_ts = g->local_ts;
        g->base_version = g->local_version;
        c->n_uploads++;
        return 0;
    case WUBU_CLOUD_DOWNLOAD:
        g->has_local = 1;
        g->local_version = g->cloud_version;
        g->local_ts = g->cloud_ts;
        g->base_version = g->cloud_version;
        c->n_downloads++;
        return 0;
    case WUBU_CLOUD_CONFLICT: {
        /* resolve per policy */
        c->n_conflicts++;
        if (c->policy == WUBU_CLOUD_RESOLVE_KEEP_LOCAL) {
            g->cloud_version = g->local_version;
            g->cloud_ts = g->local_ts;
            c->n_uploads++;
        } else if (c->policy == WUBU_CLOUD_RESOLVE_KEEP_CLOUD) {
            g->local_version = g->cloud_version;
            g->local_ts = g->cloud_ts;
            c->n_downloads++;
        } else {
            /* newer wins */
            if (g->local_ts >= g->cloud_ts) {
                g->cloud_version = g->local_version;
                g->cloud_ts = g->local_ts;
                c->n_uploads++;
            } else {
                g->local_version = g->cloud_version;
                g->local_ts = g->cloud_ts;
                c->n_downloads++;
            }
        }
        g->base_version = g->local_version;
        c->n_resolved++;
        return 0;
    }
    case WUBU_CLOUD_MISSING_LOCAL:
        /* pull cloud -> local */
        g->has_local = 1;
        g->local_version = g->cloud_version;
        g->local_ts = g->cloud_ts;
        g->base_version = g->cloud_version;
        c->n_downloads++;
        return 0;
    case WUBU_CLOUD_MISSING_CLOUD:
        /* push local -> cloud */
        g->has_cloud = 1;
        g->cloud_version = g->local_version;
        g->cloud_ts = g->local_ts;
        g->base_version = g->local_version;
        c->n_uploads++;
        return 0;
    }
    return -1;
}

int wubu_cl_sync_all(wubu_steam_cloud_t *c) {
    if (!c) return -1;
    if (!c->online) return -1;
    int n = 0;
    for (int i = 0; i < c->n_games; i++) {
        if (!c->games[i].auto_sync) continue;
        if (wubu_cl_sync(c, c->games[i].name) == 0) n++;
    }
    return n;
}

int wubu_cl_set_auto(wubu_steam_cloud_t *c, const char *name, int auto_sync) {
    if (!c) return -1;
    int i = find(c, name);
    if (i < 0) return -1;
    c->games[i].auto_sync = auto_sync ? 1 : 0;
    return 0;
}

void wubu_cl_set_online(wubu_steam_cloud_t *c, int online) {
    if (c) c->online = online ? 1 : 0;
}

int wubu_cl_quota_exceeded(const wubu_steam_cloud_t *c, uint64_t bytes) {
    if (!c) return -1;
    return (c->used_bytes + bytes > c->quota_bytes) ? 1 : 0;
}
