/* wubu_steam_offline.c -- Steam offline mode (cached library play).
 * See wubu_steam_offline.h. Real, deterministic offline-mode gate. */

#include "wubu_steam_offline.h"
#include <string.h>

int wubu_off_init(wubu_steam_offline_t *o) {
    if (!o) return -1;
    memset(o, 0, sizeof(*o));
    o->offline = 0;
    return 0;
}

static int find(const wubu_steam_offline_t *o, const char *name) {
    if (!name) return -1;
    for (int i = 0; i < o->n_games; i++)
        if (strcmp(o->games[i].name, name) == 0) return i;
    return -1;
}

int wubu_off_add_game(wubu_steam_offline_t *o, const char *name,
                      uint64_t size_mb, int always_online) {
    if (!o || !name) return -1;
    if (find(o, name) >= 0) return -1;   /* dup */
    if (o->n_games >= WUBU_OFF_MAX_GAMES) return -1;
    wubu_off_game_t *g = &o->games[o->n_games];
    memset(g, 0, sizeof(*g));
    strncpy(g->name, name, WUBU_OFF_NAME_MAX - 1);
    g->size_mb = size_mb;
    g->always_online = always_online ? 1 : 0;
    g->install = WUBU_OFF_NOT_INSTALLED;
    g->offline_cached = 0;
    return o->n_games++;
}

int wubu_off_mark_installed(wubu_steam_offline_t *o, const char *name,
                            int offline_cached) {
    int i = find(o, name);
    if (i < 0) return -1;
    o->games[i].install = WUBU_OFF_INSTALLED;
    o->games[i].offline_cached = offline_cached ? 1 : 0;
    return 0;
}

int wubu_off_set_mode(wubu_steam_offline_t *o, int offline) {
    if (!o) return -1;
    o->offline = offline ? 1 : 0;
    return 0;
}

int wubu_off_can_launch(wubu_steam_offline_t *o, const char *name) {
    if (!o) return -1;
    int i = find(o, name);
    if (i < 0) return -1;
    wubu_off_game_t *g = &o->games[i];
    if (g->install != WUBU_OFF_INSTALLED) return 0;   /* must be installed */
    if (!o->offline) return 1;                         /* online: installed is enough */
    /* offline: needs a cached entitlement and no live-connection requirement */
    if (!g->offline_cached) return 0;
    if (g->always_online) return 0;
    return 1;
}

int wubu_off_launch(wubu_steam_offline_t *o, const char *name) {
    if (!o) return -1;
    int can = wubu_off_can_launch(o, name);
    if (can < 0) return -1;
    if (can == 0) { o->n_blocked++; return 0; }
    return 1;
}

int wubu_off_playable_count(wubu_steam_offline_t *o) {
    if (!o) return -1;
    int n = 0;
    for (int i = 0; i < o->n_games; i++)
        if (wubu_off_can_launch(o, o->games[i].name) == 1) n++;
    return n;
}
