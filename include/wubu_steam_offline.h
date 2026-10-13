#ifndef WUBU_STEAM_OFFLINE_H
#define WUBU_STEAM_OFFLINE_H

/* wubu_steam_offline.h -- Steam offline mode (cached library play).
 *
 * Models Steam's offline mode: when the network drops, only games that are
 * fully installed + have a cached local entitlement (offline license token)
 * can be launched; online-only games (DRM check, no cached token) are blocked.
 * Tracks per-game install state + cached-offline status, an offline-mode
 * toggle, and the count of playable games. Deterministic, no stubs. */

#include <stddef.h>
#include <stdint.h>

#define WUBU_OFF_MAX_GAMES 64
#define WUBU_OFF_NAME_MAX 48

typedef enum {
    WUBU_OFF_NOT_INSTALLED = 0,
    WUBU_OFF_INSTALLED,
    WUBU_OFF_DOWNLOADING
} wubu_off_install_t;

typedef struct {
    char name[WUBU_OFF_NAME_MAX];
    wubu_off_install_t install;
    int offline_cached;     /* 1 = has a cached offline entitlement token */
    int always_online;      /* 1 = requires a live connection (DRM check) */
    uint64_t size_mb;
} wubu_off_game_t;

typedef struct {
    wubu_off_game_t games[WUBU_OFF_MAX_GAMES];
    int n_games;
    int offline;            /* offline mode active */
    int n_blocked;          /* launch attempts blocked by offline policy */
} wubu_steam_offline_t;

/* Initialise (online mode). */
int wubu_off_init(wubu_steam_offline_t *o);

/* Add a game to the library. Returns index or -1 (dup/full). */
int wubu_off_add_game(wubu_steam_offline_t *o, const char *name,
                      uint64_t size_mb, int always_online);

/* Mark a game installed + offline-cached (the entitlement was cached while
 * online). Returns 0/-1. */
int wubu_off_mark_installed(wubu_steam_offline_t *o, const char *name,
                            int offline_cached);

/* Enter/exit offline mode. */
int wubu_off_set_mode(wubu_steam_offline_t *o, int offline);

/* Can this game be launched in the current mode?
 *   online  -> yes if installed
 *   offline -> yes if installed AND offline_cached AND NOT always_online
 * Returns 1/0/-1 (unknown game). */
int wubu_off_can_launch(wubu_steam_offline_t *o, const char *name);

/* Attempt a launch: returns 1 (launched), 0 (blocked, increments n_blocked),
 * -1 (bad args/unknown). */
int wubu_off_launch(wubu_steam_offline_t *o, const char *name);

/* Count games playable in the current mode. */
int wubu_off_playable_count(wubu_steam_offline_t *o);

#endif
