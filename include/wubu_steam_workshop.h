#ifndef WUBU_STEAM_WORKSHOP_H
#define WUBU_STEAM_WORKSHOP_H

/* wubu_steam_workshop.h -- Steam Workshop / mod manager.
 *
 * Models Steam Workshop mod management for a game: subscribe/unsubscribe to
 * mods, track download state, and maintain an explicit load order (the order
 * mods apply, later overrides earlier). Supports enable/disable per mod,
 * conflict detection (two mods touching the same file), and dependency
 * resolution (a mod requiring another must load after it). Deterministic, no
 * stubs. */

#include <stddef.h>
#include <stdint.h>

#define WUBU_WS_MAX_MODS 64
#define WUBU_WS_MAX_DEPS 8
#define WUBU_WS_MAX_FILES 16
#define WUBU_WS_NAME_MAX 48
#define WUBU_WS_FILE_MAX 64

typedef enum {
    WUBU_WS_NONE = 0,        /* not subscribed */
    WUBU_WS_SUBSCRIBED,      /* subscribed, not downloaded */
    WUBU_WS_DOWNLOADING,
    WUBU_WS_DOWNLOADED,      /* ready */
    WUBU_WS_FAILED           /* download failed */
} wubu_ws_dlstate_t;

typedef struct {
    uint64_t id;
    char name[WUBU_WS_NAME_MAX];
    char author[WUBU_WS_NAME_MAX];
    wubu_ws_dlstate_t dl;
    int enabled;             /* enabled mods load; disabled are skipped */
    uint64_t deps[WUBU_WS_MAX_DEPS];   /* mod ids this mod requires */
    int n_deps;
    char files[WUBU_WS_MAX_FILES][WUBU_WS_FILE_MAX];  /* files it provides */
    int n_files;
    uint64_t version;        /* for update detection */
} wubu_ws_mod_t;

typedef struct {
    wubu_ws_mod_t mods[WUBU_WS_MAX_MODS];
    int n_mods;
    uint64_t load_order[WUBU_WS_MAX_MODS];  /* mod ids in load order */
    int n_load_order;
    uint64_t next_id;
    int n_conflicts;         /* detected file conflicts in current load order */
} wubu_steam_workshop_t;

/* Initialise. */
int wubu_ws_init(wubu_steam_workshop_t *w);

/* Publish/subscribe a mod. Returns the mod id, or -1. */
uint64_t wubu_ws_subscribe(wubu_steam_workshop_t *w, const char *name,
                           const char *author);

/* Unsubscribe (remove) a mod by id. Also removes it from the load order. */
int wubu_ws_unsubscribe(wubu_steam_workshop_t *w, uint64_t id);

/* Add a dependency (mod `id` requires `dep_id`). Returns 0/-1. */
int wubu_ws_add_dep(wubu_steam_workshop_t *w, uint64_t id, uint64_t dep_id);

/* Add a file the mod provides (for conflict detection). */
int wubu_ws_add_file(wubu_steam_workshop_t *w, uint64_t id, const char *file);

/* Start/complete a download. */
int wubu_ws_download_start(wubu_steam_workshop_t *w, uint64_t id);
int wubu_ws_download_done(wubu_steam_workshop_t *w, uint64_t id);
int wubu_ws_download_fail(wubu_steam_workshop_t *w, uint64_t id);

/* Enable/disable a downloaded mod. */
int wubu_ws_set_enabled(wubu_steam_workshop_t *w, uint64_t id, int enabled);

/* Append a mod to the end of the load order. Only downloaded+enabled mods
 * load. Returns 0/-1. */
int wubu_ws_load(wubu_steam_workshop_t *w, uint64_t id);

/* Remove a mod from the load order. */
int wubu_ws_unload(wubu_steam_workshop_t *w, uint64_t id);

/* Move a loaded mod earlier/later in the load order. Returns 0/-1. */
int wubu_ws_reorder(wubu_steam_workshop_t *w, uint64_t id, int delta);

/* Count file conflicts in the current load order (two loaded+enabled mods
 * providing the same file). */
int wubu_ws_count_conflicts(wubu_steam_workshop_t *w);

/* Query. */
int wubu_ws_find(const wubu_steam_workshop_t *w, uint64_t id);
wubu_ws_dlstate_t wubu_ws_dl_state(const wubu_steam_workshop_t *w, uint64_t id);
int wubu_ws_n_loaded(const wubu_steam_workshop_t *w);

#endif
