#ifndef WUBU_STEAM_UPDATE_H
#define WUBU_STEAM_UPDATE_H

/* wubu_steam_update.h -- Steam update channel management.
 *
 * Models the SteamOS update system: three channels (stable / beta / preview),
 * a pending-update queue with download + install phases, a rollback snapshot
 * taken before each install, and a version string per channel. Switching
 * channels queues a channel-switch update; installing applies the pending
 * update and bumps the version; rollback restores the previous snapshot.
 * Deterministic, no stubs. */

#include <stddef.h>

#define WUBU_UPD_NAME_MAX 16

typedef enum {
    WUBU_UPD_STABLE = 0,
    WUBU_UPD_BETA,
    WUBU_UPD_PREVIEW,
    WUBU_UPD_COUNT
} wubu_upd_channel_t;

typedef enum {
    WUBU_UPD_NONE = 0,       /* no pending update */
    WUBU_UPD_QUEUED,         /* update queued, not downloaded */
    WUBU_UPD_DOWNLOADED,     /* downloaded, ready to install */
    WUBU_UPD_INSTALLING,     /* install in progress */
    WUBU_UPD_INSTALLED       /* just installed */
} wubu_upd_state_t;

typedef struct {
    wubu_upd_channel_t channel;
    char version[WUBU_UPD_NAME_MAX];   /* installed version on this channel */
} wubu_upd_chan_t;

typedef struct {
    wubu_upd_chan_t channels[WUBU_UPD_COUNT];
    wubu_upd_channel_t active;          /* current channel */
    wubu_upd_state_t state;
    char pending_version[WUBU_UPD_NAME_MAX];  /* version being applied */
    /* rollback snapshot */
    int has_snapshot;
    char snapshot_version[WUBU_UPD_NAME_MAX];
    wubu_upd_channel_t snapshot_channel;
    int auto_update;                    /* auto-download+install on channel */
    int n_installs;
    int n_rollbacks;
} wubu_steam_update_t;

/* Initialise (stable active, v3.5.9, no pending update). */
int wubu_up_init(wubu_steam_update_t *u);

/* The installed version on the active channel. */
const char *wubu_up_version(const wubu_steam_update_t *u);

/* Switch the active channel. Queues a channel-switch update if the target
 * channel's version differs from what's installed. Returns 0/-1. */
int wubu_up_switch_channel(wubu_steam_update_t *u, wubu_upd_channel_t ch);

/* Download the pending update (QUEUED -> DOWNLOADED). Returns 0/1(nothing
 * pending)/-1. */
int wubu_up_download(wubu_steam_update_t *u);

/* Install the pending update (takes a rollback snapshot first). Returns
 * 0 on install, 1 if nothing to install, -1 on bad state. */
int wubu_up_install(wubu_steam_update_t *u);

/* Roll back to the pre-install snapshot. Returns 0/-1 (no snapshot). */
int wubu_up_rollback(wubu_steam_update_t *u);

/* Queue a specific new version on the active channel. */
int wubu_up_queue_version(wubu_steam_update_t *u, const char *version);

/* Set the version available on a channel (simulates a channel release). */
int wubu_up_set_channel_version(wubu_steam_update_t *u, wubu_upd_channel_t ch,
                                const char *version);

/* Enable/disable auto-update. */
int wubu_up_set_auto(wubu_steam_update_t *u, int auto_update);

/* Query. */
wubu_upd_state_t wubu_up_state(const wubu_steam_update_t *u);
wubu_upd_channel_t wubu_up_active_channel(const wubu_steam_update_t *u);
const char *wubu_up_channel_name(wubu_upd_channel_t ch);

#endif
