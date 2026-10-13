/* wubu_steam_update.c -- Steam update channel management.
 * See wubu_steam_update.h. Real, deterministic update channel state machine. */

#include "wubu_steam_update.h"
#include <string.h>

static const char *g_channel_names[WUBU_UPD_COUNT] = {
    "stable", "beta", "preview"
};

const char *wubu_up_channel_name(wubu_upd_channel_t ch) {
    if (ch < 0 || ch >= WUBU_UPD_COUNT) return NULL;
    return g_channel_names[ch];
}

int wubu_up_init(wubu_steam_update_t *u) {
    if (!u) return -1;
    memset(u, 0, sizeof(*u));
    u->active = WUBU_UPD_STABLE;
    u->state = WUBU_UPD_NONE;
    u->auto_update = 1;
    /* each channel has its own installed version */
    for (int i = 0; i < WUBU_UPD_COUNT; i++) {
        u->channels[i].channel = (wubu_upd_channel_t)i;
        strncpy(u->channels[i].version, "3.5.9", WUBU_UPD_NAME_MAX - 1);
    }
    return 0;
}

const char *wubu_up_version(const wubu_steam_update_t *u) {
    if (!u) return NULL;
    return u->channels[u->active].version;
}

int wubu_up_set_channel_version(wubu_steam_update_t *u, wubu_upd_channel_t ch,
                                const char *version) {
    if (!u || !version || ch < 0 || ch >= WUBU_UPD_COUNT) return -1;
    strncpy(u->channels[ch].version, version, WUBU_UPD_NAME_MAX - 1);
    u->channels[ch].version[WUBU_UPD_NAME_MAX - 1] = '\0';
    return 0;
}

int wubu_up_switch_channel(wubu_steam_update_t *u, wubu_upd_channel_t ch) {
    if (!u || ch < 0 || ch >= WUBU_UPD_COUNT) return -1;
    if (u->state == WUBU_UPD_INSTALLING) return -1;   /* can't switch mid-install */
    u->active = ch;
    /* if the newly-active channel's version differs from the installed base,
     * queue a channel-switch update. */
    /* We treat a channel switch as always requiring a re-sync of that
     * channel's version, so queue it. */
    strncpy(u->pending_version, u->channels[ch].version, WUBU_UPD_NAME_MAX - 1);
    u->pending_version[WUBU_UPD_NAME_MAX - 1] = '\0';
    u->state = WUBU_UPD_QUEUED;
    return 0;
}

int wubu_up_queue_version(wubu_steam_update_t *u, const char *version) {
    if (!u || !version) return -1;
    if (u->state == WUBU_UPD_INSTALLING) return -1;
    strncpy(u->pending_version, version, WUBU_UPD_NAME_MAX - 1);
    u->pending_version[WUBU_UPD_NAME_MAX - 1] = '\0';
    u->state = WUBU_UPD_QUEUED;
    return 0;
}

int wubu_up_download(wubu_steam_update_t *u) {
    if (!u) return -1;
    if (u->state != WUBU_UPD_QUEUED) return 1;   /* nothing pending */
    u->state = WUBU_UPD_DOWNLOADED;
    return 0;
}

int wubu_up_install(wubu_steam_update_t *u) {
    if (!u) return -1;
    if (u->state == WUBU_UPD_QUEUED) {
        /* auto-download before install when auto_update is on */
        if (!u->auto_update) return -1;
        u->state = WUBU_UPD_DOWNLOADED;
    }
    if (u->state != WUBU_UPD_DOWNLOADED) return 1;   /* nothing to install */
    u->state = WUBU_UPD_INSTALLING;
    /* take a rollback snapshot of the current installed version */
    u->has_snapshot = 1;
    strncpy(u->snapshot_version, u->channels[u->active].version,
            WUBU_UPD_NAME_MAX - 1);
    u->snapshot_channel = u->active;
    /* apply the pending version */
    strncpy(u->channels[u->active].version, u->pending_version,
            WUBU_UPD_NAME_MAX - 1);
    u->channels[u->active].version[WUBU_UPD_NAME_MAX - 1] = '\0';
    u->state = WUBU_UPD_INSTALLED;
    u->n_installs++;
    return 0;
}

int wubu_up_rollback(wubu_steam_update_t *u) {
    if (!u || !u->has_snapshot) return -1;
    if (u->state == WUBU_UPD_INSTALLING) return -1;
    strncpy(u->channels[u->snapshot_channel].version, u->snapshot_version,
            WUBU_UPD_NAME_MAX - 1);
    u->channels[u->snapshot_channel].version[WUBU_UPD_NAME_MAX - 1] = '\0';
    u->has_snapshot = 0;
    u->state = WUBU_UPD_NONE;
    u->n_rollbacks++;
    return 0;
}

int wubu_up_set_auto(wubu_steam_update_t *u, int auto_update) {
    if (!u) return -1;
    u->auto_update = auto_update ? 1 : 0;
    return 0;
}

wubu_upd_state_t wubu_up_state(const wubu_steam_update_t *u) {
    if (!u) return WUBU_UPD_NONE;
    return u->state;
}

wubu_upd_channel_t wubu_up_active_channel(const wubu_steam_update_t *u) {
    if (!u) return WUBU_UPD_STABLE;
    return u->active;
}
