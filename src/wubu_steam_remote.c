/* wubu_steam_remote.c -- Steam Remote Play (host/client streaming session).
 * See wubu_steam_remote.h. Real, deterministic streaming session model. */

#include "wubu_steam_remote.h"
#include <string.h>

int wubu_rm_init(wubu_steam_remote_t *rm) {
    if (!rm) return -1;
    memset(rm, 0, sizeof(*rm));
    rm->state = WUBU_RM_IDLE;
    rm->adaptive = 1;
    return 0;
}

int wubu_rm_set_link(wubu_steam_remote_t *rm, int host_gpu_capable,
                     int client_bandwidth_mbps) {
    if (!rm || client_bandwidth_mbps < 0) return -1;
    rm->host_gpu_capable = host_gpu_capable ? 1 : 0;
    rm->client_bandwidth_mbps = client_bandwidth_mbps;
    return 0;
}

int wubu_rm_connect(wubu_steam_remote_t *rm, const wubu_rm_streamcfg_t *want) {
    if (!rm || !want) return -1;
    if (rm->state != WUBU_RM_IDLE) return -1;
    /* handshake */
    rm->state = WUBU_RM_HANDSHAKE;
    /* negotiate: clamp resolution + bitrate by bandwidth; HDR needs GPU */
    wubu_rm_streamcfg_t cfg = *want;
    if (cfg.width <= 0 || cfg.height <= 0 || cfg.fps <= 0) return -1;
    /* HDR requires a capable host GPU */
    if (cfg.hdr && !rm->host_gpu_capable) cfg.hdr = 0;
    /* clamp bitrate to the client bandwidth */
    if (cfg.bitrate_mbps > rm->client_bandwidth_mbps)
        cfg.bitrate_mbps = rm->client_bandwidth_mbps;
    /* if bandwidth can't carry even a low bitrate, refuse */
    if (cfg.bitrate_mbps <= 0) { rm->state = WUBU_RM_IDLE; return -1; }
    rm->cfg = cfg;
    rm->state = WUBU_RM_STREAMING;
    return 0;
}

int wubu_rm_disconnect(wubu_steam_remote_t *rm) {
    if (!rm) return -1;
    if (rm->state == WUBU_RM_IDLE || rm->state == WUBU_RM_DISCONNECTED) return -1;
    rm->state = WUBU_RM_DISCONNECTED;
    return 0;
}

int wubu_rm_pause(wubu_steam_remote_t *rm) {
    if (!rm || rm->state != WUBU_RM_STREAMING) return -1;
    rm->state = WUBU_RM_PAUSED;
    return 0;
}

int wubu_rm_resume(wubu_steam_remote_t *rm) {
    if (!rm || rm->state != WUBU_RM_PAUSED) return -1;
    rm->state = WUBU_RM_STREAMING;
    return 0;
}

int wubu_rm_send_frames(wubu_steam_remote_t *rm, uint64_t n) {
    if (!rm) return -1;
    if (rm->state != WUBU_RM_STREAMING) return -1;
    uint64_t sent = n;
    if (rm->adaptive) {
        /* drop frames proportionally if the negotiated bitrate exceeds the
         * available bandwidth (a simple congestion model). */
        if (rm->cfg.bitrate_mbps > rm->client_bandwidth_mbps) {
            uint64_t excess = rm->cfg.bitrate_mbps - rm->client_bandwidth_mbps;
            uint64_t drop = (uint64_t)((double)n * excess / rm->cfg.bitrate_mbps);
            if (drop > n) drop = n;
            rm->frames_dropped += drop;
            sent = n - drop;
        }
    }
    rm->frames_sent += sent;
    return (int)sent;
}

int wubu_rm_input(wubu_steam_remote_t *rm, double latency_ms) {
    if (!rm || latency_ms < 0.0) return -1;
    /* rolling average */
    double total = rm->avg_input_latency_ms * rm->n_input_events + latency_ms;
    rm->n_input_events++;
    rm->avg_input_latency_ms = total / rm->n_input_events;
    return 0;
}

wubu_rm_state_t wubu_rm_state(const wubu_steam_remote_t *rm) {
    if (!rm) return WUBU_RM_IDLE;
    return rm->state;
}

const wubu_rm_streamcfg_t *wubu_rm_config(const wubu_steam_remote_t *rm) {
    if (!rm) return NULL;
    return &rm->cfg;
}
