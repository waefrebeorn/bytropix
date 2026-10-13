#ifndef WUBU_STEAM_REMOTE_H
#define WUBU_STEAM_REMOTE_H

/* wubu_steam_remote.h -- Steam Remote Play (host/client streaming session).
 *
 * Models a Remote Play streaming session between a host (runs the game) and a
 * client (streams it): capability negotiation (resolution, fps, bitrate vs
 * available bandwidth), a session lifecycle (idle -> handshake -> streaming ->
 * paused/disconnected), input round-trip latency tracking, and adaptive
 * bitrate. Deterministic, no stubs. */

#include <stddef.h>
#include <stdint.h>

typedef enum {
    WUBU_RM_IDLE = 0,
    WUBU_RM_HANDSHAKE,     /* negotiating capabilities */
    WUBU_RM_STREAMING,     /* active stream */
    WUBU_RM_PAUSED,        /* stream paused (host suspended) */
    WUBU_RM_DISCONNECTED
} wubu_rm_state_t;

typedef struct {
    int width, height;
    int fps;
    int bitrate_mbps;
    int hdr;
} wubu_rm_streamcfg_t;

typedef struct {
    wubu_rm_state_t state;
    wubu_rm_streamcfg_t cfg;         /* negotiated stream config */
    int host_gpu_capable;
    int client_bandwidth_mbps;       /* client's available uplink */
    uint64_t frames_sent;
    uint64_t frames_dropped;         /* dropped for bandwidth */
    double avg_input_latency_ms;     /* rolling input RTT */
    uint64_t n_input_events;
    int adaptive;                    /* adaptive bitrate enabled */
} wubu_steam_remote_t;

/* Initialise (idle). */
int wubu_rm_init(wubu_steam_remote_t *rm);

/* Set host GPU capability + client bandwidth (pre-negotiation). */
int wubu_rm_set_link(wubu_steam_remote_t *rm, int host_gpu_capable,
                     int client_bandwidth_mbps);

/* Begin a session: negotiate a stream config for the requested target,
 * clamped by the client bandwidth and host GPU. Returns 0/-1. */
int wubu_rm_connect(wubu_steam_remote_t *rm, const wubu_rm_streamcfg_t *want);

/* Disconnect the session. */
int wubu_rm_disconnect(wubu_steam_remote_t *rm);

/* Pause/resume the stream. */
int wubu_rm_pause(wubu_steam_remote_t *rm);
int wubu_rm_resume(wubu_steam_remote_t *rm);

/* Emit `n` frames; drops frames that exceed the bandwidth budget when
 * adaptive. Returns frames actually sent, -1 on bad state. */
int wubu_rm_send_frames(wubu_steam_remote_t *rm, uint64_t n);

/* Record an input round-trip latency sample (ms). */
int wubu_rm_input(wubu_steam_remote_t *rm, double latency_ms);

/* Query. */
wubu_rm_state_t wubu_rm_state(const wubu_steam_remote_t *rm);
const wubu_rm_streamcfg_t *wubu_rm_config(const wubu_steam_remote_t *rm);

#endif
