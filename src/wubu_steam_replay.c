/* wubu_steam_replay.c -- Steam background video recording / replay buffer.
 * See wubu_steam_replay.h. Real, deterministic ring-buffer recorder. */

#include "wubu_steam_replay.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

int wubu_rp_init(wubu_steam_replay_t *r, double capacity_sec) {
    if (!r || capacity_sec <= 0.0) return -1;
    memset(r, 0, sizeof(*r));
    r->capacity_sec = capacity_sec;
    return 0;
}

int wubu_rp_set_recording(wubu_steam_replay_t *r, int recording) {
    if (!r) return -1;
    r->recording = recording ? 1 : 0;
    if (!r->recording && r->manual_recording) {
        /* stopping the buffer also stops a manual recording */
        r->manual_recording = 0;
    }
    return 0;
}

int wubu_rp_tick(wubu_steam_replay_t *r, double dt) {
    if (!r || dt < 0.0) return -1;
    r->now_sec += dt;
    if (!r->recording) return 0;
    int frames = (int)floor(dt * WUBU_RP_FPS + 1e-9);
    for (int i = 0; i < frames; i++) {
        uint64_t idx = r->frame_index;
        r->frame_index++;
        r->n_frames++;
        /* ring capacity in frames */
        uint64_t cap_frames = (uint64_t)(r->capacity_sec * WUBU_RP_FPS + 1e-9);
        if (cap_frames == 0) cap_frames = 1;
        if (r->frame_index - r->buffer_start > cap_frames) {
            r->buffer_start++;   /* age out the oldest frame */
            r->n_dropped++;
        }
        (void)idx;
    }
    return frames;
}

int wubu_rp_manual_start(wubu_steam_replay_t *r) {
    if (!r || !r->recording) return -1;
    if (r->manual_recording) return -1;   /* already recording */
    r->manual_recording = 1;
    r->manual_start_ts = r->now_sec;
    return 0;
}

static int add_clip(wubu_steam_replay_t *r, const char *name,
                    double duration, uint64_t fstart, uint64_t fend,
                    int manual) {
    if (r->n_clips >= WUBU_RP_MAX_CLIPS) return -1;
    wubu_rp_clip_t *c = &r->clips[r->n_clips];
    memset(c, 0, sizeof(*c));
    if (name) { strncpy(c->name, name, WUBU_RP_NAME_MAX - 1); }
    c->duration_sec = duration;
    c->end_ts = r->now_sec;
    c->frame_start = fstart;
    c->frame_end = fend;
    c->manual = manual;
    r->n_clips++;
    return r->n_clips - 1;
}

int wubu_rp_manual_stop(wubu_steam_replay_t *r) {
    if (!r || !r->manual_recording) return -1;
    double dur = r->now_sec - r->manual_start_ts;
    if (dur < 0.0) dur = 0.0;
    uint64_t fstart = (uint64_t)(r->manual_start_ts * WUBU_RP_FPS + 1e-9);
    uint64_t fend = r->frame_index;
    if (fstart < r->buffer_start) fstart = r->buffer_start;  /* clamp */
    char name[WUBU_RP_NAME_MAX];
    snprintf(name, sizeof(name), "clip_%d", r->n_clips + 1);
    int idx = add_clip(r, name, dur, fstart, fend, 1);
    r->manual_recording = 0;
    return idx;
}

int wubu_rp_save_replay(wubu_steam_replay_t *r, double duration_sec,
                        const char *name) {
    if (!r || !name) return -1;
    if (!r->recording) return -1;
    if (duration_sec <= 0.0) return -1;
    if (duration_sec > wubu_rp_buffered_sec(r)) return -1;   /* not enough buffered */
    /* slice the last `duration_sec` from the ring */
    uint64_t want = (uint64_t)(duration_sec * WUBU_RP_FPS + 1e-9);
    uint64_t avail = r->frame_index - r->buffer_start;
    if (want > avail) want = avail;
    uint64_t fend = r->frame_index;
    uint64_t fstart = fend - want;
    if (fstart < r->buffer_start) fstart = r->buffer_start;
    double actual_dur = (double)(fend - fstart) / WUBU_RP_FPS;
    return add_clip(r, name, actual_dur, fstart, fend, 0);
}

double wubu_rp_buffered_sec(const wubu_steam_replay_t *r) {
    if (!r) return -1.0;
    uint64_t avail = r->frame_index - r->buffer_start;
    return (double)avail / WUBU_RP_FPS;
}

int wubu_rp_n_clips(const wubu_steam_replay_t *r) {
    if (!r) return -1;
    return r->n_clips;
}

const wubu_rp_clip_t *wubu_rp_clip(const wubu_steam_replay_t *r, int idx) {
    if (!r || idx < 0 || idx >= r->n_clips) return NULL;
    return &r->clips[idx];
}
