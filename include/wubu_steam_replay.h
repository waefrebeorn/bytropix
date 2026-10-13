#ifndef WUBU_STEAM_REPLAY_H
#define WUBU_STEAM_REPLAY_H

/* wubu_steam_replay.h -- Steam background video recording / replay buffer.
 *
 * Models Steam's background recording: a continuous ring buffer of recent
 * frames (the last `capacity_sec` seconds) that persists a clip on demand
 * ("save last 30s") without having recorded the whole session. The buffer
 * overwrites the oldest frames once full; saving slices the ring back by the
 * requested duration and emits a clip descriptor. Also supports manual
 * start/stop recording of a full clip. Deterministic, no stubs. */

#include <stddef.h>
#include <stdint.h>

#define WUBU_RP_MAX_CLIPS 32
#define WUBU_RP_NAME_MAX 64
#define WUBU_RP_FPS 30

typedef struct {
    char name[WUBU_RP_NAME_MAX];   /* clip file name */
    double duration_sec;           /* clip length */
    double end_ts;                 /* wall-clock time the clip ends (sec) */
    uint64_t frame_start;          /* first frame index in the clip */
    uint64_t frame_end;            /* last frame index (exclusive) */
    int manual;                    /* 1 = manual record, 0 = replay-buffer slice */
} wubu_rp_clip_t;

typedef struct {
    int recording;             /* background buffer running */
    int manual_recording;      /* manual clip in progress */
    double capacity_sec;       /* ring buffer capacity (seconds) */
    double now_sec;            /* current session time */
    uint64_t frame_index;      /* monotonically increasing frame counter */
    uint64_t buffer_start;     /* oldest frame still in the ring */
    double manual_start_ts;
    wubu_rp_clip_t clips[WUBU_RP_MAX_CLIPS];
    int n_clips;
    uint64_t n_frames;
    uint64_t n_dropped;        /* frames overwritten (aged out) */
} wubu_steam_replay_t;

/* Initialise with a ring-buffer capacity in seconds. Returns 0/-1. */
int wubu_rp_init(wubu_steam_replay_t *r, double capacity_sec);

/* Advance the session clock by `dt` seconds, emitting floor(dt*FPS) frames
 * into the ring buffer (overwriting the oldest when full). Returns frames
 * emitted, -1 on bad args. */
int wubu_rp_tick(wubu_steam_replay_t *r, double dt);

/* Start/stop the background ring buffer. */
int wubu_rp_set_recording(wubu_steam_replay_t *r, int recording);

/* Begin a manual recording (captures from now until stop). Returns 0/-1. */
int wubu_rp_manual_start(wubu_steam_replay_t *r);

/* Stop the manual recording and save it as a clip. Returns 0/-1. */
int wubu_rp_manual_stop(wubu_steam_replay_t *r);

/* Save the last `duration_sec` from the ring buffer as a clip. Returns the
 * clip index, -1 on bad args (not recording / too long / buffer empty). */
int wubu_rp_save_replay(wubu_steam_replay_t *r, double duration_sec,
                        const char *name);

/* Query. */
double wubu_rp_buffered_sec(const wubu_steam_replay_t *r);
int wubu_rp_n_clips(const wubu_steam_replay_t *r);
const wubu_rp_clip_t *wubu_rp_clip(const wubu_steam_replay_t *r, int idx);

#endif
