/* test_steam_replay.c -- Steam background video recording / replay buffer. */
#include "wubu_steam_replay.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)
#define NEAR(a,b) (fabs((a)-(b)) < 0.01)

int main(void)
{
    wubu_steam_replay_t r;

    /* init: 30s capacity, not recording */
    CHECK(wubu_rp_init(&r, 30.0) == 0);
    CHECK(r.recording == 0);
    CHECK(wubu_rp_buffered_sec(&r) == 0.0);
    CHECK(wubu_rp_init(&r, 0.0) == -1);
    CHECK(wubu_rp_init(&r, -1.0) == -1);
    CHECK(wubu_rp_init(NULL, 30.0) == -1);

    /* tick while not recording: clock advances, no frames buffered */
    CHECK(wubu_rp_tick(&r, 1.0) == 0);
    CHECK(r.now_sec == 1.0);
    CHECK(r.n_frames == 0);

    /* start recording, tick 10s -> 300 frames, 10s buffered */
    CHECK(wubu_rp_set_recording(&r, 1) == 0);
    CHECK(wubu_rp_tick(&r, 10.0) == 300);
    CHECK(r.n_frames == 300);
    CHECK(NEAR(wubu_rp_buffered_sec(&r), 10.0));
    CHECK(r.n_dropped == 0);

    /* save replay of last 5s */
    int c0 = wubu_rp_save_replay(&r, 5.0, "moment1");
    CHECK(c0 == 0);
    CHECK(r.n_clips == 1);
    const wubu_rp_clip_t *clip = wubu_rp_clip(&r, 0);
    CHECK(clip != NULL);
    CHECK(NEAR(clip->duration_sec, 5.0));
    CHECK(clip->manual == 0);
    CHECK(strcmp(clip->name, "moment1") == 0);
    CHECK(clip->frame_end - clip->frame_start == 150);   /* 5s * 30fps */

    /* can't save more than buffered */
    CHECK(wubu_rp_save_replay(&r, 100.0, "toolong") == -1);

    /* overflow the ring: capacity 30s, tick 40 more seconds -> oldest dropped */
    CHECK(wubu_rp_tick(&r, 40.0) == 1200);
    /* total elapsed recording = 10 + 40 = 50s; capacity 30s -> 20s dropped */
    CHECK(NEAR(wubu_rp_buffered_sec(&r), 30.0));   /* capped at capacity */
    CHECK(r.n_dropped == (uint64_t)(20.0 * WUBU_RP_FPS));   /* 600 frames aged out */

    /* save last 30s now fills the whole ring */
    int c1 = wubu_rp_save_replay(&r, 30.0, "big");
    CHECK(c1 == 1);
    clip = wubu_rp_clip(&r, 1);
    CHECK(NEAR(clip->duration_sec, 30.0));
    CHECK(clip->frame_end - clip->frame_start == 900);   /* 30s * 30fps */

    /* manual recording: start, tick, stop */
    CHECK(wubu_rp_manual_start(&r) == 0);
    CHECK(r.manual_recording == 1);
    CHECK(wubu_rp_manual_start(&r) == -1);   /* already recording */
    CHECK(wubu_rp_tick(&r, 3.0) == 90);
    int c2 = wubu_rp_manual_stop(&r);
    CHECK(c2 == 2);
    clip = wubu_rp_clip(&r, 2);
    CHECK(clip->manual == 1);
    CHECK(NEAR(clip->duration_sec, 3.0));
    CHECK(r.manual_recording == 0);
    CHECK(wubu_rp_manual_stop(&r) == -1);   /* not recording */

    /* save_replay fails while not recording */
    CHECK(wubu_rp_set_recording(&r, 0) == 0);
    CHECK(wubu_rp_save_replay(&r, 5.0, "off") == -1);
    CHECK(wubu_rp_manual_start(&r) == -1);   /* buffer off */

    /* bad args */
    CHECK(wubu_rp_tick(&r, -1.0) == -1);
    CHECK(wubu_rp_tick(NULL, 1.0) == -1);
    CHECK(wubu_rp_save_replay(&r, 5.0, NULL) == -1);
    CHECK(wubu_rp_save_replay(&r, 0.0, "x") == -1);
    CHECK(wubu_rp_manual_start(NULL) == -1);
    CHECK(wubu_rp_manual_stop(NULL) == -1);
    CHECK(wubu_rp_buffered_sec(NULL) == -1.0);
    CHECK(wubu_rp_n_clips(NULL) == -1);
    CHECK(wubu_rp_clip(&r, 999) == NULL);
    CHECK(wubu_rp_clip(NULL, 0) == NULL);

    if (fails == 0) printf("ALL STEAM_REPLAY TESTS PASSED\n");
    else printf("%d STEAM_REPLAY TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
