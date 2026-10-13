/* test_steam_remote.c -- Steam Remote Play (host/client streaming) tests. */
#include "wubu_steam_remote.h"
#include <stdio.h>
#include <math.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)
#define NEAR(a,b) (fabs((a)-(b)) < 0.01)

int main(void)
{
    wubu_steam_remote_t rm;

    CHECK(wubu_rm_init(&rm) == 0);
    CHECK(rm.state == WUBU_RM_IDLE);
    CHECK(rm.adaptive == 1);
    CHECK(wubu_rm_init(NULL) == -1);

    /* set link: capable host, 50 Mbps client */
    CHECK(wubu_rm_set_link(&rm, 1, 50) == 0);
    CHECK(rm.host_gpu_capable == 1);
    CHECK(rm.client_bandwidth_mbps == 50);
    CHECK(wubu_rm_set_link(&rm, 1, -1) == -1);   /* negative bandwidth */
    CHECK(wubu_rm_set_link(NULL, 1, 50) == -1);

    /* connect with a want within bandwidth */
    wubu_rm_streamcfg_t want = {1920, 1080, 60, 30, 1};   /* 1080p60, 30Mbps, HDR */
    CHECK(wubu_rm_connect(&rm, &want) == 0);
    CHECK(rm.state == WUBU_RM_STREAMING);
    const wubu_rm_streamcfg_t *cfg = wubu_rm_config(&rm);
    CHECK(cfg->width == 1920 && cfg->height == 1080);
    CHECK(cfg->fps == 60);
    CHECK(cfg->bitrate_mbps == 30);   /* within 50Mbps */
    CHECK(cfg->hdr == 1);             /* GPU capable */

    /* can't connect twice */
    CHECK(wubu_rm_connect(&rm, &want) == -1);

    /* send frames while streaming */
    CHECK(wubu_rm_send_frames(&rm, 100) == 100);   /* no drop (30<50) */
    CHECK(rm.frames_sent == 100);
    CHECK(rm.frames_dropped == 0);

    /* input latency rolling average */
    CHECK(wubu_rm_input(&rm, 10.0) == 0);
    CHECK(wubu_rm_input(&rm, 20.0) == 0);
    CHECK(NEAR(rm.avg_input_latency_ms, 15.0));
    CHECK(rm.n_input_events == 2);
    CHECK(wubu_rm_input(&rm, -1.0) == -1);

    /* pause/resume */
    CHECK(wubu_rm_pause(&rm) == 0);
    CHECK(rm.state == WUBU_RM_PAUSED);
    CHECK(wubu_rm_send_frames(&rm, 10) == -1);   /* paused: no send */
    CHECK(wubu_rm_resume(&rm) == 0);
    CHECK(rm.state == WUBU_RM_STREAMING);

    /* disconnect */
    CHECK(wubu_rm_disconnect(&rm) == 0);
    CHECK(rm.state == WUBU_RM_DISCONNECTED);
    CHECK(wubu_rm_send_frames(&rm, 10) == -1);   /* disconnected */
    CHECK(wubu_rm_disconnect(&rm) == -1);         /* already idle-ish: not IDLE */
    /* reconnect after disconnect is refused (state != IDLE) */
    CHECK(wubu_rm_connect(&rm, &want) == -1);

    /* --- fresh session: bandwidth clamp + HDR drop --- */
    wubu_rm_init(&rm);
    CHECK(wubu_rm_set_link(&rm, 0, 20) == 0);     /* weak host, 20Mbps client */
    wubu_rm_streamcfg_t want2 = {3840, 2160, 120, 40, 1};   /* 4K120, 40Mbps, HDR */
    CHECK(wubu_rm_connect(&rm, &want2) == 0);
    cfg = wubu_rm_config(&rm);
    CHECK(cfg->bitrate_mbps == 20);   /* clamped to bandwidth */
    CHECK(cfg->hdr == 0);             /* host GPU not capable -> HDR off */

    /* adaptive drop: bitrate == bandwidth here, so no drop. Force a config
     * above bandwidth to exercise the drop path. */
    rm.cfg.bitrate_mbps = 40;   /* now exceeds 20Mbps link */
    int sent = wubu_rm_send_frames(&rm, 100);
    CHECK(sent < 100);          /* some dropped */
    CHECK(rm.frames_dropped > 0);
    CHECK((uint64_t)sent + rm.frames_dropped == 100);

    /* connect refused when bandwidth too low for any bitrate */
    wubu_rm_init(&rm);
    CHECK(wubu_rm_set_link(&rm, 1, 0) == 0);      /* 0 Mbps */
    wubu_rm_streamcfg_t want3 = {1280, 720, 60, 10, 0};
    CHECK(wubu_rm_connect(&rm, &want3) == -1);    /* can't carry */
    CHECK(rm.state == WUBU_RM_IDLE);

    /* bad want */
    wubu_rm_init(&rm);
    CHECK(wubu_rm_set_link(&rm, 1, 50) == 0);
    wubu_rm_streamcfg_t bad = {0, 1080, 60, 10, 0};
    CHECK(wubu_rm_connect(&rm, &bad) == -1);
    CHECK(wubu_rm_connect(&rm, NULL) == -1);

    /* bad args */
    CHECK(wubu_rm_disconnect(NULL) == -1);
    CHECK(wubu_rm_pause(NULL) == -1);
    CHECK(wubu_rm_resume(NULL) == -1);
    CHECK(wubu_rm_send_frames(NULL, 1) == -1);
    CHECK(wubu_rm_input(NULL, 1.0) == -1);
    CHECK(wubu_rm_config(NULL) == NULL);
    CHECK(wubu_rm_state(NULL) == WUBU_RM_IDLE);

    if (fails == 0) printf("ALL STEAM_REMOTE TESTS PASSED\n");
    else printf("%d STEAM_REMOTE TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
