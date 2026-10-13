/* test_steam_shader.c -- Steam shader pre-caching tests. */
#include "wubu_steam_shader.h"
#include <stdio.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_steam_shader_t sc;

    /* init: 5ms compile per tick */
    CHECK(wubu_shdr_init(&sc, 5) == 0);
    CHECK(sc.compile_rate_ms == 5);
    CHECK(sc.running == 1);
    CHECK(wubu_shdr_init(&sc, 0) == -1);
    CHECK(wubu_shdr_init(&sc, -1) == -1);
    CHECK(wubu_shdr_init(NULL, 5) == -1);

    /* enqueue jobs: priority + cost */
    CHECK(wubu_shdr_enqueue(&sc, "pipe_low", 1, 10) == 0);
    CHECK(wubu_shdr_enqueue(&sc, "pipe_high", 9, 10) == 0);
    CHECK(wubu_shdr_enqueue(&sc, "pipe_mid", 5, 20) == 0);
    CHECK(sc.n_jobs == 3);
    CHECK(wubu_shdr_pending(&sc) == 3);
    CHECK(wubu_shdr_done_count(&sc) == 0);
    CHECK(wubu_shdr_enqueue(&sc, "pipe_low", 1, 5) == -1);   /* dup */
    CHECK(wubu_shdr_enqueue(&sc, NULL, 1, 5) == -1);
    CHECK(wubu_shdr_enqueue(&sc, "neg", 1, -1) == -1);       /* negative cost */
    CHECK(wubu_shdr_enqueue(NULL, "x", 1, 5) == -1);

    /* work: highest priority first (pipe_high pri 9) */
    int idx = wubu_shdr_work(&sc);
    CHECK(idx == 1);   /* pipe_high */
    CHECK(sc.jobs[1].state == WUBU_SHDR_COMPILING);
    CHECK(sc.jobs[1].done_ms == 5);   /* 5 of 10 */
    /* work again: still pipe_high (highest pri, still compiling) -> completes */
    idx = wubu_shdr_work(&sc);
    CHECK(idx == 1);
    CHECK(sc.jobs[1].state == WUBU_SHDR_DONE);
    CHECK(wubu_shdr_done_count(&sc) == 1);
    CHECK(wubu_shdr_job_state(&sc, "pipe_high") == WUBU_SHDR_DONE);

    /* next: pipe_mid (pri 5) */
    idx = wubu_shdr_work(&sc);
    CHECK(idx == 2);
    /* drain the rest */
    int worked = wubu_shdr_drain(&sc);
    CHECK(worked > 0);
    CHECK(wubu_shdr_done_count(&sc) == 3);
    CHECK(wubu_shdr_pending(&sc) == 0);
    CHECK(wubu_shdr_job_state(&sc, "pipe_low") == WUBU_SHDR_DONE);
    CHECK(wubu_shdr_job_state(&sc, "pipe_mid") == WUBU_SHDR_DONE);

    /* empty queue: work returns -1 */
    CHECK(wubu_shdr_work(&sc) == -1);
    CHECK(wubu_shdr_drain(&sc) == 0);

    /* fail a job */
    CHECK(wubu_shdr_enqueue(&sc, "pipe_bad", 3, 10) == 0);
    CHECK(wubu_shdr_fail(&sc, "pipe_bad") == 0);
    CHECK(sc.jobs[3].state == WUBU_SHDR_FAILED);
    CHECK(sc.n_failed == 1);
    /* failed job is skipped by work */
    CHECK(wubu_shdr_work(&sc) == -1);   /* only pipe_bad, which is failed */
    CHECK(wubu_shdr_pending(&sc) == 0);
    /* can't fail an already-done or unknown job */
    CHECK(wubu_shdr_fail(&sc, "pipe_high") == -1);   /* already done */
    CHECK(wubu_shdr_fail(&sc, "nope") == -1);
    CHECK(wubu_shdr_fail(&sc, "pipe_bad") == -1);     /* already failed */

    /* stopped pre-cacher: work refuses */
    sc.running = 0;
    CHECK(wubu_shdr_enqueue(&sc, "pipe_more", 1, 5) == 0);   /* enqueue still ok */
    CHECK(wubu_shdr_work(&sc) == -1);   /* not running */
    sc.running = 1;
    CHECK(wubu_shdr_work(&sc) == 4);    /* resumes on pipe_more (idx 4) */
    CHECK(sc.jobs[4].state == WUBU_SHDR_DONE);   /* cost 5 == rate 5 -> done in one tick */

    /* bad args */
    CHECK(wubu_shdr_work(NULL) == -1);
    CHECK(wubu_shdr_drain(NULL) == -1);
    CHECK(wubu_shdr_fail(NULL, "x") == -1);
    CHECK(wubu_shdr_pending(NULL) == -1);
    CHECK(wubu_shdr_done_count(NULL) == -1);
    CHECK(wubu_shdr_job_state(NULL, "x") == WUBU_SHDR_PENDING);
    CHECK(wubu_shdr_job_state(&sc, NULL) == WUBU_SHDR_PENDING);

    if (fails == 0) printf("ALL STEAM_SHADER TESTS PASSED\n");
    else printf("%d STEAM_SHADER TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
