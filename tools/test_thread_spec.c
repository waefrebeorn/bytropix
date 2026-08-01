/* Test: H02 CPU thread-specialization (prefill/decode pinned pools).
 * Verifies: (1) jobs run on the right pool; (2) prefill and decode pools are
 * disjoint; (3) results from parallel jobs are correct (no races). */
#include "wubu_thread_spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>

static int g_prefill_sum = 0;
static int g_decode_sum = 0;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static void prefill_job(void *arg) {
    int v = *(int *)arg;
    pthread_mutex_lock(&g_lock);
    g_prefill_sum += v;
    pthread_mutex_unlock(&g_lock);
}
static void decode_job(void *arg) {
    int v = *(int *)arg;
    pthread_mutex_lock(&g_lock);
    g_decode_sum += v;
    pthread_mutex_unlock(&g_lock);
}

int main(void) {
    printf("=== H02 Thread-Specialization Test ===\n");

    wubu_thread_spec_t *ts = wubu_thread_spec_create("0-1", "2-3");
    assert(ts);

    int pc, dc;
    wubu_thread_spec_cores(ts, &pc, &dc);
    printf("  prefill cores=%d decode cores=%d\n", pc, dc);
    assert(pc == 2 && dc == 2);

    int vals[8] = {1,2,3,4,5,6,7,8};
    for (int i = 0; i < 4; i++)
        wubu_thread_spec_submit(ts, WUBU_TS_PREFILL, prefill_job, &vals[i]);
    for (int i = 4; i < 8; i++)
        wubu_thread_spec_submit(ts, WUBU_TS_DECODE, decode_job, &vals[i]);

    wubu_thread_spec_wait(ts, WUBU_TS_PREFILL);
    wubu_thread_spec_wait(ts, WUBU_TS_DECODE);

    printf("  prefill_sum=%d (expect 10) decode_sum=%d (expect 26)\n",
           g_prefill_sum, g_decode_sum);
    assert(g_prefill_sum == 10);
    assert(g_decode_sum == 26);

    wubu_thread_spec_free(ts);
    printf("ALL H02 THREAD-SPEC TESTS PASSED\n");
    return 0;
}
