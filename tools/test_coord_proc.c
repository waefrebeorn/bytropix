/* test_coord_proc.c — AGENT #336 cross-process promotion lock.
 *
 * Two forked agents race for the promotion path. The in-process lock table
 * cannot arbitrate them (separate address spaces), so the winner is decided by
 * an exclusive flock. Asserts:
 *   - exactly one child acquires the lock
 *   - only the winner promotes (wubu_promote_accepted == 1)
 *   - the loser's promotion is rejected
 *   - the lock is free again after release
 */
#define _POSIX_C_SOURCE 200809L
#include "wubu_coord.h"
#include "wubu_promote.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/wait.h>

static int fails = 0, checks = 0;
#define CHECK(c, m) do { checks++; if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else printf("  [PASS] %s\n", m); } while (0)

typedef struct { volatile int start; volatile int accepted; volatile int rejected; } shared_t;

int main(void) {
    const char *lock = "test_coord_proc.lock";
    unlink(lock);

    shared_t *sh = mmap(NULL, sizeof(shared_t), PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (sh == MAP_FAILED) { printf("mmap failed\n"); return 1; }
    memset((void *)sh, 0, sizeof(*sh));

    pid_t pids[2];
    for (int i = 0; i < 2; i++) {
        pids[i] = fork();
        if (pids[i] == 0) {
            while (!sh->start) { }           /* barrier: both race together */
            int fd = wubu_proc_lock_acquire(lock);
            if (fd < 0) {
                __sync_fetch_and_add(&sh->rejected, 1);
                _exit(3);
            }
            wubu_promote_t p;
            wubu_promote_init(&p);
            int idx = wubu_promote_propose(&p, "mut-race", "agent", 7);
            uint64_t span = wubu_promote_gate(&p, idx, "verifier", 1);
            if (span > 0 && wubu_promote_accepted(&p) == 1)
                __sync_fetch_and_add(&sh->accepted, 1);
            struct timespec ts = {0, 150000000L};
            nanosleep(&ts, NULL);            /* hold long enough to overlap */
            wubu_proc_lock_release(fd);
            _exit(0);
        }
    }

    sh->start = 1;
    int code[2] = {-1, -1};
    for (int i = 0; i < 2; i++) {
        int st = 0;
        waitpid(pids[i], &st, 0);
        code[i] = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
    }

    int winners = (code[0] == 0) + (code[1] == 0);
    int losers  = (code[0] == 3) + (code[1] == 3);
    printf("  exit codes: %d %d | accepted=%d rejected=%d\n",
           code[0], code[1], sh->accepted, sh->rejected);

    CHECK(winners == 1, "exactly one process wins the lock");
    CHECK(losers == 1, "the other process is refused");
    CHECK(sh->accepted == 1, "accepted promotions == 1 (never exceeds 1)");
    CHECK(sh->rejected == 1, "the loser's promote is rejected");

    int fd = wubu_proc_lock_acquire(lock);
    CHECK(fd >= 0, "lock is free again after release");
    if (fd >= 0) wubu_proc_lock_release(fd);

    CHECK(wubu_proc_lock_acquire(NULL) == -1, "NULL path -> -1");
    CHECK(wubu_proc_lock_release(-1) == -1, "bad fd -> -1");

    unlink(lock);
    munmap((void *)sh, sizeof(*sh));
    printf("\n=== test_coord_proc: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}