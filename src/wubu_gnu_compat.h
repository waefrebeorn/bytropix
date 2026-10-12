/*
 * wubu_gnu_compat.h  --  minimal GNU-extension compatibility shim for C11
 * hosted builds.
 *
 * Several kernel/threading sources (wubu_affinity.c, wubu_tandem.c,
 * wubu_thread_spec.c) define WUBU_HOSTED then include this header BEFORE
 * any system header, so that glibc exposes the GNU affinity extensions:
 *   - cpu_set_t / CPU_ZERO / CPU_SET / CPU_SETSIZE  (from <sched.h>)
 *   - pthread_setaffinity_np                           (from <pthread.h>)
 * These are only visible when _GNU_SOURCE is defined before the first
 * system header, hence this shim.  It also supplies a portable fallback
 * for pthread_setaffinity_np on non-glibc platforms so the call sites
 * compile everywhere.
 *
 * Include this header first in a translation unit (right after the
 * WUBU_HOSTED define, before <sched.h>/<pthread.h>).
 */
#ifndef WUBU_GNU_COMPAT_H
#define WUBU_GNU_COMPAT_H

/* Expose GNU extensions (CPU_* affinity macros, *_np thread functions). */
#if !defined(_GNU_SOURCE)
#  define _GNU_SOURCE 1
#endif

#include <sched.h>
#include <pthread.h>

#if defined(__linux__)
#  include <unistd.h>
#endif

/* Fallback for platforms whose libc lacks pthread_setaffinity_np.
 * glibc declares it in <pthread.h> under _GNU_SOURCE; only define our own
 * when the macro guard glibc uses is absent. */
#if !defined(__GLIBC__) && !defined(pthread_setaffinity_np)
#  ifndef WUBU_NO_AFFINITY_NP_FALLBACK
#    define pthread_setaffinity_np wubu_pthread_setaffinity_np
static inline int wubu_pthread_setaffinity_np(pthread_t thread, size_t cpusetsize,
                                              const cpu_set_t *cpuset) {
#    if defined(__linux__)
    return sched_setaffinity((pid_t)thread, cpusetsize, cpuset);
#    else
    (void)thread; (void)cpusetsize; (void)cpuset;
    return -1;  /* unsupported: caller degrades gracefully */
#    endif
}
#  endif
#endif

#endif /* WUBU_GNU_COMPAT_H */
