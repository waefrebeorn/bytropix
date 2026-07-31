#ifndef WUBU_EDR_H
#define WUBU_EDR_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * wubu_edr.h — Lightweight Endpoint Detection & Response for WuBuOS.
 *
 * Self-hosted, dependency-free. Monitors process behavior for anomalous
 * patterns (sudden memory growth, new threads, file writes, network).
 * Designed for "human-pointed compatibility" — the human reviews EDR
 * alerts, not automated blocking.
 *
 * Architecture:
 *  - Samples /proc/PID/status for memory growth rate
 *  - Samples /proc/PID/task/ for thread count
 *  - Records file writes via inotify (optional)
 *  - Logs to structured audit trail with UUIDv7 timestamps
 *
 * Triple-DA:
 *  Decision: self-hosted EDR, no cloud upload, no telemetry.
 *  Design:   sample-based (1 Hz by default), configurable.
 *  Robustness: graceful degradation if /proc not available (non-Linux).
 */

typedef struct {
    int pid;
    char uuid[37];       /* session UUIDv7 */
    int64_t start_ns;    /* monotonic start time */
    size_t last_rss_kb;  /* last RSS sample */
    int last_threads;    /* last thread count */
    size_t peak_rss_kb;
    int max_threads;
    int alert_flags;     /* bitmask of ALERT_* */
} wubu_edr_session_t;

#define WUBU_EDR_ALERT_RSS_GROW   0x01
#define WUBU_EDR_ALERT_THREAD_SPROUT  0x02
#define WUBU_EDR_ALERT_FD_LEAK    0x04

typedef struct {
    int sampling_hz;
    size_t rss_growth_threshold_kb;   /* alert if RSS grows by this much in 1s */
    int thread_sprouting_threshold;   /* alert if thread count jumps by this */
    int fd_leak_threshold;            /* alert if open FDs grow by this in 1s */
    int fd_warn_threshold;            /* warn-level FD count */
} wubu_edr_config_t;

/* Initialize EDR system. Returns 0 on success. */
int wubu_edr_init(const wubu_edr_config_t *cfg);

/* Register a process for monitoring by PID. Returns session handle. */
wubu_edr_session_t *wubu_edr_register(int pid);

/* Sample all registered processes. Returns number of alerts. */
int wubu_edr_sample(void);

/* Get current RSS (KB) for a PID. Returns 0 on error. */
size_t wubu_edr_get_rss_kb(int pid);

/* Get current thread count for a PID. Returns -1 on error. */
int wubu_edr_get_thread_count(int pid);

/* Get open FD count for a PID. Returns -1 on error. */
int wubu_edr_get_fd_count(int pid);

/* Retrieve the last alert (structured string). Caller must free if duped. */
const char *wubu_edr_last_alert(void);

/* Shut down EDR system. */
void wubu_edr_shutdown(void);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_EDR_H */
