/*
 * wubu_edr.c — Lightweight Endpoint Detection & Response for WuBuOS.
 *
 * Self-hosted, dependency-free. Samples /proc for anomalies:
 *   - Memory growth rate (RSS delta)
 *   - Thread sprouting (task count delta)
 *   - FD leak (open file descriptor growth)
 *
 * Triple-DA:
 *   Decision: no external deps — read /proc directly.
 *   Design:   linked list of sessions, O(n) sample. Config-driven thresholds.
 *   Robustness: every /proc read is guarded with error handling.
 */

#include "wubu_edr.h"
#include "wubu_uuid.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

static wubu_edr_config_t g_config;
static wubu_edr_session_t *g_sessions = NULL;
static wubu_edr_session_t *g_sessions_tail = NULL;
static char g_last_alert[512];
static int g_initialized = 0;

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

static wubu_edr_session_t *find_session(int pid) {
    wubu_edr_node_t *n = (wubu_edr_node_t *)g_sessions;
    while (n) {
        if (n->base.pid == pid) return &n->base;
        n = n->next;
    }
    return NULL;
}

/* Read /proc/PID/status for VmRSS */
size_t wubu_edr_get_rss_kb(int pid) {
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/status", pid);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[256];
    size_t rss = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%zu", &rss);
            break;
        }
    }
    fclose(f);
    return rss;
}

/* Count entries in /proc/PID/task/ */
int wubu_edr_get_thread_count(int pid) {
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/task", pid);
    DIR *d = opendir(path);
    if (!d) return -1;
    int count = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        if (de->d_name[0] >= '0' && de->d_name[0] <= '9')
            count++;
    }
    closedir(d);
    return count;
}

/* Count open FDs via /proc/PID/fd/ */
int wubu_edr_get_fd_count(int pid) {
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/fd", pid);
    DIR *d = opendir(path);
    if (!d) return -1;
    int count = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        if (de->d_name[0] >= '0' && de->d_name[0] <= '9')
            count++;
    }
    closedir(d);
    return count;
}

int wubu_edr_init(const wubu_edr_config_t *cfg) {
    if (g_initialized) return 0;
    if (cfg) {
        g_config = *cfg;
    } else {
        g_config.sampling_hz = 1;
        g_config.rss_growth_threshold_kb = 50000;  /* 50 MB/s */
        g_config.thread_sprouting_threshold = 50;
        g_config.fd_leak_threshold = 100;
        g_config.fd_warn_threshold = 500;
    }
    g_sessions = NULL;
    g_sessions_tail = NULL;
    g_last_alert[0] = '\0';
    g_initialized = 1;
    return 0;
}

/* Link list via next pointer at start of struct (reuse memory safely) */
typedef struct wubu_edr_node {
    wubu_edr_session_t base;
    struct wubu_edr_node *next;
    size_t prev_rss;
    int prev_threads;
    int prev_fds;
} wubu_edr_node_t;

wubu_edr_session_t *wubu_edr_register(int pid) {
    if (!g_initialized) return NULL;

    /* Check duplicates */
    wubu_edr_node_t *n = (wubu_edr_node_t *)g_sessions;
    while (n) {
        if (n->base.pid == pid) return &n->base;
        n = n->next;
    }

    wubu_edr_node_t *node = (wubu_edr_node_t *)calloc(1, sizeof(wubu_edr_node_t));
    if (!node) return NULL;
    node->base.pid = pid;
    wubu_uuid_v7(node->base.uuid, sizeof(node->base.uuid));
    node->base.start_ns = now_ns();
    node->base.last_rss_kb = wubu_edr_get_rss_kb(pid);
    node->base.last_threads = wubu_edr_get_thread_count(pid);
    node->base.peak_rss_kb = node->base.last_rss_kb;
    node->base.max_threads = node->base.last_threads;
    node->prev_rss = node->base.last_rss_kb;
    node->prev_threads = node->base.last_threads;
    node->prev_fds = wubu_edr_get_fd_count(pid);
    node->next = NULL;

    /* Link into list */
    if (!g_sessions) {
        g_sessions = &node->base;
        g_sessions_tail = node;
    } else {
        /* Find last node */
        wubu_edr_node_t *cur = (wubu_edr_node_t *)g_sessions;
        while (cur->next) cur = cur->next;
        cur->next = node;
    }
    return &node->base;
}

int wubu_edr_sample(void) {
    if (!g_initialized || !g_sessions) return 0;
    int alert_count = 0;
    wubu_edr_node_t *n = (wubu_edr_node_t *)g_sessions;
    while (n) {
        size_t cur_rss = wubu_edr_get_rss_kb(n->base.pid);
        int cur_threads = wubu_edr_get_thread_count(n->base.pid);
        int cur_fds = wubu_edr_get_fd_count(n->base.pid);
        int alerts = 0;

        if (cur_rss > n->base.peak_rss_kb) n->base.peak_rss_kb = cur_rss;
        if (cur_threads > n->base.max_threads) n->base.max_threads = cur_threads;

        size_t rss_delta = (cur_rss > n->prev_rss) ? cur_rss - n->prev_rss : 0;
        if (rss_delta > g_config.rss_growth_threshold_kb) {
            alerts |= WUBU_EDR_ALERT_RSS_GROW;
            snprintf(g_last_alert, sizeof(g_last_alert),
                     "EDR ALERT [%s] pid=%d RSS growth +%lu KB/s (threshold %lu KB/s)",
                     n->base.uuid, n->base.pid, (unsigned long)rss_delta,
                     (unsigned long)g_config.rss_growth_threshold_kb);
        }

        if (cur_threads > 0 && n->prev_threads > 0) {
            int thread_delta = cur_threads - n->prev_threads;
            if (thread_delta > g_config.thread_sprouting_threshold) {
                alerts |= WUBU_EDR_ALERT_THREAD_SPROUT;
                snprintf(g_last_alert, sizeof(g_last_alert),
                         "EDR ALERT [%s] pid=%d threads +%d (threshold %d)",
                         n->base.uuid, n->base.pid, thread_delta,
                         g_config.thread_sprouting_threshold);
            }
        }

        if (cur_fds > 0 && n->prev_fds > 0) {
            int fd_delta = cur_fds - n->prev_fds;
            if (fd_delta > g_config.fd_leak_threshold) {
                alerts |= WUBU_EDR_ALERT_FD_LEAK;
                snprintf(g_last_alert, sizeof(g_last_alert),
                         "EDR ALERT [%s] pid=%d FDs +%d (threshold %d)",
                         n->base.uuid, n->base.pid, fd_delta,
                         g_config.fd_leak_threshold);
            }
        }

        n->base.alert_flags = (int)alerts;
        n->base.last_rss_kb = cur_rss;
        n->base.last_threads = cur_threads;
        n->prev_rss = cur_rss;
        n->prev_threads = cur_threads;
        n->prev_fds = cur_fds;

        if (alerts) alert_count++;
        n = n->next;
    }
    return alert_count;
}

const char *wubu_edr_last_alert(void) {
    return g_last_alert[0] ? g_last_alert : NULL;
}

void wubu_edr_shutdown(void) {
    if (!g_initialized) return;
    wubu_edr_node_t *n = (wubu_edr_node_t *)g_sessions;
    while (n) {
        wubu_edr_node_t *next = n->next;
        free(n);
        n = next;
    }
    g_sessions = NULL;
    g_initialized = 0;
}
