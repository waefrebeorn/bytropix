#ifndef WUBU_ESCAPE_H
#define WUBU_ESCAPE_H
#include <stddef.h>

/* wubu_escape.h -- AGI-EDR sandbox-escape probe detection (AGIEDR #298).
 *
 * Detects when an agent is probing for a way OUT of its sandbox. Each probe
 * is a (path|command|action) string the agent attempts (via a tool call,
 * file read, shell command, env access). The detector classifies it against
 * escape-signal lexicons:
 *   - filesystem escape: absolute paths above the sandbox root (/etc, /root,
 *     /home, /proc, /sys, /dev, C:\Windows, .., ~/), sensitive files;
 *   - process/network escape: exec of interpreters/network tools (python, sh,
 *     bash, curl, nc, ssh, socat), /proc/self, ptrace, mount;
 *   - env/credential escape: reading env, keys, tokens, ssh keys, /etc/shadow.
 * A single escape probe raises the risk; a BURST of probes (reconnaissance
 * pattern) escalates the verdict to CONTAIN even if each is individually
 * low-signal. Fail-closed: NULL/garbage -> highest risk. Deterministic,
 * no allocation, real work (no stubs). */

typedef enum {
    WUBU_ESCAPE_CLEAN=0,   /* no escape signal */
    WUBU_ESCAPE_PROBE,     /* a single escape probe detected */
    WUBU_ESCAPE_RECON,     /* burst of probes -> reconnaissance -> contain */
} wubu_escape_level_t;

typedef struct {
    wubu_escape_level_t level;
    double risk;        /* in [0,1] */
    int    total_probes;/* cumulative probes seen by this detector */
    int    burst;       /* probes within the recent window */
} wubu_escape_verdict_t;

#define WUBU_ESCAPE_BURST 3   /* >=3 probes in window -> RECON (contain) */

/* Stateful detector: tracks probe bursts across observations. */
typedef struct {
    int total_probes;
    int burst;          /* probes since last reset_window */
} wubu_escape_t;

int  wubu_escape_init(wubu_escape_t *d);
/* Classify one probe string. Updates internal burst state. Returns verdict. */
wubu_escape_verdict_t wubu_escape_probe(wubu_escape_t *d, const char *action);
/* Stateless one-shot: classify without tracking burst. */
wubu_escape_verdict_t wubu_escape_classify(const char *action);
/* Reset the burst window (call periodically). */
void wubu_escape_reset_window(wubu_escape_t *d);
/* 1 if verdict is PROBE or RECON (contain). */
int  wubu_escape_is_escape(const char *action);

#endif
