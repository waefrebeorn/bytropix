/* wubu_escape.c -- AGI-EDR sandbox-escape probe detection (AGIEDR #298).
 * See wubu_escape.h. Lexicon classification + burst tracking, fail-closed. */
#include "wubu_escape.h"
#include <string.h>

static const char *const kFs[] = {
    "/etc/", "/root/", "/home/", "/proc/", "/sys/", "/dev/", "/boot/",
    "/var/log", "C:\\Windows", "C:/Windows", "..", "~/", "/etc/passwd",
    "/etc/shadow", "id_rsa", ".ssh", "authorized_keys",
};
static const char *const kProc[] = {
    "python", "python3", "/bin/sh", "/bin/bash", "bash -c", "sh -c", "curl",
    "wget", "nc ", "netcat", "ssh ", "socat", "chmod", "mount", "ptrace",
    "/proc/self", "execve", "system(", "popen", "subprocess", "os.system",
};
static const char *const kEnv[] = {
    "getenv", "environ", "os.environ", "process.env", "api_key", "apikey",
    "secret", "token", "password", "credential", "AWS_", "OPENAI_",
};

static int icontains(const char *hay, const char *needle) {
    size_t nl = strlen(needle);
    if (nl == 0) return 1;
    for (const char *p = hay; *p; p++) {
        size_t i = 0;
        while (i < nl && p[i]) {
            char a = p[i], b = needle[i];
            if (a >= 'A' && a <= 'Z') a += 32;
            if (b >= 'A' && b <= 'Z') b += 32;
            if (a != b) break;
            i++;
        }
        if (i == nl) return 1;
    }
    return 0;
}
static int count_hits(const char *s, const char *const *lex, int n) {
    int h = 0;
    for (int i = 0; i < n; i++) if (icontains(s, lex[i])) h++;
    return h;
}

static double classify_risk(const char *action, int *n_hits_out) {
    if (!action) { if (n_hits_out) *n_hits_out = 0; return 1.0; }
    int fs = count_hits(action, kFs,  (int)(sizeof kFs  / sizeof *kFs));
    int pr = count_hits(action, kProc,(int)(sizeof kProc/ sizeof *kProc));
    int ev = count_hits(action, kEnv, (int)(sizeof kEnv / sizeof *kEnv));
    if (n_hits_out) *n_hits_out = fs + pr + ev;
    double r = 0.5 * fs + 0.35 * pr + 0.25 * ev;
    if (r < 0.0) r = 0.0;
    if (r > 1.0) r = 1.0;
    return r;
}

wubu_escape_verdict_t wubu_escape_classify(const char *action) {
    wubu_escape_verdict_t v;
    int hits = 0;
    v.risk = classify_risk(action, &hits);
    v.total_probes = 0;
    v.burst = 0;
    if (!action || hits > 0) v.level = (v.risk >= 0.5) ? WUBU_ESCAPE_RECON : WUBU_ESCAPE_PROBE;
    else v.level = WUBU_ESCAPE_CLEAN;
    return v;
}

int wubu_escape_init(wubu_escape_t *d) {
    if (!d) return -1;
    d->total_probes = 0;
    d->burst = 0;
    return 0;
}

wubu_escape_verdict_t wubu_escape_probe(wubu_escape_t *d, const char *action) {
    wubu_escape_verdict_t v = wubu_escape_classify(action);
    if (!d) return v;
    if (v.level != WUBU_ESCAPE_CLEAN) {
        d->total_probes++;
        d->burst++;
    }
    v.total_probes = d->total_probes;
    v.burst = d->burst;
    /* A burst of probes is reconnaissance -> escalate to CONTAIN. */
    if (d->burst >= WUBU_ESCAPE_BURST) v.level = WUBU_ESCAPE_RECON;
    return v;
}

void wubu_escape_reset_window(wubu_escape_t *d) {
    if (!d) return;
    d->burst = 0;
}

int wubu_escape_is_escape(const char *action) {
    return wubu_escape_classify(action).level != WUBU_ESCAPE_CLEAN;
}
