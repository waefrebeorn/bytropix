/* wubu_harness2.c -- Harness^2 loop in C11. See include/wubu_harness2.h.
 *
 * Deliberately no model dependency: composition is decided by check-command
 * exit status, which makes the loop testable without spending tokens.
 */
#include "wubu_harness2.h"

#include <errno.h>
#include <time.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

struct wubu_h2_harness {
    int count;
    char rules[WUBU_H2_MAX_RULES][WUBU_H2_RULE_LEN];
};

/* -- harness ---------------------------------------------------------- */

wubu_h2_harness_t *wubu_h2_harness_new(void) {
    wubu_h2_harness_t *h = (wubu_h2_harness_t *)calloc(1, sizeof(*h));
    return h;
}

void wubu_h2_harness_free(wubu_h2_harness_t *h) { free(h); }

int wubu_h2_harness_has_rule(const wubu_h2_harness_t *h, const char *rule) {
    if (!h || !rule) return 0;
    for (int i = 0; i < h->count; i++)
        if (strcmp(h->rules[i], rule) == 0) return 1;
    return 0;
}

int wubu_h2_harness_add_rule(wubu_h2_harness_t *h, const char *rule) {
    if (!h || !rule || !rule[0]) return -1;
    if (strlen(rule) >= WUBU_H2_RULE_LEN) return -1;
    if (h->count >= WUBU_H2_MAX_RULES) return -1;
    if (wubu_h2_harness_has_rule(h, rule)) return -1;
    strcpy(h->rules[h->count], rule);
    h->count++;
    return 0;
}

int wubu_h2_harness_rule_count(const wubu_h2_harness_t *h) {
    return h ? h->count : 0;
}

const char *wubu_h2_harness_rule(const wubu_h2_harness_t *h, int i) {
    if (!h || i < 0 || i >= h->count) return NULL;
    return h->rules[i];
}

int wubu_h2_harness_render(const wubu_h2_harness_t *h, char *out, size_t cap) {
    if (!out || cap == 0) return -1;
    if (!h) return -1;
    int n = snprintf(out, cap,
                     "# WuBu AGI harness\n\n"
                     "You are the executor inside the WuBu AGI loop. Work in the given workspace.\n\n"
                     "- Prefer measured facts over narration.\n"
                     "- A task is complete only when its check command exits 0.\n");
    if (n < 0 || (size_t)n >= cap) return -1;
    size_t off = (size_t)n;
    if (h->count > 0) {
        int m = snprintf(out + off, cap - off, "\n## Additional rules\n");
        if (m < 0 || (size_t)m >= cap - off) return -1;
        off += (size_t)m;
        for (int i = 0; i < h->count; i++) {
            m = snprintf(out + off, cap - off, "- %s\n", h->rules[i]);
            if (m < 0 || (size_t)m >= cap - off) return -1;
            off += (size_t)m;
        }
    }
    return (int)off;
}

int wubu_h2_harness_stage(const wubu_h2_harness_t *h, const char *dir) {
    if (!h || !dir || !dir[0]) return -1;
    char buf[8192];
    int n = wubu_h2_harness_render(h, buf, sizeof buf);
    if (n < 0) return -1;

    if (mkdir(dir, 0777) != 0 && errno != EEXIST) return -1;

    char path[WUBU_H2_PATH_LEN];
    if (snprintf(path, sizeof path, "%s/AGENTS.md", dir) >= (int)sizeof path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t wrote = fwrite(buf, 1, (size_t)n, f);
    int rc = (wrote == (size_t)n) ? 0 : -1;
    if (fclose(f) != 0) rc = -1;
    return rc;
}

/* -- probes ------------------------------------------------------------ */

int wubu_h2_probe_run(wubu_h2_probe_t *out, int probe_index,
                      const wubu_h2_harness_t *h, const char *check_cmd,
                      int timeout_s) {
    if (!out || !check_cmd || !check_cmd[0] || timeout_s <= 0) return -1;
    memset(out, 0, sizeof(*out));
    out->probe_index = probe_index;
    out->rules_count = h ? h->count : 0;
    out->harness = h;

    int fds[2];
    if (pipe(fds) != 0) return -1;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    pid_t pid = fork();
    if (pid < 0) { close(fds[0]); close(fds[1]); return -1; }
    if (pid == 0) {
        /* Own process group: a check that backgrounds work must not outlive the
         * timeout, and SIGALRM only reaches the shell. */
        setpgid(0, 0);
        close(fds[0]);
        dup2(fds[1], STDOUT_FILENO);
        dup2(fds[1], STDERR_FILENO);
        close(fds[1]);
        /* A hung check must not wedge the loop; SIGALRM's default action ends
         * the child and the parent reports 124. */
        alarm((unsigned)timeout_s);
        execl("/bin/sh", "sh", "-c", check_cmd, (char *)NULL);
        _exit(127);
    }

    close(fds[1]);
    /* Keep the LAST bytes printed: a check's conclusion is at the end. */
    char buf[4096];
    size_t have = 0;
    ssize_t rd;
    while ((rd = read(fds[0], buf + have, sizeof buf - have - 1)) > 0) {
        have += (size_t)rd;
        if (have == sizeof buf - 1) {
            memmove(buf, buf + have - 256, 256);
            have = 256;
        }
    }
    close(fds[0]);
    buf[have] = 0;
    size_t keep = have < sizeof out->check_tail - 1 ? have : sizeof out->check_tail - 1;
    memcpy(out->check_tail, buf + (have - keep), keep);
    out->check_tail[keep] = 0;

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) return -1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    out->elapsed_ms = (long)((t1.tv_sec - t0.tv_sec) * 1000 +
                             (t1.tv_nsec - t0.tv_nsec) / 1000000);

    if (WIFEXITED(status)) out->check_rc = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) {
        int sig = WTERMSIG(status);
        out->check_rc = (sig == SIGALRM) ? 124 : 128 + sig;
        if (sig == SIGALRM) kill(-pid, SIGKILL);  /* take the whole group down */
    } else out->check_rc = -1;
    return out->check_rc;
}

int wubu_h2_compose(const wubu_h2_probe_t *probes, int n) {
    if (!probes || n <= 0) return -1;
    for (int i = 0; i < n; i++)
        if (probes[i].check_rc == 0) return probes[i].probe_index;
    return -1;
}

int wubu_h2_adopt(const wubu_h2_probe_t *probes, int n, int winner,
                  char out[][WUBU_H2_RULE_LEN], int cap) {
    if (!probes || n <= 0 || !out || cap <= 0) return -1;
    if (winner < 0) return 0;
    for (int i = 0; i < n; i++) {
        if (probes[i].probe_index != winner) continue;
        const wubu_h2_harness_t *h = probes[i].harness;
        if (!h) return -1;
        int total = wubu_h2_harness_rule_count(h);
        int k = total < cap ? total : cap;
        for (int j = 0; j < k; j++) {
            const char *r = wubu_h2_harness_rule(h, j);
            if (!r) return -1;
            strncpy(out[j], r, WUBU_H2_RULE_LEN - 1);
            out[j][WUBU_H2_RULE_LEN - 1] = 0;
        }
        return k;
    }
    return -1;
}

wubu_h2_verdict_t wubu_h2_verdict(const wubu_h2_probe_t *ref,
                                  const wubu_h2_probe_t *cand) {
    if (!ref || !cand) return WUBU_H2_CONFOUNDED;
    /* Several edits at once cannot be attributed to any one of them. */
    if (cand->rules_count > 1 && ref->rules_count != cand->rules_count)
        return WUBU_H2_CONFOUNDED;
    int rp = (ref->check_rc == 0), cp = (cand->check_rc == 0);
    if (!rp && cp) return WUBU_H2_HELPED;
    if (rp && !cp) return WUBU_H2_HURT;
    return WUBU_H2_NO_EFFECT;
}

/* -- ledger ------------------------------------------------------------ */

/* Escape a string into JSON. Returns the written length, or -1 when it does not
 * fit. Refusing quotes made ordinary lesson text unwritable; escaping keeps the
 * line parseable instead. */
static int json_escape(const char *s, char *out, size_t cap) {
    if (!s || !out || cap == 0) return -1;
    size_t o = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        const char *esc = NULL;
        char ubuf[8];
        switch (*p) {
            case '"':  esc = "\\\""; break;
            case '\\': esc = "\\\\"; break;
            case '\n': esc = "\\n"; break;
            case '\r': esc = "\\r"; break;
            case '\t': esc = "\\t"; break;
            default:
                if (*p < 0x20) { snprintf(ubuf, sizeof ubuf, "\\u%04x", *p); esc = ubuf; }
                break;
        }
        size_t l = esc ? strlen(esc) : 1;
        if (o + l + 1 > cap) return -1;
        if (esc) memcpy(out + o, esc, l);
        else out[o] = (char)*p;
        o += l;
    }
    out[o] = 0;
    return (int)o;
}

int wubu_h2_ledger_append(const char *path, const char *task_id,
                          const wubu_h2_probe_t *probes, int n, int adopted_probe) {
    if (!path || !path[0] || !task_id) return -1;
    if (!probes || n <= 0 || n > WUBU_H2_MAX_PROBES) return -1;

    char esc[512];
    if (json_escape(task_id, esc, sizeof esc) < 0) return -1;

    FILE *f = fopen(path, "ab");
    if (!f) return -1;
    int rc = 0;
    if (fprintf(f, "{\"task\":\"%s\",\"probes\":[", esc) < 0) rc = -1;
    for (int i = 0; i < n && rc == 0; i++) {
        const char *v = probes[i].check_rc == 0 ? "pass" : "fail";
        wubu_h2_verdict_t vd = i == 0 ? WUBU_H2_NO_EFFECT
                                      : wubu_h2_verdict(&probes[0], &probes[i]);
        static const char *names[] = {"helped", "hurt", "no-effect", "confounded"};
        if (fprintf(f, "%s{\"i\":%d,\"rules\":%d,\"check_rc\":%d,\"result\":\"%s\","
                       "\"verdict\":\"%s\"}",
                    i ? "," : "", probes[i].probe_index, probes[i].rules_count,
                    probes[i].check_rc, v, names[(int)vd]) < 0) rc = -1;
    }
    if (rc == 0 && fprintf(f, "],\"adopted\":%d}\n", adopted_probe) < 0) rc = -1;
    if (fclose(f) != 0) rc = -1;
    return rc;
}

int wubu_h2_ledger_count(const char *path) {
    if (!path || !path[0]) return 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int count = 0, ch;
    int at_line_start = 1;
    while ((ch = fgetc(f)) != EOF) {
        if (at_line_start && ch == '{') count++;
        at_line_start = (ch == '\n');
    }
    fclose(f);
    return count;
}

/* -- executor argv ----------------------------------------------------- */

int wubu_h2_executor_argv(char *argv[], int cap,
                          const char *hermes_bin, const char *prompt,
                          const char *workspace, const char *model) {
    if (!argv || cap < 6 || !hermes_bin || !prompt || !workspace) return -1;
    int k = 0;
    argv[k++] = (char *)hermes_bin;
    argv[k++] = (char *)"-z";
    argv[k++] = (char *)prompt;
    argv[k++] = (char *)"--in";
    argv[k++] = (char *)workspace;
    if (model && model[0]) {
        if (k + 2 >= cap) return -1;
        argv[k++] = (char *)"-m";
        argv[k++] = (char *)model;
    }
    if (k + 1 >= cap) return -1;
    argv[k++] = (char *)"--pass-session-id";
    argv[k] = NULL;
    return k;
}