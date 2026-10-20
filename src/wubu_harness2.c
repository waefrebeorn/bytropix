/* wubu_harness2.c -- Harness^2 loop in C11. See include/wubu_harness2.h.
 *
 * Deliberately no model dependency: composition is decided by check-command
 * exit status, which makes the loop testable without spending tokens.
 */
#include "wubu_harness2.h"

#include <errno.h>
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

    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        /* Child: timeout is enforced by alarm so a hung check cannot wedge the
         * loop; SIGALRM's default action terminates the child. */
        alarm((unsigned)timeout_s);
        execl("/bin/sh", "sh", "-c", check_cmd, (char *)NULL);
        _exit(127);
    }

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) return -1;
    }
    if (WIFEXITED(status)) out->check_rc = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) {
        int sig = WTERMSIG(status);
        out->check_rc = (sig == SIGALRM) ? 124 : 128 + sig;
    } else out->check_rc = -1;

    char tail[256] = {0};
    (void)tail;
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
    (void)probes;
    (void)n;
    /* The adopted set is the harness that produced `winner`, which the driver
     * records by rule list; with no rule store here, adoption is reported as
     * the count the probe ran with. Callers that need the text keep their own
     * rule pointers. */
    if (winner < 0) return 0;
    int k = 0;
    for (int i = 0; i < n && k < cap; i++) {
        if (probes[i].probe_index == winner) {
            int rc = probes[i].rules_count;
            return rc < 0 ? 0 : rc;
        }
    }
    return -1;
}

/* -- ledger ------------------------------------------------------------ */

static int json_simple(const char *s) {
    if (!s) return 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\' || *p == '\n' || *p == '\r' || *p < 0x20) return 0;
    }
    return 1;
}

int wubu_h2_ledger_append(const char *path, const char *task_id,
                          const wubu_h2_probe_t *probes, int n, int adopted_probe) {
    if (!path || !path[0] || !task_id || !json_simple(task_id)) return -1;
    if (!probes || n <= 0 || n > WUBU_H2_MAX_PROBES) return -1;

    FILE *f = fopen(path, "ab");
    if (!f) return -1;
    int rc = 0;
    if (fprintf(f, "{\"task\":\"%s\",\"probes\":[", task_id) < 0) rc = -1;
    for (int i = 0; i < n && rc == 0; i++) {
        if (fprintf(f, "%s{\"i\":%d,\"rules\":%d,\"check_rc\":%d}",
                    i ? "," : "", probes[i].probe_index, probes[i].rules_count,
                    probes[i].check_rc) < 0) rc = -1;
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