/* wubu_cigate.c -- STACK #368 three-repo integration CI gate.
 * See wubu_cigate.h. Real, deterministic implementation using fork+exec with
 * captured output. */

#include "wubu_cigate.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

int wubu_ci_init(wubu_ci_report_t *r)
{
    if (!r) return -1;
    memset(r, 0, sizeof(*r));
    r->n_legs = 0;
    return 0;
}

int wubu_ci_run_leg(wubu_ci_leg_t *leg, const char *name, const char *cmd)
{
    if (!leg || !name || !cmd) return -1;
    memset(leg, 0, sizeof(*leg));
    strncpy(leg->name, name, sizeof(leg->name) - 1);
    strncpy(leg->cmd, cmd, WUBU_CI_CMD_MAX - 1);

    int pipefd[2];
    if (pipe(pipefd) != 0) { leg->exit_code = -1; leg->passed = 0; return 0; }

    pid_t pid = fork();
    if (pid < 0) { close(pipefd[0]); close(pipefd[1]); leg->exit_code = -1; return 0; }

    if (pid == 0) {
        /* child: redirect stdout+stderr into the pipe, exec the command */
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[0]);
        close(pipefd[1]);
        execl("/bin/sh", "sh", "-c", cmd, (char *)NULL);
        _exit(127);
    }

    /* parent: read the pipe, then wait. Keep the last (keep) bytes. */
    close(pipefd[1]);
    size_t total = 0;
    char tmp[4096];
    size_t keep = WUBU_CI_OUT_MAX - 1;
    ssize_t n;
    while ((n = read(pipefd[0], tmp, sizeof tmp)) > 0) {
        size_t chunk = (size_t)n;
        if (chunk >= keep) {
            /* this chunk alone fills the buffer: keep its tail */
            memcpy(leg->output, tmp + (chunk - keep), keep);
            total = keep;
            leg->output[keep] = '\0';
        } else if (total + chunk <= keep) {
            memcpy(leg->output + total, tmp, chunk);
            total += chunk;
            leg->output[total] = '\0';
        } else {
            /* shift left by the overflow, then append */
            size_t overflow = total + chunk - keep;
            memmove(leg->output, leg->output + overflow, total - overflow);
            memcpy(leg->output + (total - overflow), tmp, chunk);
            total = keep;
            leg->output[keep] = '\0';
        }
    }
    close(pipefd[0]);

    int status = 0;
    waitpid(pid, &status, 0);
    leg->ran = 1;
    if (WIFEXITED(status)) {
        leg->exit_code = WEXITSTATUS(status);
        leg->passed = (leg->exit_code == 0);
    } else {
        leg->exit_code = -1;
        leg->passed = 0;
    }
    return 0;
}

int wubu_ci_finalize(wubu_ci_report_t *r)
{
    if (!r) return -1;
    r->n_passed = 0;
    r->n_failed = 0;
    for (int i = 0; i < r->n_legs; i++) {
        if (r->legs[i].passed) r->n_passed++;
        else r->n_failed++;
    }
    r->verdict = (r->n_failed == 0 && r->n_legs > 0) ? WUBU_CI_GREEN : WUBU_CI_RED;
    return 0;
}

void wubu_ci_summary(const wubu_ci_report_t *r, char *buf, size_t cap)
{
    if (!r || !buf || cap == 0) return;
    snprintf(buf, cap, "CI %s: %d/%d legs green",
             r->verdict == WUBU_CI_GREEN ? "GREEN" : "RED",
             r->n_passed, r->n_legs);
}
