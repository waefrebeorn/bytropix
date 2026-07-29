/*
 * wubu_spawn.c — shell-free external program launcher.
 * C11, self-contained (no god headers).
 *
 * Replaces popen()/system() with fork+execvp so the caller never
 * goes through an intermediate shell (no injection, no SIGPIPE races,
 * exit-code is the child's wait status, no intermediate fd sharing).
 *
 * This module is the single bridge between wubuwizard and the host OS
 * for any subprocess that UTF-8 output into an internal buffer.
 */
#define _POSIX_C_SOURCE 200809L
#include "wubu_spawn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <errno.h>

int wubu_spawn_capture(const char *file, char *const argv[],
                       char *out_buf, size_t out_cap, int *out_exit)
{
    if (!file || !argv || !out_buf || !out_cap || !out_exit) return -1;
    *out_exit = -1;
    int pipefd[2];
    if (pipe(pipefd) != 0) return -1;

    pid_t pid = fork();
    if (pid < 0) { close(pipefd[0]); close(pipefd[1]); return -1; }
    if (pid == 0) {
        /* child */
        close(pipefd[0]);
        if (pipefd[1] != STDOUT_FILENO) {
            dup2(pipefd[1], STDOUT_FILENO);
            close(pipefd[1]);
        }
        execvp(file, argv);
        _exit(127);
    }
    /* parent */
    close(pipefd[1]);
    size_t pos = 0;
    ssize_t r;
    while (pos + 1 < out_cap &&
           (r = read(pipefd[0], out_buf + pos, out_cap - pos - 1)) > 0) {
        pos += (size_t)r;
    }
    out_buf[pos] = '\0';
    close(pipefd[0]);

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return -1;
    if (WIFEXITED(status)) *out_exit = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) *out_exit = 128 + WTERMSIG(status);
    else *out_exit = -1;
    return (int)pos;
}

int wubu_spawn_wait(const char *file, char *const argv[], bool silent)
{
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        if (silent) {
            int fd = open("/dev/null", O_WRONLY);
            if (fd >= 0) { dup2(fd, STDOUT_FILENO); dup2(fd, STDERR_FILENO); close(fd); }
        }
        execvp(file, argv);
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
}
