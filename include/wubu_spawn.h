#ifndef WUBU_SPAWN_H
#define WUBU_SPAWN_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * wubu_spawn_capture — fork+execvp `file argv[]`, capture stdout into buf.
 * Returns bytes captured, or -1 on error. Sets *out_exit to child exit code.
 * The child runs without a shell, so argv strings are NOT shell-expanded.
 */
int wubu_spawn_capture(const char *file, char *const argv[],
                       char *out_buf, size_t out_cap, int *out_exit);

/*
 * wubu_spawn_wait — fork+execvp `file argv[]`, wait for completion.
 * Returns child exit code (0..255) or -1 on fork/exec/wait error.
 * If silent, child's stdout+stderr are redirected to /dev/null.
 */
int wubu_spawn_wait(const char *file, char *const argv[], bool silent);

/* Win32 implementations (wubu_spawn_win.c) — used when _WIN32 is defined and
 * the POSIX fork/exec path is unavailable. Same contract as above. */
#if defined(_WIN32)
int wubu_spawn_win_capture(const char *file, char *const argv[],
                           char *out_buf, size_t out_cap, int *out_exit);
int wubu_spawn_win_wait(const char *file, char *const argv[], int silent);
#endif

#ifdef __cplusplus
}
#endif
#endif /* WUBU_SPAWN_H */
