/*
 * wubu_spawn_win.c — Win32 implementation of the shell-free program launcher.
 * Mirrors wubu_spawn.c's contract (capture UTF-8 stdout into a buffer / wait)
 * using CreateProcess with an inherited pipe. No shell, no sh -c.
 * SPDX-License-Identifier: WaefreBeorn-UMV3
 */
#if defined(_WIN32)

#include "wubu_spawn.h"
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Build a Win32 command line from an argv array (argv[0] is the program). */
int wubu_spawn_win_capture(const char *file, char *const argv[],
                           char *out_buf, size_t out_cap, int *out_exit) {
    /* Reconstruct a command line. CreateProcess wants a single string. */
    size_t need = 0;
    for (int i = 0; argv[i]; i++) need += strlen(argv[i]) + 3;
    char *cmd = (char *)malloc(need + 16);
    if (!cmd) return -1;
    cmd[0] = '\0';
    for (int i = 0; argv[i]; i++) {
        if (i) strcat(cmd, " ");
        strcat(cmd, "\"");
        strcat(cmd, argv[i]);
        strcat(cmd, "\"");
    }

    SECURITY_ATTRIBUTES sa = { sizeof(sa), NULL, TRUE };
    HANDLE hRead = NULL, hWrite = NULL;
    if (!CreatePipe(&hRead, &hWrite, &sa, 0)) { free(cmd); return -1; }
    SetHandleInformation(hWrite, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);
    si.hStdOutput = hWrite;
    si.hStdError = hWrite;
    si.dwFlags = STARTF_USESTDHANDLES;

    if (!CreateProcessA(NULL, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        free(cmd); CloseHandle(hRead); CloseHandle(hWrite); return -1;
    }
    free(cmd);
    CloseHandle(hWrite);

    DWORD total = 0;
    char chunk[4096];
    DWORD got;
    while (ReadFile(hRead, chunk, sizeof(chunk), &got, NULL) && got > 0) {
        if (out_buf && total < out_cap) {
            size_t c = got;
            if (total + c > out_cap - 1) c = out_cap - 1 - total;
            memcpy(out_buf + total, chunk, c);
            total += (DWORD)c;
        }
    }
    if (out_buf) out_buf[total] = '\0';
    CloseHandle(hRead);

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD ec = 0;
    GetExitCodeProcess(pi.hProcess, &ec);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    if (out_exit) *out_exit = (int)ec;
    return (int)total;
}

int wubu_spawn_win_wait(const char *file, char *const argv[], int silent) {
    (void)silent;
    char *cmd = NULL;
    size_t need = 0;
    for (int i = 0; argv[i]; i++) need += strlen(argv[i]) + 3;
    cmd = (char *)malloc(need + 16);
    if (!cmd) return -1;
    cmd[0] = '\0';
    for (int i = 0; argv[i]; i++) {
        if (i) strcat(cmd, " ");
        strcat(cmd, "\""); strcat(cmd, argv[i]); strcat(cmd, "\"");
    }
    STARTUPINFOA si; PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si)); ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);
    int ok = CreateProcessA(NULL, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi);
    free(cmd);
    if (!ok) return -1;
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD ec = 0; GetExitCodeProcess(pi.hProcess, &ec);
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
    return (int)ec;
}

#endif /* _WIN32 */
