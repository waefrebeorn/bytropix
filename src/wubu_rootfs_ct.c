/* wubu_rootfs_ct.c -- RUNTIME #349/#350 rootfs container execution leg.
 * See wubu_rootfs_ct.h. Real, deterministic implementation: validates a
 * rootfs, loads the entrypoint ELF, and runs it in an isolated child. */

#define _GNU_SOURCE
#include "wubu_rootfs_ct.h"
#include "wubu_elfload.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#if defined(__linux__)
#include <sched.h>
#endif

const char *wubu_ct_strerror(int status)
{
    switch (status) {
    case WUBU_CT_OK:          return "ok";
    case WUBU_CT_ERR_ARGS:    return "bad arguments";
    case WUBU_CT_ERR_NOROOT:  return "rootfs missing or not a directory";
    case WUBU_CT_ERR_NOENTRY: return "entrypoint ELF missing under rootfs";
    case WUBU_CT_ERR_BADELF:  return "entrypoint is not a loadable ELF";
    case WUBU_CT_ERR_EXEC:    return "child failed to run";
    case WUBU_CT_ERR_STATUS:  return "child exited non-zero";
    default:                  return "unknown";
    }
}

int wubu_ct_init(wubu_container_t *ct, const char *name,
                 const char *rootfs, const char *entrypoint, int isolate)
{
    if (!ct || !name || !rootfs || !entrypoint) return WUBU_CT_ERR_ARGS;
    memset(ct, 0, sizeof(*ct));
    strncpy(ct->name, name, WUBU_CT_NAME_MAX - 1);
    strncpy(ct->rootfs, rootfs, WUBU_CT_PATH_MAX - 1);
    strncpy(ct->entrypoint, entrypoint, WUBU_CT_PATH_MAX - 1);
    ct->isolate = isolate;
    ct->argc = 0;
    return WUBU_CT_OK;
}

int wubu_ct_add_arg(wubu_container_t *ct, const char *arg)
{
    if (!ct || !arg) return WUBU_CT_ERR_ARGS;
    if (ct->argc >= 8) return WUBU_CT_ERR_ARGS;
    strncpy(ct->argv[ct->argc], arg, 255);
    ct->argv[ct->argc][255] = '\0';
    ct->argc++;
    return WUBU_CT_OK;
}

int wubu_ct_validate(const wubu_container_t *ct)
{
    if (!ct) return WUBU_CT_ERR_ARGS;
    struct stat st;
    if (stat(ct->rootfs, &st) != 0 || !S_ISDIR(st.st_mode))
        return WUBU_CT_ERR_NOROOT;
    /* entrypoint path = rootfs + "/" + entrypoint (entrypoint is relative) */
    char ep[WUBU_CT_PATH_MAX + WUBU_CT_PATH_MAX];
    if (ct->entrypoint[0] == '/')
        snprintf(ep, sizeof ep, "%s%s", ct->rootfs, ct->entrypoint);
    else
        snprintf(ep, sizeof ep, "%s/%s", ct->rootfs, ct->entrypoint);
    if (stat(ep, &st) != 0 || !S_ISREG(st.st_mode))
        return WUBU_CT_ERR_NOENTRY;
    /* entrypoint must be a loadable ELF */
    FILE *f = fopen(ep, "rb");
    if (!f) return WUBU_CT_ERR_NOENTRY;
    static uint8_t hdr[64];
    size_t rd = fread(hdr, 1, sizeof hdr, f);
    fclose(f);
    if (rd < 64) return WUBU_CT_ERR_BADELF;
    if (wubu_elf_validate(hdr, rd) != WUBU_ELF_OK)
        return WUBU_CT_ERR_BADELF;
    return WUBU_CT_OK;
}

int wubu_ct_run(wubu_container_t *ct, int *exit_code)
{
    if (!ct || !exit_code) return WUBU_CT_ERR_ARGS;
    *exit_code = -1;
    int v = wubu_ct_validate(ct);
    if (v != WUBU_CT_OK) return v;

    char ep[WUBU_CT_PATH_MAX + WUBU_CT_PATH_MAX];
    if (ct->entrypoint[0] == '/')
        snprintf(ep, sizeof ep, "%s%s", ct->rootfs, ct->entrypoint);
    else
        snprintf(ep, sizeof ep, "%s/%s", ct->rootfs, ct->entrypoint);

    pid_t pid = fork();
    if (pid < 0) return WUBU_CT_ERR_EXEC;

    if (pid == 0) {
        /* child */
#if defined(__linux__)
        if (ct->isolate) {
            /* unshare mount namespace so the chroot is container-local */
            unshare(CLONE_NEWNS);
        }
        if (ct->isolate) {
            if (chroot(ct->rootfs) != 0) _exit(126);
            if (chdir("/") != 0) _exit(126);
        }
#endif
        /* build argv: entrypoint + configured args + NULL */
        char *cargv[16];
        int n = 0;
        cargv[n++] = (char *)ct->entrypoint;  /* path inside the chroot */
        for (int i = 0; i < ct->argc && n < 15; i++) cargv[n++] = ct->argv[i];
        cargv[n] = NULL;
        /* when not chrooting, exec the on-disk path directly */
        if (!ct->isolate) cargv[0] = ep;
        execv(cargv[0], cargv);
        _exit(127);   /* exec failed */
    }

    /* parent: wait for the child */
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return WUBU_CT_ERR_EXEC;
    if (WIFEXITED(status)) {
        *exit_code = WEXITSTATUS(status);
        return *exit_code == 0 ? WUBU_CT_OK : WUBU_CT_ERR_STATUS;
    }
    return WUBU_CT_ERR_EXEC;   /* killed by signal etc. */
}
