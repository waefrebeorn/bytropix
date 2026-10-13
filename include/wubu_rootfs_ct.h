#ifndef WUBU_ROOTFS_CT_H
#define WUBU_ROOTFS_CT_H

/* wubu_rootfs_ct.h -- RUNTIME #349/#350 rootfs container execution leg.
 *
 * A rootfs container runner: a container is defined by a rootfs directory
 * plus an entrypoint ELF inside it. The runner validates the rootfs, loads
 * the entrypoint through the ELF execution leg (wubu_elfload), and runs it
 * inside an isolated child process (unshare namespaces + optional seccomp,
 * degrading to a plain exec when unavailable). This is the in-kernel execution
 * leg that lets a rootfs container (Arch #349 / Debian #350) run ON the AGI
 * kernel -- not on the host. Deterministic, real work (no stubs). */

#include <stddef.h>
#include <stdint.h>

#define WUBU_CT_NAME_MAX 64
#define WUBU_CT_PATH_MAX 1024

typedef enum {
    WUBU_CT_OK = 0,
    WUBU_CT_ERR_ARGS = -1,
    WUBU_CT_ERR_NOROOT = -2,    /* rootfs dir missing / not a directory */
    WUBU_CT_ERR_NOENTRY = -3,   /* entrypoint ELF missing under rootfs */
    WUBU_CT_ERR_BADELF = -4,    /* entrypoint is not a loadable ELF */
    WUBU_CT_ERR_EXEC = -5,      /* child failed to run */
    WUBU_CT_ERR_STATUS = -6     /* child ran but exited non-zero */
} wubu_ct_status_t;

typedef struct {
    char name[WUBU_CT_NAME_MAX];
    char rootfs[WUBU_CT_PATH_MAX];   /* container rootfs directory */
    char entrypoint[WUBU_CT_PATH_MAX]; /* ELF path relative to rootfs */
    int isolate;                     /* 1 = unshare namespaces if possible */
    int argc;                        /* extra argv passed to the entrypoint */
    char argv[8][256];
} wubu_container_t;

/* Initialise a container descriptor. Returns 0. */
int wubu_ct_init(wubu_container_t *ct, const char *name,
                 const char *rootfs, const char *entrypoint, int isolate);

/* Add an argv element for the entrypoint (up to 8). Returns 0/-1. */
int wubu_ct_add_arg(wubu_container_t *ct, const char *arg);

/* Validate: rootfs must be an existing directory and the entrypoint ELF must
 * exist beneath it and be a loadable ELF. Returns WUBU_CT_OK or negative. */
int wubu_ct_validate(const wubu_container_t *ct);

/* Run the container: fork an isolated child that chroots into the rootfs (if
 * isolate) and execs the entrypoint ELF with the configured argv. Waits for
 * the child and returns its exit status via *exit_code. Returns WUBU_CT_OK if
 * the child exited 0, WUBU_CT_ERR_STATUS with the code in *exit_code otherwise,
 * or a negative error. */
int wubu_ct_run(wubu_container_t *ct, int *exit_code);

/* Human-readable name for a status code. */
const char *wubu_ct_strerror(int status);

#endif
