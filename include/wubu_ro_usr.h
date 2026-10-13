#ifndef WUBU_RO_USR_H
#define WUBU_RO_USR_H

/* wubu_ro_usr.h -- OMARCHY #289 immutable /usr read-only root.
 *
 * An immutable read-only root filesystem policy: system directories (/usr,
 * /bin, /lib, /etc, /boot) are mounted READ-ONLY; only state directories (/var,
 * /tmp, /home, /run) are writable. A write to a read-only path is DENIED
 * (fail-closed). This is the base of an atomic/immutable OS (like Fedora
 * Silverblue / ChromeOS): the system image can't be mutated in place, only
 * swapped whole. Deterministic, real work (no stubs). */

typedef enum {
    WUBU_FS_READONLY=0,   /* path is immutable -- writes denied */
    WUBU_FS_WRITABLE,     /* path accepts writes */
} wubu_fs_access_t;

typedef struct {
    /* policy: a path prefix -> access. Longest matching prefix wins. */
    char prefix[16][48];
    wubu_fs_access_t access[16];
    int n;
} wubu_ro_usr_t;

/* Initialise with the standard immutable-root policy (/usr,/bin,/lib,/etc,/boot
 * read-only; /var,/tmp,/home,/run writable). */
int wubu_ro_init(wubu_ro_usr_t *p);

/* Query the access class for a path (longest matching prefix). */
wubu_fs_access_t wubu_ro_classify(const wubu_ro_usr_t *p, const char *path);

/* May a write to this path proceed? 1 = writable, 0 = denied (read-only). */
int wubu_ro_may_write(const wubu_ro_usr_t *p, const char *path);

#endif
