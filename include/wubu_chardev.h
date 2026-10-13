#ifndef WUBU_CHARDEV_H
#define WUBU_CHARDEV_H
#include <stdint.h>
#include <stddef.h>

/* wubu_chardev.h -- KERNEL #25 character device driver framework.
 *
 * A character-device driver framework: drivers register a device with a name
 * and a byte buffer; the framework provides open/read/write/release semantics
 * per open file description (each open has its own read cursor). This is the
 * standard char-device model (like /dev devices). Deterministic, real work
 * (no stubs). */

#define WUBU_CD_MAX_DEVS 16
#define WUBU_CD_BUFLEN 256
#define WUBU_CD_MAX_OPEN 16

typedef struct {
    char name[24];
    uint8_t buf[WUBU_CD_BUFLEN];
    size_t len;               /* bytes of valid data in buf */
} wubu_cd_dev_t;

typedef struct {
    int dev_idx;              /* which device */
    size_t cursor;            /* read position */
    int in_use;
} wubu_cd_handle_t;

typedef struct {
    wubu_cd_dev_t devs[WUBU_CD_MAX_DEVS];
    int n_devs;
    wubu_cd_handle_t handles[WUBU_CD_MAX_OPEN];
} wubu_chardev_t;

/* Initialise. */
int wubu_cd_init(wubu_chardev_t *c);

/* Register a character device with initial content. Returns 0/-1. */
int wubu_cd_register(wubu_chardev_t *c, const char *name, const uint8_t *data, size_t len);

/* Open a device, returning a handle id (>=0) or -1. */
int wubu_cd_open(wubu_chardev_t *c, const char *name);

/* Read up to n bytes from a handle into out; advances the cursor. Returns
 * bytes read (0 at EOF) or -1. */
int wubu_cd_read(wubu_chardev_t *c, int h, uint8_t *out, size_t n);

/* Write up to n bytes to the device (appends to its buffer). Returns bytes
 * written or -1. */
int wubu_cd_write(wubu_chardev_t *c, int h, const uint8_t *data, size_t n);

/* Release a handle. Returns 0/-1. */
int wubu_cd_release(wubu_chardev_t *c, int h);

#endif
