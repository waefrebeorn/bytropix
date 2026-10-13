/* wubu_chardev.c -- KERNEL #25 character device driver framework.
 * See wubu_chardev.h. */
#include "wubu_chardev.h"
#include <string.h>

int wubu_cd_init(wubu_chardev_t *c) {
    if (!c) return -1;
    memset(c->devs, 0, sizeof c->devs);
    memset(c->handles, 0, sizeof c->handles);
    c->n_devs = 0;
    return 0;
}

int wubu_cd_register(wubu_chardev_t *c, const char *name, const uint8_t *data, size_t len) {
    if (!c || !name || c->n_devs >= WUBU_CD_MAX_DEVS) return -1;
    if (strlen(name) >= 24) return -1;
    if (len > WUBU_CD_BUFLEN) len = WUBU_CD_BUFLEN;
    wubu_cd_dev_t *d = &c->devs[c->n_devs];
    strncpy(d->name, name, 23); d->name[23] = '\0';
    d->len = len;
    if (data && len) memcpy(d->buf, data, len);
    c->n_devs++;
    return 0;
}

static int find_dev(const wubu_chardev_t *c, const char *name) {
    for (int i = 0; i < c->n_devs; i++)
        if (strcmp(c->devs[i].name, name) == 0) return i;
    return -1;
}

int wubu_cd_open(wubu_chardev_t *c, const char *name) {
    if (!c || !name) return -1;
    int di = find_dev(c, name);
    if (di < 0) return -1;
    for (int i = 0; i < WUBU_CD_MAX_OPEN; i++) {
        if (!c->handles[i].in_use) {
            c->handles[i].in_use = 1;
            c->handles[i].dev_idx = di;
            c->handles[i].cursor = 0;
            return i;
        }
    }
    return -1;   /* no free handle */
}

int wubu_cd_read(wubu_chardev_t *c, int h, uint8_t *out, size_t n) {
    if (!c || !out || h < 0 || h >= WUBU_CD_MAX_OPEN || !c->handles[h].in_use) return -1;
    wubu_cd_handle_t *hd = &c->handles[h];
    wubu_cd_dev_t *d = &c->devs[hd->dev_idx];
    if (hd->cursor >= d->len) return 0;   /* EOF */
    size_t avail = d->len - hd->cursor;
    if (n > avail) n = avail;
    memcpy(out, d->buf + hd->cursor, n);
    hd->cursor += n;
    return (int)n;
}

int wubu_cd_write(wubu_chardev_t *c, int h, const uint8_t *data, size_t n) {
    if (!c || !data || h < 0 || h >= WUBU_CD_MAX_OPEN || !c->handles[h].in_use) return -1;
    wubu_cd_dev_t *d = &c->devs[c->handles[h].dev_idx];
    size_t space = WUBU_CD_BUFLEN - d->len;
    if (n > space) n = space;
    if (n > 0) { memcpy(d->buf + d->len, data, n); d->len += n; }
    return (int)n;
}

int wubu_cd_release(wubu_chardev_t *c, int h) {
    if (!c || h < 0 || h >= WUBU_CD_MAX_OPEN || !c->handles[h].in_use) return -1;
    c->handles[h].in_use = 0;
    return 0;
}
