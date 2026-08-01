/*
 * wubu_kv_transfer.c — Localhost KV transfer layer (D05, doc 007/002).
 *
 * NIXL/UCX analog for a single host: a prefill "instance" writes KV blocks
 * for a completed prefix to a transfer buffer (mmap'd temp file); a decode
 * "instance" reads them back. This is the PD-disaggregation handoff — KV
 * produced by the prefill engine is shipped to the decode engine without
 * recomputation.
 *
 * Self-contained C11, no third-party deps. Uses posix shm or a temp file
 * mmap (falls back to plain file read/write if shm unavailable).
 */
#include "wubu_kv_transfer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>

struct wubu_kv_transfer {
    char *path;
    int fd;
    void *base;       /* mmap base (NULL if using raw file I/O) */
    size_t capacity;
    size_t used;
};

wubu_kv_transfer_t *wubu_kv_transfer_create(const char *shm_path, size_t capacity_bytes) {
    wubu_kv_transfer_t *t = (wubu_kv_transfer_t *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->capacity = capacity_bytes > 0 ? capacity_bytes : (size_t)64 * 1024 * 1024;
    t->path = strdup(shm_path ? shm_path : "/tmp/wubu_kv_xfer.bin");
    if (!t->path) { free(t); return NULL; }

    t->fd = open(t->path, O_CREAT | O_RDWR, 0600);
    if (t->fd < 0) { free(t->path); free(t); return NULL; }
    if (ftruncate(t->fd, (off_t)t->capacity) != 0) {
        close(t->fd); free(t->path); free(t); return NULL;
    }
    t->base = mmap(NULL, t->capacity, PROT_READ | PROT_WRITE, MAP_SHARED, t->fd, 0);
    if (t->base == MAP_FAILED) {
        /* Fall back to raw file I/O */
        t->base = NULL;
    }
    return t;
}

void wubu_kv_transfer_free(wubu_kv_transfer_t *t) {
    if (!t) return;
    if (t->base) munmap(t->base, t->capacity);
    if (t->fd >= 0) close(t->fd);
    if (t->path) { unlink(t->path); free(t->path); }
    free(t);
}

/* Write a KV block (len bytes) at a given slot offset. Returns 0 on success. */
int wubu_kv_transfer_put(wubu_kv_transfer_t *t, size_t slot, const void *data, size_t len) {
    if (!t || !data || len == 0) return -1;
    size_t off = slot * len; /* fixed-size slot addressing */
    if (off + len > t->capacity) return -1;

    if (t->base) {
        memcpy((uint8_t *)t->base + off, data, len);
        /* Ensure visibility to other mappers */
        msync((uint8_t *)t->base + off, len, MS_SYNC);
    } else {
        if (lseek(t->fd, (off_t)off, SEEK_SET) < 0) return -1;
        ssize_t wr = write(t->fd, data, len);
        if (wr != (ssize_t)len) return -1;
    }
    if (off + len > t->used) t->used = off + len;
    return 0;
}

/* Read a KV block (len bytes) from a slot offset. Returns 0 on success. */
int wubu_kv_transfer_get(wubu_kv_transfer_t *t, size_t slot, void *out, size_t len) {
    if (!t || !out || len == 0) return -1;
    size_t off = slot * len;
    if (off + len > t->capacity) return -1;

    if (t->base) {
        memcpy(out, (uint8_t *)t->base + off, len);
    } else {
        if (lseek(t->fd, (off_t)off, SEEK_SET) < 0) return -1;
        ssize_t rd = read(t->fd, out, len);
        if (rd != (ssize_t)len) return -1;
    }
    return 0;
}

size_t wubu_kv_transfer_used(const wubu_kv_transfer_t *t) {
    return t ? t->used : 0;
}
