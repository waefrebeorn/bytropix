/* wubu_checkpoint.c -- BRAIN #83 checkpoint round-trips lossless.
 * See wubu_checkpoint.h. */
#include "wubu_checkpoint.h"
#include <string.h>

uint32_t wubu_crc32(const void *data, size_t len) {
    const uint8_t *p = data;
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= p[i];
        for (int b = 0; b < 8; b++)
            crc = (crc >> 1) ^ (0xEDB88320u & (uint32_t)(-(int32_t)(crc & 1)));
    }
    return crc ^ 0xFFFFFFFFu;
}

static void put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v); p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static uint32_t get32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int wubu_ckpt_save(const void *payload, size_t len, uint8_t *out, size_t outsz) {
    if (!payload || !out) return -1;
    if (len > 0xFFFFFFFFu) return -1;
    size_t total = WUBU_CKPT_HDR + len;
    if (total > outsz) return -1;   /* overflow: fail-closed */
    put32(out + 0, WUBU_CKPT_MAGIC);
    put32(out + 4, WUBU_CKPT_VERSION);
    put32(out + 8, (uint32_t)len);
    uint32_t crc = wubu_crc32(payload, len);
    put32(out + 12, crc);
    memcpy(out + WUBU_CKPT_HDR, payload, len);
    return (int)total;
}

int wubu_ckpt_restore(const uint8_t *buf, size_t bufsz, void *dst, size_t dstsz) {
    if (!buf || !dst) return -1;
    if (bufsz < WUBU_CKPT_HDR) return -1;   /* truncated header */
    uint32_t magic = get32(buf + 0);
    uint32_t version = get32(buf + 4);
    uint32_t plen = get32(buf + 8);
    uint32_t crc = get32(buf + 12);
    if (magic != WUBU_CKPT_MAGIC) return -1;        /* bad magic */
    if (version != WUBU_CKPT_VERSION) return -1;    /* unsupported version */
    if ((size_t)plen > bufsz - WUBU_CKPT_HDR) return -1;  /* truncated payload */
    if ((size_t)plen > dstsz) return -1;            /* dst too small */
    const uint8_t *payload = buf + WUBU_CKPT_HDR;
    if (wubu_crc32(payload, plen) != crc) return -1;   /* corruption: fail-closed */
    memcpy(dst, payload, plen);
    return (int)plen;
}
