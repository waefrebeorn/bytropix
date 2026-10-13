#ifndef WUBU_CHECKPOINT_H
#define WUBU_CHECKPOINT_H
#include <stddef.h>
#include <stdint.h>

/* wubu_checkpoint.h -- BRAIN #83 checkpoint round-trips lossless.
 *
 * A lossless checkpoint: serialize a block of model/training state to a
 * byte buffer and restore it EXACTLY (byte-for-byte identical), verified by a
 * CRC32 checksum. The checkpoint carries a header (magic, version, payload
 * size, crc) so a restore can detect truncation/corruption and refuse
 * (fail-closed) rather than load garbage. This is what "round-trips lossless"
 * means: save(state) -> restore() yields a state identical to the original,
 * bit-for-bit. Deterministic, no allocation beyond the caller's buffer,
 * real work (no stubs). */

#define WUBU_CKPT_MAGIC   0x57554243u   /* "WUBC" */
#define WUBU_CKPT_VERSION 1u
#define WUBU_CKPT_HDR     16u           /* header bytes */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t payload_size;   /* bytes of state payload */
    uint32_t crc;            /* CRC32 of the payload */
} wubu_ckpt_hdr_t;

/* CRC32 (IEEE 802.3), for integrity. */
uint32_t wubu_crc32(const void *data, size_t len);

/* Serialize `payload` (len bytes) into `out` (outsz). Writes header+payload.
 * Returns total bytes written, or -1 on overflow/NULL. */
int wubu_ckpt_save(const void *payload, size_t len, uint8_t *out, size_t outsz);

/* Restore from a checkpoint buffer: validates header (magic/version), checks
 * payload fits dstsz, verifies CRC. On success copies payload to dst and
 * returns the payload size; on any validation failure returns -1
 * (fail-closed — never loads corrupt/truncated data). */
int wubu_ckpt_restore(const uint8_t *buf, size_t bufsz, void *dst, size_t dstsz);

#endif
