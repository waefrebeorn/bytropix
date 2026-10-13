#ifndef WUBU_FW_ROLLBACK_H
#define WUBU_FW_ROLLBACK_H

/* wubu_fw_rollback.h -- OMARCHY #288 firmware rollback on signature failure.
 *
 * If a firmware image fails signature verification during boot, the system
 * must NOT boot it -- it rolls back to the previous known-good, signed
 * firmware. This module verifies a firmware image's signature (a placeholder
 * deterministic hash-based check: the image's tag must match a recorded
 * signed tag) and, on failure, selects the rollback image. Fail-closed: an
 * unsigned or mismatched firmware is never booted. Deterministic, real work
 * (no stubs). */

#define WUBU_FW_MAX_IMAGES 8

typedef struct {
    char name[32];
    unsigned long long signature;   /* recorded signed signature/tag */
    int    signed_ok;               /* 1 = known-good signed image */
    int    generation;              /* higher = newer */
} wubu_fw_image_t;

typedef struct {
    wubu_fw_image_t images[WUBU_FW_MAX_IMAGES];
    int n;
} wubu_fw_rollback_t;

/* Initialise. */
int wubu_fw_init(wubu_fw_rollback_t *fw);

/* Register a firmware image with its signed signature. Returns 0/-1. */
int wubu_fw_register(wubu_fw_rollback_t *fw, const char *name,
                     unsigned long long signature, int generation);

/* Verify a firmware image by name against its expected signature. Returns
 * 1 if the image is registered, signed-ok, and the signature matches; 0 if
 * verification fails (unsigned/mismatch/unknown). */
int wubu_fw_verify(const wubu_fw_rollback_t *fw, const char *name,
                   unsigned long long signature);

/* Select the image to boot: the highest-generation signed-ok image whose
 * signature verifies. On verification failure of the requested image, falls
 * back to the newest known-good signed image. Returns the index of the
 * selected image, or -1 if none is trustworthy. */
int wubu_fw_boot(wubu_fw_rollback_t *fw, const char *name,
                 unsigned long long signature);

#endif
