/* wubu_fw_rollback.c -- OMARCHY #288 firmware rollback on signature failure.
 * See wubu_fw_rollback.h. */
#include "wubu_fw_rollback.h"
#include <string.h>

int wubu_fw_init(wubu_fw_rollback_t *fw) {
    if (!fw) return -1;
    memset(fw->images, 0, sizeof fw->images);
    fw->n = 0;
    return 0;
}

int wubu_fw_register(wubu_fw_rollback_t *fw, const char *name,
                     unsigned long long signature, int generation) {
    if (!fw || !name || fw->n >= WUBU_FW_MAX_IMAGES) return -1;
    if (strlen(name) >= 32) return -1;
    wubu_fw_image_t *im = &fw->images[fw->n];
    strncpy(im->name, name, 31); im->name[31] = '\0';
    im->signature = signature;
    im->signed_ok = 1;   /* registered images are known-good signed */
    im->generation = generation;
    fw->n++;
    return 0;
}

int wubu_fw_verify(const wubu_fw_rollback_t *fw, const char *name,
                   unsigned long long signature) {
    if (!fw || !name) return 0;
    for (int i = 0; i < fw->n; i++) {
        if (strcmp(fw->images[i].name, name) == 0) {
            /* must be signed-ok AND the signature must match */
            return (fw->images[i].signed_ok && fw->images[i].signature == signature) ? 1 : 0;
        }
    }
    return 0;   /* unknown image -> not verified */
}

int wubu_fw_boot(wubu_fw_rollback_t *fw, const char *name,
                 unsigned long long signature) {
    if (!fw || !name) return -1;
    /* if the requested image verifies, boot it */
    if (wubu_fw_verify(fw, name, signature)) {
        for (int i = 0; i < fw->n; i++)
            if (strcmp(fw->images[i].name, name) == 0) return i;
    }
    /* verification failed -> roll back to the newest known-good signed image */
    int best = -1, best_gen = -1;
    for (int i = 0; i < fw->n; i++) {
        if (fw->images[i].signed_ok && fw->images[i].generation > best_gen) {
            best_gen = fw->images[i].generation;
            best = i;
        }
    }
    return best;   /* -1 if no trustworthy image */
}
