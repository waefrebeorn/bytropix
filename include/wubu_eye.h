/*
 * wubu_eye.h -- frame differencing for the "eyeballs": which patches changed?
 *
 * Lifted as a technique from solidsf-inc/solidcv (Apache-2.0): a VLM does not
 * need to re-encode a whole frame when 95% of it is static. Split the frame into
 * a patch grid, compare against the previous frame, and hand the model only the
 * active patches plus a reuse mask for the rest.
 *
 * This is a technique port, not a dependency: pure C11, integer-only, no Rust,
 * no OpenCV, no cloud API. The crate's published benchmarks are unverified here;
 * what is verified is the arithmetic in tools/test_eye.c.
 */
#ifndef WUBU_EYE_H
#define WUBU_EYE_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int grid_w, grid_h;          /* patches across / down */
    int active;                  /* patches whose changed-pixel count >= threshold */
    int total;
    unsigned long changed_pixels;
    float reused_ratio;          /* (total - active) / total, 1.0 when unchanged */
} wubu_eye_stats;

/* Mark patches to re-encode. Grayscale, row-major, width*height bytes.
 *   patch      patch edge in pixels (>=1)
 *   pixel_tol  a pixel counts as changed when |prev-curr| > pixel_tol
 *   threshold  a patch is active when its changed-pixel count >= threshold
 * mask must hold ceil(w/patch)*ceil(h/patch) bytes; 1 = active, 0 = reuse.
 * Returns 0, or -1 on bad arguments. */
int wubu_eye_patch_mask(const uint8_t *prev, const uint8_t *curr,
                        int w, int h, int patch, int pixel_tol, int threshold,
                        uint8_t *mask, wubu_eye_stats *st);

/* Exact XOR + population count over n bytes, 64 bits at a time. This is the
 * carrier primitive the patch mask is built on; exposed because it is useful
 * on its own (and cheap to verify against a known vector). */
unsigned long wubu_eye_xor_popcount(const uint8_t *a, const uint8_t *b, size_t n);

#endif /* WUBU_EYE_H */