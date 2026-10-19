/*
 * wubu_glass.h -- 64-D hand-auditable image signature (GlassBox).
 *
 * solidcv's GlassBoxSignature64D is a descriptor where every coordinate has a
 * stated definition, so a probe can be reasoned about instead of trusted.
 *
 * Deviations from the crate's layout are marked in the .c file. In particular
 * coordinates [24..32) use 2D Haar detail ratios and [32..40) are reserved:
 * the crate fills those with 3D Schur-Feshbach and worldtube-causality terms,
 * and we have no metric 3D pipeline to feed them. Reserving is honest; filling
 * them with invented numbers would not be.
 */
#ifndef WUBU_GLASS_H
#define WUBU_GLASS_H

#include "wubu_cv.h"
#include <stdint.h>

#define WUBU_GLASS_DIMS 64

/* Fill out[0..63] with values in [0,1]. Returns 0, or -1. */
int wubu_glassbox64(const wubu_img *im, float out[WUBU_GLASS_DIMS]);

/* Canonical hash: the signature of each of the 8 dihedral transforms is
 * quantised to bytes and hashed; the smallest hash wins, so a rotated or
 * mirrored image hashes the same. Returns 0 on bad input. */
uint64_t wubu_glassbox_hash(const wubu_img *im);

#endif /* WUBU_GLASS_H */