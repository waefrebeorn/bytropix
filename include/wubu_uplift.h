/*
 * wubu_uplift.h -- mass-conserving 2x zoom and D4 canonical hashing.
 *
 * solidcv's `uplift` claim that matters: a zoom that is the exact right-inverse
 * of a 2x2 box downsample conserves mass and cannot invent pixels. That is a
 * property we can test, unlike a GAN's plausibility.
 */
#ifndef WUBU_UPLIFT_H
#define WUBU_UPLIFT_H

#include "wubu_cv.h"
#include <stdint.h>

/* Each source pixel becomes a 2x2 block of the same value. */
int wubu_uplift2x(const wubu_img *in, wubu_img *out);

/* 2x2 box average with rounding; out must be (w/2)x(h/2) or allocatable. */
int wubu_down2x_box(const wubu_img *in, wubu_img *out);

/* FNV-1a over the lexicographically smallest of the 8 dihedral transforms, so
 * a rotated or mirrored image hashes the same. */
uint64_t wubu_d4_hash(const wubu_img *im);

#endif /* WUBU_UPLIFT_H */