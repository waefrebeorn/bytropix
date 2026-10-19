/*
 * wubu_features.h -- FAST-9 corners + 256-bit BRIEF + Hamming matching.
 *
 * solidcv's features2d reimplementation: the detector and descriptor are the
 * parts an OS eye actually needs (track a window between two frames). No ORB
 * pyramid yet — that is the next rung, and this header says so rather than
 * implying it exists.
 */
#ifndef WUBU_FEATURES_H
#define WUBU_FEATURES_H

#include "wubu_cv.h"
#include <stdint.h>

typedef struct {
    int x, y;
    uint32_t score;
    uint8_t desc[32];   /* 256 bits */
} wubu_keypoint;

/* FAST-9 with a 16-pixel ring at radius 3. Returns the number written, or -1. */
int wubu_fast9(const wubu_img *im, int threshold, int border,
               wubu_keypoint *out, int max_keypoints);

/* Fill desc[] for one keypoint using a fixed (seeded) sampling pattern. */
int wubu_brief256(const wubu_img *im, wubu_keypoint *kp);

/* Hamming distance between two 256-bit descriptors. */
int wubu_hamming256(const uint8_t a[32], const uint8_t b[32]);

#endif /* WUBU_FEATURES_H */