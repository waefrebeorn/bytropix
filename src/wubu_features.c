/*
 * wubu_features.c -- FAST-9, BRIEF-256, Hamming. See wubu_features.h.
 */
#include "wubu_features.h"
#include <string.h>

static const int RING[16][2] = {
    { 0, -3}, { 1, -3}, { 2, -2}, { 3, -1}, { 3,  0}, { 3,  1}, { 2,  2}, { 1,  3},
    { 0,  3}, {-1,  3}, {-2,  2}, {-3,  1}, {-3,  0}, {-3, -1}, {-2, -2}, {-1, -3},
};

int wubu_fast9(const wubu_img *im, int threshold, int border,
               wubu_keypoint *out, int max_keypoints)
{
    if (!im || !im->data || !out || max_keypoints <= 0 || threshold < 0) return -1;
    if (border < 3) border = 3;

    int n = 0;
    for (int y = border; y < im->h - border && n < max_keypoints; y++) {
        for (int x = border; x < im->w - border && n < max_keypoints; x++) {
            int p = wubu_img_at(im, x, y);
            int v[16], brighter[16], darker[16];
            for (int i = 0; i < 16; i++) {
                v[i] = wubu_img_at(im, x + RING[i][0], y + RING[i][1]);
                brighter[i] = v[i] > p + threshold;
                darker[i] = v[i] < p - threshold;
            }

            int best_run = 0, score = 0;
            for (int cls = 0; cls < 2; cls++) {
                const int *flag = cls == 0 ? brighter : darker;
                int run = 0;
                /* Walk twice around the ring so a run crossing index 15 counts. */
                for (int k = 0; k < 32; k++) {
                    if (flag[k % 16]) { run++; if (run > best_run) best_run = run; }
                    else run = 0;
                }
            }
            if (best_run < 9) continue;

            for (int i = 0; i < 16; i++) {
                int d = v[i] - p;
                score += d < 0 ? -d : d;
            }
            out[n].x = x;
            out[n].y = y;
            out[n].score = (uint32_t)score;
            memset(out[n].desc, 0, sizeof out[n].desc);
            n++;
        }
    }
    return n;
}

/* Deterministic sampling pattern: 256 pairs inside [-15, 15]. */
static void brief_pair(int i, int *x1, int *y1, int *x2, int *y2)
{
    uint32_t s = (uint32_t)(i * 2654435761u + 0x9E3779B9u);
    s ^= s >> 16; s *= 0x7FEB352Du; s ^= s >> 15;
    *x1 = (int)(s % 31u) - 15; s = s * 1664525u + 1013904223u;
    *y1 = (int)(s % 31u) - 15; s = s * 1664525u + 1013904223u;
    *x2 = (int)(s % 31u) - 15; s = s * 1664525u + 1013904223u;
    *y2 = (int)(s % 31u) - 15;
}

int wubu_brief256(const wubu_img *im, wubu_keypoint *kp)
{
    if (!im || !im->data || !kp) return -1;
    memset(kp->desc, 0, sizeof kp->desc);

    for (int i = 0; i < 256; i++) {
        int x1, y1, x2, y2;
        brief_pair(i, &x1, &y1, &x2, &y2);
        int a = wubu_img_at(im, kp->x + x1, kp->y + y1);
        int b = wubu_img_at(im, kp->x + x2, kp->y + y2);
        if (a < b) kp->desc[i >> 3] |= (uint8_t)(1u << (i & 7));
    }
    return 0;
}

int wubu_hamming256(const uint8_t a[32], const uint8_t b[32])
{
    if (!a || !b) return -1;
    int d = 0;
    for (int i = 0; i < 32; i++) d += __builtin_popcount((unsigned)(a[i] ^ b[i]));
    return d;
}