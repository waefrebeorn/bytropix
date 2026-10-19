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
/* ---------------- ORB pyramid + matching (wave 3) ---------------- */

#include "wubu_uplift.h"
#include <limits.h>

int wubu_orb_detect(const wubu_img *im, int levels, int threshold,
                    wubu_keypoint *out, int max_keypoints)
{
    if (!im || !im->data || !out || levels < 1 || max_keypoints <= 0) return -1;

    wubu_img cur = {0};
    if (wubu_img_clone(im, &cur) != 0) return -1;

    int n = 0, scale = 1;
    for (int L = 0; L < levels && n < max_keypoints; L++) {
        wubu_keypoint tmp[256];
        int k = wubu_fast9(&cur, threshold, 3, tmp, 256);
        if (k < 0) break;

        for (int i = 0; i < k && n < max_keypoints; i++) {
            tmp[i].level = L;
            wubu_brief256(&cur, &tmp[i]);
            int ox = tmp[i].x * scale, oy = tmp[i].y * scale;
            if (ox >= im->w || oy >= im->h) continue;
            out[n] = tmp[i];
            out[n].x = ox;
            out[n].y = oy;
            n++;
        }

        if (L + 1 < levels) {
            wubu_img next = {0};
            if (wubu_down2x_box(&cur, &next) != 0) break;
            wubu_img_free(&cur);
            cur = next;
            scale *= 2;
        }
    }
    wubu_img_free(&cur);
    return n;
}

int wubu_match_descriptors(const wubu_keypoint *a, int na,
                           const wubu_keypoint *b, int nb,
                           int max_dist, float ratio,
                           wubu_match *out, int max_matches)
{
    if (!a || !b || !out || na <= 0 || nb <= 0 || max_matches <= 0) return -1;
    if (max_dist < 0 || ratio <= 0.0f) return -1;

    int m = 0;
    for (int i = 0; i < na && m < max_matches; i++) {
        int best = INT_MAX, second = INT_MAX, bi = -1;
        for (int j = 0; j < nb; j++) {
            int d = wubu_hamming256(a[i].desc, b[j].desc);
            if (d < best) { second = best; best = d; bi = j; }
            else if (d < second) second = d;
        }
        if (bi < 0 || best > max_dist) continue;
        /* Lowe ratio test: an ambiguous best (second close to it) is dropped. */
        if (second != INT_MAX && !((float)best < ratio * (float)second)) continue;
        out[m].a = i;
        out[m].b = bi;
        out[m].dist = best;
        m++;
    }
    return m;
}


int wubu_harris_score(const wubu_img *im, wubu_keypoint *kp, int n)
{
    if (!im || !im->data || !kp || n < 0) return -1;

    for (int i = 0; i < n; i++) {
        int px = kp[i].x, py = kp[i].y;
        double A = 0, B = 0, C = 0;

        for (int oy = -2; oy <= 2; oy++) {
            for (int ox = -2; ox <= 2; ox++) {
                int x = px + ox, y = py + oy;
                float Ix = ((float)wubu_img_at(im, x + 1, y)
                          - (float)wubu_img_at(im, x - 1, y)) * 0.5f;
                float Iy = ((float)wubu_img_at(im, x, y + 1)
                          - (float)wubu_img_at(im, x, y - 1)) * 0.5f;
                A += (double)Ix * Ix;
                B += (double)Ix * Iy;
                C += (double)Iy * Iy;
            }
        }
        double tr = A + C;
        double det = A * C - B * B;
        double r = det - 0.04 * tr * tr;
        kp[i].score = r > 0.0 ? (uint32_t)r : 0u;
    }
    return 0;
}

int wubu_keypoint_dedupe(wubu_keypoint *kp, int n, int radius)
{
    if (!kp || n < 0 || radius < 0) return -1;

    /* Selection sort by score descending (n is small at the call sites). */
    for (int i = 0; i < n; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++)
            if (kp[j].score > kp[best].score) best = j;
        if (best != i) { wubu_keypoint t = kp[i]; kp[i] = kp[best]; kp[best] = t; }
    }

    int kept = 0;
    for (int i = 0; i < n; i++) {
        int near = 0;
        for (int j = 0; j < kept; j++) {
            int dx = kp[i].x - kp[j].x, dy = kp[i].y - kp[j].y;
            if (dx * dx + dy * dy <= radius * radius) { near = 1; break; }
        }
        if (!near) kp[kept++] = kp[i];
    }
    return kept;
}
