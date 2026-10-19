/*
 * wubu_eye.c -- patch-level frame differencing. See wubu_eye.h.
 */
#include "wubu_eye.h"
#include <stdlib.h>
#include <string.h>

unsigned long wubu_eye_xor_popcount(const uint8_t *a, const uint8_t *b, size_t n)
{
    if (!a || !b) return 0;

    unsigned long bits = 0;
    size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        uint64_t x, y;
        memcpy(&x, a + i, 8);
        memcpy(&y, b + i, 8);
        bits += (unsigned long)__builtin_popcountll(x ^ y);
    }
    for (; i < n; i++) {
        bits += (unsigned long)__builtin_popcount((unsigned)(a[i] ^ b[i]));
    }
    return bits;
}

int wubu_eye_patch_mask(const uint8_t *prev, const uint8_t *curr,
                        int w, int h, int patch, int pixel_tol, int threshold,
                        uint8_t *mask, wubu_eye_stats *st)
{
    if (!prev || !curr || !mask || !st) return -1;
    if (w <= 0 || h <= 0 || patch <= 0 || pixel_tol < 0 || threshold < 1) return -1;

    int gw = (w + patch - 1) / patch;
    int gh = (h + patch - 1) / patch;
    size_t cells = (size_t)gw * (size_t)gh;

    memset(mask, 0, cells);

    /* Count changed pixels per patch with one allocation for the grid. */
    if (cells > 16u * 1024u * 1024u) return -1;
    unsigned *counts = (unsigned *)calloc(cells, sizeof *counts);
    if (!counts) return -1;

    unsigned long changed = 0;
    for (int y = 0; y < h; y++) {
        const uint8_t *pr = prev + (size_t)y * (size_t)w;
        const uint8_t *cu = curr + (size_t)y * (size_t)w;
        unsigned *row = counts + (size_t)(y / patch) * (size_t)gw;
        for (int x = 0; x < w; x++) {
            int d = (int)pr[x] - (int)cu[x];
            if (d < 0) d = -d;
            if (d > pixel_tol) {
                row[x / patch]++;
                changed++;
            }
        }
    }

    int active = 0;
    for (size_t i = 0; i < cells; i++) {
        if (counts[i] >= (unsigned)threshold) {
            mask[i] = 1;
            active++;
        }
    }

    st->grid_w = gw;
    st->grid_h = gh;
    st->total = (int)cells;
    st->active = active;
    st->changed_pixels = changed;
    st->reused_ratio = cells ? (float)(cells - (size_t)active) / (float)cells : 0.0f;
    free(counts);
    return 0;
}