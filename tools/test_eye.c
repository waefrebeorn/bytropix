/*
 * test_eye.c -- patch-level frame differencing (the "eyeballs").
 *
 * The point is measurable: a frame where one 16x16 patch moved must hand the
 * model 1 patch, not 4,096 tokens. These checks pin the arithmetic, including
 * the partial-patch edges and the tolerance/threshold split.
 */
#include "wubu_eye.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(c, m) do { checks++; if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else printf("  [PASS] %s\n", m); } while (0)

int main(void)
{
    printf("=== test_eye (frame differencing) ===\n");

    /* 1. XOR + popcount carrier. */
    uint8_t a[16], b[16];
    memset(a, 0xFF, sizeof a);
    memset(b, 0x00, sizeof b);
    CHECK(wubu_eye_xor_popcount(a, b, sizeof a) == 128, "128 bits differ over 16 bytes");
    CHECK(wubu_eye_xor_popcount(a, a, sizeof a) == 0, "identical buffers differ in 0 bits");
    memset(b, 0xFF, sizeof b);
    b[15] = 0xFE;
    CHECK(wubu_eye_xor_popcount(a, b, sizeof a) == 1, "tail byte handled outside the 64-bit loop");

    /* 2. Identical 64x64 frame, 16px patches -> 16 cells, all reusable. */
    const int W = 64, H = 64, P = 16;
    uint8_t *prev = (uint8_t *)malloc((size_t)W * H);
    uint8_t *curr = (uint8_t *)malloc((size_t)W * H);
    uint8_t mask[64];
    wubu_eye_stats st;

    memset(prev, 30, (size_t)W * H);
    memset(curr, 30, (size_t)W * H);
    CHECK(wubu_eye_patch_mask(prev, curr, W, H, P, 0, 1, mask, &st) == 0, "mask runs");
    CHECK(st.grid_w == 4 && st.grid_h == 4 && st.total == 16, "4x4 grid of patches");
    CHECK(st.active == 0 && st.changed_pixels == 0, "no patch active on an identical frame");
    CHECK(st.reused_ratio > 0.999f, "100% reuse when nothing moved");

    /* 3. One patch changed -> exactly one active. */
    for (int y = 0; y < P; y++)
        for (int x = 0; x < P; x++)
            curr[y * W + x] = 200;
    CHECK(wubu_eye_patch_mask(prev, curr, W, H, P, 0, 1, mask, &st) == 0, "changed frame runs");
    CHECK(st.active == 1 && st.changed_pixels == 256, "one 16x16 patch, 256 changed pixels");
    CHECK(st.reused_ratio > 0.9374f && st.reused_ratio < 0.9376f, "15/16 reuse ratio");
    int marked = 0;
    for (int i = 0; i < 16; i++) marked += mask[i];
    CHECK(marked == 1 && mask[0] == 1, "only the top-left cell is marked");

    /* 4. Threshold vs tolerance are different knobs. */
    memset(curr, 30, (size_t)W * H);
    curr[0] = 31;                          /* one pixel, delta 1 */
    CHECK(wubu_eye_patch_mask(prev, curr, W, H, P, 0, 2, mask, &st) == 0, "threshold run");
    CHECK(st.active == 0, "a single changed pixel is below a 2-pixel threshold");
    CHECK(wubu_eye_patch_mask(prev, curr, W, H, P, 0, 1, mask, &st) == 0, "threshold 1 run");
    CHECK(st.active == 1, "the same pixel clears a 1-pixel threshold");
    CHECK(wubu_eye_patch_mask(prev, curr, W, H, P, 1, 1, mask, &st) == 0, "tolerance run");
    CHECK(st.active == 0, "pixel_tol=1 ignores a delta of 1");
    CHECK(wubu_eye_patch_mask(prev, curr, W, H, P, 0, 1, mask, &st) == 0, "tolerance 0 run");
    CHECK(st.active == 1, "pixel_tol=0 counts it");

    /* 5. Partial patches at the right/bottom edges. */
    const int W2 = 70, H2 = 50;
    uint8_t *p2 = (uint8_t *)malloc((size_t)W2 * H2);
    uint8_t *c2 = (uint8_t *)malloc((size_t)W2 * H2);
    memset(p2, 10, (size_t)W2 * H2);
    memset(c2, 10, (size_t)W2 * H2);
    c2[(size_t)(H2 - 1) * W2 + (W2 - 1)] = 200;   /* bottom-right corner */
    uint8_t mask2[32];
    CHECK(wubu_eye_patch_mask(p2, c2, W2, H2, P, 0, 1, mask2, &st) == 0, "uneven frame runs");
    CHECK(st.grid_w == 5 && st.grid_h == 4 && st.total == 20, "ceil(70/16) x ceil(50/16)");
    CHECK(st.active == 1 && mask2[19] == 1, "corner pixel lands in the last partial patch");

    /* 6. Arguments. */
    CHECK(wubu_eye_patch_mask(NULL, curr, W, H, P, 0, 1, mask, &st) == -1, "NULL frame -> -1");
    CHECK(wubu_eye_patch_mask(prev, curr, W, H, 0, 0, 1, mask, &st) == -1, "patch 0 -> -1");
    CHECK(wubu_eye_patch_mask(prev, curr, W, H, P, 0, 0, mask, &st) == -1, "threshold 0 -> -1");
    CHECK(wubu_eye_xor_popcount(NULL, b, 8) == 0, "NULL xor -> 0");

    free(prev); free(curr); free(p2); free(c2);
    printf("\n=== test_eye: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}