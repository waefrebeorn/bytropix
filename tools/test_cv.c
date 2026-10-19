/*
 * test_cv.c -- core + imgproc + features + uplift (solidcv reimplementation).
 */
#include "wubu_cv.h"
#include "wubu_features.h"
#include "wubu_uplift.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(c, m) do { checks++; if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else printf("  [PASS] %s\n", m); } while (0)

int main(void)
{
    printf("=== test_cv (core, imgproc, features, uplift) ===\n");

    /* --- core --- */
    wubu_img im = {0};
    CHECK(wubu_img_alloc(&im, 8, 8) == 0, "alloc 8x8");
    wubu_img_fill(&im, 10);
    CHECK(wubu_img_at(&im, 3, 3) == 10, "fill + at");
    wubu_img outimg = {0};
    CHECK(wubu_img_clone(&im, &outimg) == 0 && wubu_img_at(&outimg, 7, 7) == 10, "clone");

    /* --- integral + O(1) box stats --- */
    uint64_t *isum = NULL, *isq = NULL;
    CHECK(wubu_integral_u64(&im, &isum, &isq) == 0, "integral builds");
    double mean = 0, var = -1;
    CHECK(wubu_box_stats(isum, isq, 8, 8, 2, 2, 4, 4, &mean, &var) == 0, "box stats run");
    CHECK(fabs(mean - 10.0) < 1e-9 && var < 1e-12, "uniform box: mean 10, var 0");
    free(isum); free(isq);

    /* --- Sobel on a vertical edge --- */
    wubu_img edge = {0};
    wubu_img_alloc(&edge, 16, 8);
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 16; x++)
            edge.data[y * edge.stride + x] = (x < 8) ? 0 : 255;
    int16_t *gx = (int16_t *)calloc(16 * 8, sizeof *gx);
    int16_t *gy = (int16_t *)calloc(16 * 8, sizeof *gy);
    CHECK(wubu_sobel3(&edge, gx, gy) == 0, "sobel runs");
    CHECK(gx[4 * 16 + 8] > 500, "gx is strong on a vertical edge");
    CHECK(abs((int)gy[4 * 16 + 8]) <= 4, "gy is ~0 on a vertical edge");
    free(gx); free(gy);

    /* --- Otsu on a bimodal image --- */
    wubu_img bi = {0};
    wubu_img_alloc(&bi, 16, 16);
    for (int y = 0; y < 16; y++)
        for (int x = 0; x < 16; x++)
            bi.data[y * bi.stride + x] = (x < 8) ? 20 : 210;
    int t = wubu_otsu(&bi);
    CHECK(t >= 20 && t < 210, "otsu picks a threshold between the two modes");

    /* --- Gaussian keeps a uniform image uniform --- */
    wubu_img g = {0};
    wubu_img_alloc(&g, 8, 8);
    CHECK(wubu_gauss3(&im, &g) == 0, "gauss runs");
    CHECK(wubu_img_at(&g, 4, 4) == 10 && wubu_img_at(&g, 0, 0) == 10,
          "uniform stays uniform (edge clamp)");

    /* --- FAST-9 + BRIEF on a bright square --- */
    wubu_img sqimg = {0};
    wubu_img_alloc(&sqimg, 64, 64);
    wubu_img_fill(&sqimg, 20);
    for (int y = 24; y < 40; y++)
        for (int x = 24; x < 40; x++)
            sqimg.data[y * sqimg.stride + x] = 220;

    wubu_keypoint kps[64];
    int n = wubu_fast9(&sqimg, 40, 3, kps, 64);
    printf("  fast9 corners: %d\n", n);
    CHECK(n >= 4, "FAST-9 finds the square's corners");

    int rc = 0;
    for (int i = 0; i < n; i++) rc |= wubu_brief256(&sqimg, &kps[i]);
    CHECK(rc == 0, "BRIEF computes for every keypoint");
    uint8_t copy[32];
    memcpy(copy, kps[0].desc, 32);
    CHECK(wubu_hamming256(kps[0].desc, copy) == 0, "hamming of a descriptor with itself is 0");
    CHECK(wubu_hamming256(kps[0].desc, kps[1].desc) > 0,
          "different corners have different descriptors");

    /* --- uplift is the exact right-inverse of 2x2 box down --- */
    wubu_img rnd = {0};
    wubu_img_alloc(&rnd, 32, 32);
    unsigned s = 12345;
    for (int i = 0; i < 32 * 32; i++) { s = s * 1103515245u + 12345u; rnd.data[i] = (uint8_t)(s >> 24); }

    wubu_img up = {0}, down = {0};
    CHECK(wubu_uplift2x(&rnd, &up) == 0, "uplift 2x runs");
    CHECK(up.w == 64 && up.h == 64, "uplift doubles dimensions");
    unsigned mass_in = 0, mass_up = 0;
    for (int i = 0; i < 32 * 32; i++) mass_in += rnd.data[i];
    for (int i = 0; i < 64 * 64; i++) mass_up += up.data[i];
    CHECK(mass_up == mass_in * 4, "uplift conserves mass (x4 pixels)");

    CHECK(wubu_down2x_box(&up, &down) == 0, "down 2x runs");
    int same = (down.w == rnd.w && down.h == rnd.h);
    for (int i = 0; same && i < 32 * 32; i++) same = (down.data[i] == rnd.data[i]);
    CHECK(same, "down2x(uplift2x(x)) == x exactly");

    /* --- D4 canonical hash --- */
    wubu_img rot = {0};
    wubu_img_alloc(&rot, rnd.h, rnd.w);
    for (int y = 0; y < rnd.h; y++)
        for (int x = 0; x < rnd.w; x++)
            rot.data[(size_t)x * rot.stride + (size_t)(rnd.h - 1 - y)] = rnd.data[(size_t)y * rnd.stride + x];
    uint64_t h0 = wubu_d4_hash(&rnd), h1 = wubu_d4_hash(&rot);
    CHECK(h0 != 0 && h0 == h1, "D4 hash is invariant under 90-degree rotation");

    /* --- Canny + contours (wave 2) --- */
    wubu_img rect = {0};
    wubu_img_alloc(&rect, 64, 64);
    wubu_img_fill(&rect, 20);
    for (int y = 16; y < 48; y++)
        for (int x = 16; x < 48; x++)
            rect.data[y * rect.stride + x] = 220;

    wubu_img edges = {0};
    wubu_img_alloc(&edges, 64, 64);
    CHECK(wubu_canny(&rect, 50, 150, &edges) == 0, "canny runs");
    int edge_pixels = 0;
    for (int i = 0; i < 64 * 64; i++) if (edges.data[i]) edge_pixels++;
    printf("  canny edge pixels: %d\n", edge_pixels);
    CHECK(edge_pixels > 40, "canny finds the rectangle border");
    CHECK(edges.data[32 * 64 + 32] == 0, "canny leaves the interior empty");
    int edge_near_border = 0;
    for (int y = 14; y <= 18; y++)
        for (int x = 30; x <= 34; x++)
            if (edges.data[y * 64 + x]) edge_near_border = 1;
    CHECK(edge_near_border == 1, "canny marks the top border");
    CHECK(edge_pixels < 2 * (32 + 32) * 2, "edges are thin, not a filled band");

    wubu_contour ct[8];
    int nc = wubu_find_contours(&edges, ct, 8, 20);
    printf("  contours: %d (first bbox %d,%d %dx%d area %d)\n",
           nc, nc > 0 ? ct[0].x : -1, nc > 0 ? ct[0].y : -1,
           nc > 0 ? ct[0].w : 0, nc > 0 ? ct[0].h : 0, nc > 0 ? ct[0].area : 0);
    CHECK(nc == 1, "one contour for one rectangle");
    if (nc == 1) {
        CHECK(abs(ct[0].x - 16) <= 2 && abs(ct[0].y - 16) <= 2, "bbox origin matches the rectangle");
        CHECK(abs(ct[0].w - 32) <= 3 && abs(ct[0].h - 32) <= 3, "bbox size matches the rectangle");
        CHECK(ct[0].area >= 40, "contour area counts edge pixels");
    }

    /* Two squares -> two contours. */
    wubu_img two = {0};
    wubu_img_alloc(&two, 64, 64);
    wubu_img_fill(&two, 20);
    for (int y = 8; y < 24; y++)
        for (int x = 8; x < 24; x++) two.data[y * two.stride + x] = 220;
    for (int y = 40; y < 56; y++)
        for (int x = 40; x < 56; x++) two.data[y * two.stride + x] = 220;
    wubu_img edges2 = {0};
    wubu_img_alloc(&edges2, 64, 64);
    CHECK(wubu_canny(&two, 50, 150, &edges2) == 0, "canny on two squares");
    CHECK(wubu_find_contours(&edges2, ct, 8, 12) == 2, "two contours for two squares");

    /* Arguments. */
    CHECK(wubu_canny(NULL, 10, 20, &edges) == -1, "NULL image -> -1");
    CHECK(wubu_canny(&rect, 200, 100, &edges) == -1, "high < low -> -1");
    CHECK(wubu_find_contours(NULL, ct, 8, 1) == -1, "NULL edges -> -1");

    wubu_img_free(&rect); wubu_img_free(&edges);
    wubu_img_free(&two); wubu_img_free(&edges2);

    /* --- argument guards --- */
    CHECK(wubu_img_alloc(NULL, 4, 4) == -1, "NULL img -> -1");
    CHECK(wubu_fast9(NULL, 10, 3, kps, 4) == -1, "NULL im -> -1");
    CHECK(wubu_hamming256(NULL, kps[0].desc) == -1, "NULL descriptor -> -1");

    wubu_img_free(&im); wubu_img_free(&outimg); wubu_img_free(&edge);
    wubu_img_free(&bi); wubu_img_free(&g); wubu_img_free(&sqimg);
    wubu_img_free(&rnd); wubu_img_free(&up); wubu_img_free(&down); wubu_img_free(&rot);

    printf("\n=== test_cv: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}