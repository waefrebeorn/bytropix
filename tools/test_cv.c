/*
 * test_cv.c -- core + imgproc + features + uplift (solidcv reimplementation).
 */
#include "wubu_cv.h"
#include "wubu_features.h"
#include "wubu_uplift.h"
#include "wubu_flow.h"
#include "wubu_sae.h"
#include "wubu_glass.h"
#include "wubu_calib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(c, m) do { checks++; if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else printf("  [PASS] %s\n", m); } while (0)


static float bilin(const wubu_img *im, float x, float y)
{
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    float fx = x - (float)x0, fy = y - (float)y0;
    float a = (float)wubu_img_at(im, x0, y0);
    float b = (float)wubu_img_at(im, x0 + 1, y0);
    float c = (float)wubu_img_at(im, x0, y0 + 1);
    float d = (float)wubu_img_at(im, x0 + 1, y0 + 1);
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

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

    /* --- ORB pyramid + matcher (wave 3) --- */
    wubu_keypoint orb[128];
    int no = wubu_orb_detect(&sqimg, 2, 40, orb, 128);
    printf("  orb keypoints: %d\n", no);
    CHECK(no > 0, "ORB detects across the pyramid");
    int in_bounds = 1, levels_ok = 1;
    for (int i = 0; i < no; i++) {
        if (orb[i].x < 0 || orb[i].y < 0 || orb[i].x >= 64 || orb[i].y >= 64) in_bounds = 0;
        if (orb[i].level < 0 || orb[i].level > 1) levels_ok = 0;
    }
    CHECK(in_bounds, "ORB maps coordinates back inside the original image");
    CHECK(levels_ok, "ORB records a valid pyramid level");

    wubu_keypoint A[4], B[4];
    memset(A, 0, sizeof A); memset(B, 0, sizeof B);
    for (int i = 0; i < 4; i++) { A[i].desc[i] = 0xFF; B[i].desc[i] = 0xFF; }

    wubu_match mm[8];
    int nm = wubu_match_descriptors(A, 4, B, 4, 10, 0.8f, mm, 8);
    CHECK(nm == 4, "self-matching yields one match per query");
    int all_zero = 1;
    for (int i = 0; i < nm; i++) if (mm[i].dist != 0) all_zero = 0;
    CHECK(all_zero, "identical descriptors match at distance 0");

    memcpy(B[1].desc, B[0].desc, 32);
    CHECK(wubu_match_descriptors(A, 1, B, 2, 10, 0.8f, mm, 8) == 0,
          "ratio test rejects an ambiguous best");

    CHECK(wubu_match_descriptors(A, 1, B, 1, 0, 0.8f, mm, 8) == 1,
          "max_dist=0 still accepts an exact match");
    CHECK(wubu_match_descriptors(NULL, 1, B, 1, 0, 0.8f, mm, 8) == -1,
          "NULL query set -> -1");

    /* --- morphology (wave 4) --- */
    wubu_img dot = {0}, dil = {0};
    wubu_img_alloc(&dot, 16, 16);
    wubu_img_alloc(&dil, 16, 16);
    wubu_img_fill(&dot, 0);
    dot.data[8 * dot.stride + 8] = 255;
    CHECK(wubu_morph3(&dot, &dil, 1) == 0, "dilate runs");
    int lit = 0;
    for (int i = 0; i < 16 * 16; i++) if (dil.data[i]) lit++;
    CHECK(lit == 9, "a single pixel dilates to a 3x3 block");

    wubu_img block = {0}, ero = {0};
    wubu_img_alloc(&block, 16, 16);
    wubu_img_alloc(&ero, 16, 16);
    for (int y = 5; y < 10; y++) for (int x = 5; x < 10; x++) block.data[y * block.stride + x] = 200;
    CHECK(wubu_morph3(&block, &ero, 0) == 0, "erode runs");
    int inside = 0, outside = 0;
    for (int y = 0; y < 16; y++)
        for (int x = 0; x < 16; x++) {
            int v = ero.data[y * ero.stride + x];
            if (x >= 6 && x <= 8 && y >= 6 && y <= 8) { if (v == 200) inside++; }
            else if (v != 0) outside++;
        }
    CHECK(inside == 9 && outside == 0, "a 5x5 block erodes to its 3x3 core");
    CHECK(wubu_morph3(&dot, &dil, 2) == -1, "unknown mode -> -1");

    /* --- Lucas-Kanade (wave 5) --- */
    wubu_img s2 = {0}, moved = {0};
    wubu_img_alloc(&s2, 96, 96);
    wubu_img_alloc(&moved, 96, 96);
    /* A deterministic, well-conditioned texture: LK needs gradient in both axes. */
    for (int y = 0; y < 96; y++) {
        for (int x = 0; x < 96; x++) {
            double v = 128.0 + 55.0 * sin(x * 0.31) + 45.0 * cos(y * 0.27)
                     + 20.0 * sin((x + y) * 0.13);
            if (v < 0.0) v = 0.0;
            if (v > 255.0) v = 255.0;
            s2.data[y * s2.stride + x] = (uint8_t)v;
        }
    }

    for (int y = 0; y < 96; y++)
        for (int x = 0; x < 96; x++)
            moved.data[y * moved.stride + x] = wubu_img_at(&s2, x - 2, y - 3);

    wubu_keypoint fp[3];
    fp[0].x = 40; fp[0].y = 40;
    fp[1].x = 48; fp[1].y = 52;
    fp[2].x = 56; fp[2].y = 44;
    float fdx[3], fdy[3];
    CHECK(wubu_lk_flow(&s2, &moved, fp, 3, 15, 10, fdx, fdy) == 0, "LK runs");
    printf("  flow: (%.2f,%.2f) (%.2f,%.2f) (%.2f,%.2f)\n",
           fdx[0], fdy[0], fdx[1], fdy[1], fdx[2], fdy[2]);
    int flow_ok = 1;
    for (int i = 0; i < 3; i++)
        if (fabsf(fdx[i] - 2.0f) > 0.6f || fabsf(fdy[i] - 3.0f) > 0.6f) flow_ok = 0;
    CHECK(flow_ok, "LK recovers a known (2,3) shift within 0.6 px");

    float idx_[3], idy[3];
    CHECK(wubu_lk_flow(&s2, &s2, fp, 3, 15, 10, idx_, idy) == 0, "LK identity runs");
    CHECK(fabsf(idx_[0]) < 0.05f && fabsf(idy[0]) < 0.05f, "identity pair gives ~zero flow");
    CHECK(wubu_lk_flow(&s2, &s2, fp, 3, 4, 10, idx_, idy) == -1, "even window -> -1");

    wubu_img_free(&dot); wubu_img_free(&dil); wubu_img_free(&block); wubu_img_free(&ero);
    wubu_img_free(&s2); wubu_img_free(&moved);


    /* --- SAE 256-bit matcher (wave 6) --- */
    wubu_sae_atom atoms[3];
    memset(atoms, 0, sizeof atoms);
    atoms[0].id = 10; atoms[1].id = 20; atoms[2].id = 30;
    atoms[1].sig[0] = 0xFF;
    atoms[2].sig[1] = 0xFF;

    uint8_t q[32]; memset(q, 0, sizeof q);
    int sai = -1, sad = -1;
    CHECK(wubu_sae_match(q, atoms, 3, &sai, &sad) == 0 && sai == 0 && sad == 0,
          "SAE: exact match at distance 0");
    q[0] = 0xFF;
    CHECK(wubu_sae_match(q, atoms, 3, &sai, &sad) == 0 && sai == 1 && sad == 0,
          "SAE: nearest atom by Hamming");
    q[0] = 0x0F;
    CHECK(wubu_sae_match(q, atoms, 3, &sai, &sad) == 0 && sai == 0 && sad == 4,
          "SAE: equal distances resolve to the lowest index");

    memset(q, 0, sizeof q); q[0] = 0xFF;
    int oi = -9, od = -9;
    CHECK(wubu_sae_match_batch(q, 1, atoms, 3, 0, &oi, &od) == 0 && oi == 1 && od == 0,
          "SAE batch: max_dist accepts the exact hit");
    memset(q, 0, sizeof q); q[5] = 0xFF;
    CHECK(wubu_sae_match_batch(q, 1, atoms, 3, 0, &oi, &od) == 0 && oi == -1,
          "SAE batch: unmatched query reports -1");

    float vec[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint8_t sig[32];
    wubu_sae_signature(vec, 4, sig);
    CHECK(sig[0] == 0x0C && sig[1] == 0,
          "SAE signature sets bits above the mean only");

    /* --- foveated ROI selection (wave 7) --- */
    wubu_img blank = {0};
    wubu_img_alloc(&blank, 128, 128);
    wubu_img_fill(&blank, 40);
    wubu_roi rois[4];
    CHECK(wubu_foveated_rois(&blank, 16, 2, 1, rois, 4) == 0,
          "ROI: a featureless frame yields no regions");

    wubu_img busy = {0};
    wubu_img_alloc(&busy, 128, 128);
    wubu_img_fill(&busy, 40);
    for (int y = 16; y < 48; y++)
        for (int x = 16; x < 48; x++)
            busy.data[y * busy.stride + x] = ((x / 2 + y / 2) & 1) ? 220 : 30;
    for (int y = 80; y < 112; y++)
        for (int x = 80; x < 112; x++)
            busy.data[y * busy.stride + x] = ((x / 2 + y / 2) & 1) ? 220 : 30;

    int nr = wubu_foveated_rois(&busy, 16, 2, 1, rois, 4);
    printf("  rois: %d (first %d,%d %dx%d)\n", nr,
           nr > 0 ? rois[0].x : -1, nr > 0 ? rois[0].y : -1,
           nr > 0 ? rois[0].w : 0, nr > 0 ? rois[0].h : 0);
    CHECK(nr == 2, "ROI: two busy squares select two regions");
    int hit_a = 0, hit_b = 0;
    for (int i = 0; i < nr; i++) {
        int cx = rois[i].x + rois[i].w / 2, cy = rois[i].y + rois[i].h / 2;
        if (cx >= 16 && cx < 48 && cy >= 16 && cy < 48) hit_a = 1;
        if (cx >= 80 && cx < 112 && cy >= 80 && cy < 112) hit_b = 1;
    }
    CHECK(hit_a && hit_b, "ROI: both regions are covered by the selected tiles");
    CHECK(wubu_foveated_rois(&busy, 16, 1, 1, rois, 4) == 1, "ROI: k caps the selection");
    CHECK(wubu_foveated_rois(NULL, 16, 1, 1, rois, 4) == -1, "ROI: NULL image -> -1");
    wubu_img_free(&blank); wubu_img_free(&busy);


    /* --- pyramid LK: motion beyond the single-scale limit (wave 8) --- */
    wubu_img big = {0}, bigm = {0};
    wubu_img_alloc(&big, 192, 192);
    wubu_img_alloc(&bigm, 192, 192);
    for (int y = 0; y < 192; y++) {
        for (int x = 0; x < 192; x++) {
            double v = 128.0 + 55.0 * sin(x * 0.29) + 45.0 * cos(y * 0.23)
                     + 20.0 * sin((x + y) * 0.11);
            if (v < 0.0) v = 0.0;
            if (v > 255.0) v = 255.0;
            big.data[y * big.stride + x] = (uint8_t)v;
        }
    }
    for (int y = 0; y < 192; y++)
        for (int x = 0; x < 192; x++)
            bigm.data[y * bigm.stride + x] = wubu_img_at(&big, x - 6, y - 7);

    wubu_keypoint bp[2];
    bp[0].x = 96; bp[0].y = 96;
    bp[1].x = 120; bp[1].y = 80;
    float pdx[2], pdy[2];
    CHECK(wubu_lk_flow_pyr(&big, &bigm, bp, 2, 3, 15, 10, pdx, pdy) == 0, "pyramid LK runs");
    printf("  pyramid flow: (%.2f,%.2f) (%.2f,%.2f)\n", pdx[0], pdy[0], pdx[1], pdy[1]);
    int pyr_ok = 1;
    for (int i = 0; i < 2; i++)
        if (fabsf(pdx[i] - 6.0f) > 0.8f || fabsf(pdy[i] - 7.0f) > 0.8f) pyr_ok = 0;
    CHECK(pyr_ok, "pyramid LK recovers a (6,7) shift within 0.8 px");
    CHECK(wubu_lk_flow_pyr(&big, &bigm, bp, 2, 0, 15, 10, pdx, pdy) == -1,
          "levels=0 -> -1");
    wubu_img_free(&big); wubu_img_free(&bigm);


    /* --- Harris scoring + dedupe (wave 9) --- */
    wubu_keypoint hk[64];
    int hn = wubu_fast9(&sqimg, 40, 3, hk, 64);
    CHECK(wubu_harris_score(&sqimg, hk, hn) == 0, "Harris scoring runs");
    int scored = 0;
    for (int i = 0; i < hn; i++) if (hk[i].score > 0) scored++;
    CHECK(scored > 0, "corner responses are positive");

    wubu_keypoint dup[4];
    memset(dup, 0, sizeof dup);
    dup[0].x = 30; dup[0].y = 30; dup[0].score = 100;
    dup[1].x = 32; dup[1].y = 31; dup[1].score = 50;
    dup[2].x = 60; dup[2].y = 60; dup[2].score = 70;
    dup[3].x = 90; dup[3].y = 90; dup[3].score = 10;
    int dn = wubu_keypoint_dedupe(dup, 4, 5);
    CHECK(dn == 3 && dup[0].score == 100 && dup[1].score == 70,
          "dedupe keeps the strongest of a cluster and sorts by score");
    CHECK(wubu_harris_score(NULL, hk, hn) == -1, "Harris NULL -> -1");


    /* --- contour point tracing (wave 10) --- */
    wubu_img tr = {0}, te = {0};
    wubu_img_alloc(&tr, 64, 64);
    wubu_img_fill(&tr, 20);
    for (int y = 16; y < 48; y++)
        for (int x = 16; x < 48; x++) tr.data[y * tr.stride + x] = 220;
    wubu_img_alloc(&te, 64, 64);
    wubu_canny(&tr, 50, 150, &te);

    int sx0 = -1, sy0 = -1;
    for (int y = 0; y < 64 && sy0 < 0; y++)
        for (int x = 0; x < 64; x++) if (te.data[y * 64 + x]) { sx0 = x; sy0 = y; break; }

    int tx[512], ty[512];
    int tn = (sx0 >= 0) ? wubu_contour_trace(&te, sx0, sy0, tx, ty, 512) : -1;
    printf("  traced boundary points: %d (seed %d,%d)\n", tn, sx0, sy0);
    CHECK(tn > 20, "contour trace returns a boundary polyline");
    int all_on_edge = 1;
    for (int i = 0; i < tn; i++)
        if (!te.data[ty[i] * 64 + tx[i]]) all_on_edge = 0;
    CHECK(all_on_edge, "every traced point is an edge pixel");

    if (tn > 3) {
        double cx = 0, cy = 0;
        for (int i = 0; i < tn; i++) { cx += tx[i]; cy += ty[i]; }
        cx /= tn; cy /= tn;
        int inversions = 0;
        double prev = atan2(ty[0] - cy, tx[0] - cx);
        for (int i = 1; i < tn; i++) {
            double a = atan2(ty[i] - cy, tx[i] - cx);
            if (a < prev - 1e-9) inversions++;
            prev = a;
        }
        CHECK(inversions <= 1, "points are ordered around the centroid (one wrap allowed)");
        CHECK(abs(tx[0] - tx[tn - 1]) + abs(ty[0] - ty[tn - 1]) <= 3,
              "the polyline closes back near its start");
    }
    CHECK(wubu_contour_trace(&te, 0, 0, tx, ty, 512) == 0,
          "tracing an empty pixel returns 0 points");
    CHECK(wubu_contour_trace(NULL, 0, 0, tx, ty, 512) == -1, "NULL edges -> -1");
    wubu_img_free(&tr); wubu_img_free(&te);


    /* --- burst super-resolution (wave 11) --- */
    const int FW = 64, FH = 64;
    wubu_img fr[4];
    memset(fr, 0, sizeof fr);
    for (int i = 0; i < 4; i++) wubu_img_alloc(&fr[i], FW, FH);

    static const float odx[4] = {0.00f, 0.50f, 0.25f, 0.75f};
    static const float ody[4] = {0.00f, 0.50f, 0.75f, 0.25f};

    /* frame_i(x,y) samples the same 2x finer scene at (x + odx, y + ody). */
    for (int i = 0; i < 4; i++) {
        for (int y = 0; y < FH; y++) {
            for (int x = 0; x < FW; x++) {
                double sx = 2.0 * ((double)x + odx[i]);
                double sy = 2.0 * ((double)y + ody[i]);
                double v = 128.0 + 60.0 * sin(sx * 0.21) + 50.0 * cos(sy * 0.17);
                if (v < 0.0) v = 0.0;
                if (v > 255.0) v = 255.0;
                fr[i].data[y * fr[i].stride + x] = (uint8_t)v;
            }
        }
    }

    const wubu_img *cf[4] = {&fr[0], &fr[1], &fr[2], &fr[3]};
    wubu_img sr = {0};
    CHECK(wubu_burst_super_resolve(cf, odx, ody, 4, 2, &sr) == 0, "burst SR runs");
    CHECK(sr.w == 128 && sr.h == 128, "burst SR doubles the grid");

    wubu_img naive = {0};
    wubu_uplift2x(&fr[0], &naive);

    double err_sr = 0, err_naive = 0;
    int cnt = 0;
    for (int y = 4; y < 124; y++) {
        for (int x = 4; x < 124; x++) {
            double sx = (double)x, sy = (double)y;
            double v = 128.0 + 60.0 * sin(sx * 0.21) + 50.0 * cos(sy * 0.17);
            if (v < 0.0) v = 0.0;
            if (v > 255.0) v = 255.0;
            double a = (double)sr.data[y * sr.stride + x] - v;
            double b = (double)naive.data[y * naive.stride + x] - v;
            err_sr += a * a;
            err_naive += b * b;
            cnt++;
        }
    }
    err_sr = sqrt(err_sr / cnt);
    err_naive = sqrt(err_naive / cnt);
    printf("  burst RMSE: sr=%.3f naive=%.3f\n", err_sr, err_naive);
    CHECK(err_sr < err_naive, "burst SR beats naive 2x upsampling in RMSE");
    CHECK(err_sr < 6.0, "burst SR stays within 6 grey levels RMS");

    for (int i = 0; i < 4; i++) wubu_img_free(&fr[i]);
    wubu_img_free(&sr); wubu_img_free(&naive);


    /* --- GlassBox 64-D signature (wave 12) --- */
    wubu_img ones = {0};
    wubu_img_alloc(&ones, 16, 16);
    wubu_img_fill(&ones, 1);
    float gs[WUBU_GLASS_DIMS];
    CHECK(wubu_glassbox64(&ones, gs) == 0, "glassbox runs");
    CHECK(fabsf(gs[0] - 1.0f) < 1e-6f, "bitplane 0 is fully occupied for value 1");
    CHECK(gs[1] == 0.0f && gs[7] == 0.0f, "higher bitplanes are empty");
    int in_range = 1;
    for (int i = 0; i < WUBU_GLASS_DIMS; i++)
        if (gs[i] < 0.0f || gs[i] > 1.0f) in_range = 0;
    CHECK(in_range, "every coordinate lands in [0,1]");
    int reserved_zero = 1;
    for (int i = 32; i < 40; i++) if (gs[i] != 0.0f) reserved_zero = 0;
    CHECK(reserved_zero, "reserved worldtube block is honestly zero, not invented");

    float gs2[WUBU_GLASS_DIMS];
    wubu_glassbox64(&sqimg, gs2);
    float gs3[WUBU_GLASS_DIMS];
    wubu_glassbox64(&sqimg, gs3);
    CHECK(memcmp(gs2, gs3, sizeof gs2) == 0, "signature is deterministic");

    uint64_t gh0 = wubu_glassbox_hash(&sqimg);
    wubu_img grot = {0};
    wubu_img_alloc(&grot, sqimg.h, sqimg.w);
    for (int y = 0; y < sqimg.h; y++)
        for (int x = 0; x < sqimg.w; x++)
            grot.data[(size_t)x * grot.stride + (size_t)(sqimg.h - 1 - y)] =
                sqimg.data[(size_t)y * sqimg.stride + x];
    uint64_t gh1 = wubu_glassbox_hash(&grot);
    CHECK(gh0 != 0 && gh0 == gh1, "glassbox hash is D4-invariant");

    wubu_img other = {0};
    wubu_img_alloc(&other, 64, 64);
    wubu_img_fill(&other, 200);
    CHECK(wubu_glassbox_hash(&other) != gh0, "different content hashes differently");
    CHECK(wubu_glassbox64(NULL, gs) == -1, "NULL image -> -1");

    wubu_img_free(&ones); wubu_img_free(&grot); wubu_img_free(&other);


    /* --- dense block-matching flow (wave 14) --- */
    wubu_img bm_a = {0}, bm_b = {0};
    wubu_img_alloc(&bm_a, 128, 128);
    wubu_img_alloc(&bm_b, 128, 128);
    unsigned bs = 99;
    for (int i = 0; i < 128 * 128; i++) { bs = bs * 1664525u + 1013904223u; bm_a.data[i] = (uint8_t)(bs >> 24); }
    for (int y = 0; y < 128; y++)
        for (int x = 0; x < 128; x++)
            bm_b.data[y * bm_b.stride + x] = wubu_img_at(&bm_a, x - 3, y - 4);

    int16_t bfx[128], bfy[128];
    int nb = wubu_block_flow(&bm_a, &bm_b, 16, 6, bfx, bfy);
    CHECK(nb == 64, "block flow covers an 8x8 block grid");
    /* Border blocks lose content to clamping, so measure the interior. */
    int agree = 0, interior = 0;
    for (int by = 1; by < 7; by++) {
        for (int bx = 1; bx < 7; bx++) {
            int i = by * 8 + bx;
            interior++;
            if (bfx[i] == 3 && bfy[i] == 4) agree++;
        }
    }
    printf("  block flow agreement: %d/%d interior\n", agree, interior);
    CHECK(agree >= (interior * 9) / 10, ">=90% of interior blocks recover the (3,4) shift");
    CHECK(wubu_block_flow(&bm_a, &bm_b, 15, 6, bfx, bfy) == -1, "odd block size -> -1");
    wubu_img_free(&bm_a); wubu_img_free(&bm_b);


    /* --- sub-pixel block flow (wave 15) --- */
    wubu_img sp_a = {0}, sp_s = {0}, sp_b = {0};
    wubu_img_alloc(&sp_a, 128, 128);
    wubu_img_alloc(&sp_s, 128, 128);
    wubu_img_alloc(&sp_b, 128, 128);
    unsigned ss = 1234;
    for (int i = 0; i < 128 * 128; i++) { ss = ss * 1664525u + 1013904223u; sp_a.data[i] = (uint8_t)(ss >> 24); }
    wubu_gauss3(&sp_a, &sp_s);
    for (int y = 0; y < 128; y++)
        for (int x = 0; x < 128; x++)
            sp_b.data[y * sp_b.stride + x] = (uint8_t)(bilin(&sp_s, (float)x - 3.4f, (float)y - 4.6f) + 0.5f);

    float sfx[128], sfy[128];
    int snb = wubu_block_flow_subpix(&sp_s, &sp_b, 16, 8, sfx, sfy);
    CHECK(snb == 64, "sub-pixel flow covers the block grid");
    double ex = 0.0, ey = 0.0;
    int cntb = 0;
    for (int by = 1; by < 7; by++)
        for (int bx = 1; bx < 7; bx++) {
            int i = by * 8 + bx;
            ex += fabs((double)sfx[i] - 3.4);
            ey += fabs((double)sfy[i] - 4.6);
            cntb++;
        }
    ex /= cntb; ey /= cntb;
    printf("  sub-pixel flow mean error: dx=%.3f dy=%.3f\n", ex, ey);
    CHECK(ex < 0.35 && ey < 0.35, "sub-pixel flow recovers (3.4,4.6) within 0.35 px");
    CHECK(wubu_block_flow_subpix(&sp_s, &sp_b, 16, 20, sfx, sfy) == -1, "search > 16 -> -1");
    wubu_img_free(&sp_a); wubu_img_free(&sp_s); wubu_img_free(&sp_b);


    /* --- dense Horn-Schunck flow (wave 16) --- */
    wubu_img hs_a = {0}, hs_b = {0};
    wubu_img_alloc(&hs_a, 128, 128);
    wubu_img_alloc(&hs_b, 128, 128);
    for (int y = 0; y < 128; y++)
        for (int x = 0; x < 128; x++) {
            double v = 128.0 + 60.0 * sin(x * 0.19) + 45.0 * cos(y * 0.23) + 15.0 * sin((x + y) * 0.07);
            hs_a.data[y * hs_a.stride + x] = (uint8_t)v;
        }
    for (int y = 0; y < 128; y++)
        for (int x = 0; x < 128; x++)
            hs_b.data[y * hs_b.stride + x] = (uint8_t)(bilin(&hs_a, (float)x - 2.0f, (float)y - 3.0f) + 0.5f);

    float hdx[128 * 128], hdy[128 * 128];
    CHECK(wubu_horn_schunck(&hs_a, &hs_b, 300, 8.0f, hdx, hdy) == 0, "Horn-Schunck runs");
    double hex = 0.0, hey = 0.0;
    int hc = 0;
    for (int y = 6; y < 122; y++)
        for (int x = 6; x < 122; x++) {
            hex += fabs((double)hdx[y * 128 + x] - 2.0);
            hey += fabs((double)hdy[y * 128 + x] - 3.0);
            hc++;
        }
    hex /= hc; hey /= hc;
    printf("  dense HS mean error: dx=%.3f dy=%.3f\n", hex, hey);
    CHECK(hex < 0.5 && hey < 0.5, "dense HS recovers (2,3) within 0.5 px");

    float hdx0[128 * 128], hdy0[128 * 128];
    wubu_horn_schunck(&hs_a, &hs_a, 100, 8.0f, hdx0, hdy0);
    double zero_err = 0.0;
    for (int i = 0; i < 128 * 128; i++) zero_err += fabs(hdx0[i]) + fabs(hdy0[i]);
    CHECK(zero_err < 1e-3, "identical frames give a zero field");
    CHECK(wubu_horn_schunck(&hs_a, &hs_b, 0, 8.0f, hdx, hdy) == -1, "iters=0 -> -1");
    wubu_img_free(&hs_a); wubu_img_free(&hs_b);


    /* --- dense Harris corners / checkerboard lattice (wave 18) --- */
    wubu_img cb = {0};
    wubu_img_alloc(&cb, 160, 160);
    for (int y = 0; y < 160; y++)
        for (int x = 0; x < 160; x++)
            cb.data[y * cb.stride + x] = ((x / 20 + y / 20) & 1) ? 220 : 30;

    wubu_corner cc[128];
    int ccn = wubu_harris_corners(&cb, 0.02f, 5, cc, 128);
    printf("  checkerboard corners: %d\n", ccn);
    CHECK(ccn >= 36, "checkerboard yields at least 36 corners");

    int on_lattice = 0;
    for (int i = 0; i < ccn; i++) {
        int mx = (cc[i].x + 10) / 20 * 20;
        int my = (cc[i].y + 10) / 20 * 20;
        if (abs(cc[i].x - mx) <= 2 && abs(cc[i].y - my) <= 2) on_lattice++;
    }
    printf("  on lattice: %d/%d\n", on_lattice, ccn);
    CHECK(on_lattice == ccn, "every corner sits within 2 px of the cell lattice");
    CHECK(wubu_harris_corners(&cb, 1.0f, 5, cc, 128) == -1, "ratio 1.0 -> -1");


    /* --- chessboard grid ordering (wave 19) --- */
    wubu_corner shuf[49], ordered[49];
    for (int i = 0; i < 49; i++) {           /* deterministic shuffle */
        int j = (i * 17 + 11) % 49;
        shuf[i] = cc[j];
    }
    int on = wubu_chessboard_order(shuf, 49, 7, 7, ordered);
    CHECK(on == 49, "ordering accepts a 7x7 corner set in shuffled order");
    int x_ok = 1, y_ok = 1;
    for (int r = 0; r < 7; r++) {
        for (int c = 1; c < 7; c++) {
            int a = r * 7 + c - 1, b = r * 7 + c;
            if (ordered[b].x <= ordered[a].x) x_ok = 0;
            if (abs(ordered[b].y - ordered[a].y) > 2) y_ok = 0;
        }
        if (r) if (ordered[r * 7].y <= ordered[(r - 1) * 7].y) y_ok = 0;
    }
    CHECK(x_ok, "within each row x increases");
    CHECK(y_ok, "rows are ordered top to bottom");
    printf("  grid origin (%d,%d) far corner (%d,%d)\n",
           ordered[0].x, ordered[0].y, ordered[48].x, ordered[48].y);
    CHECK(abs(ordered[0].x - 20) <= 2 && abs(ordered[48].x - 140) <= 2,
          "grid spans the expected lattice");
    CHECK(wubu_chessboard_order(shuf, 48, 7, 7, ordered) == -1, "count mismatch -> -1");


    /* --- sub-pixel corner refinement (wave 20) --- */
    wubu_img cb2 = {0};
    wubu_img_alloc(&cb2, 160, 160);
    for (int y = 0; y < 160; y++)
        for (int x = 0; x < 160; x++)
            cb2.data[y * cb2.stride + x] = ((x / 20 + y / 20) & 1) ? 220 : 30;
    /* render the same board shifted by a known sub-pixel offset */
    wubu_img cb3 = {0};
    wubu_img_alloc(&cb3, 160, 160);
    for (int y = 0; y < 160; y++)
        for (int x = 0; x < 160; x++)
            cb3.data[y * cb3.stride + x] = (uint8_t)(bilin(&cb2, (float)x - 0.4f, (float)y - 0.4f) + 0.5f);

    wubu_corner sc[128];
    int scn = wubu_harris_corners(&cb3, 0.02f, 5, sc, 128);
    CHECK(scn >= 36, "shifted board still yields corners");
    CHECK(wubu_corner_refine_subpix(&cb3, sc, scn, 3) == 0, "sub-pixel refinement runs");
    int near = 0;
    for (int i = 0; i < scn; i++) {
        int mx = (sc[i].x + 10) / 20 * 20;
        int my = (sc[i].y + 10) / 20 * 20;
        if (abs(sc[i].x - mx) <= 1 && abs(sc[i].y - my) <= 1) near++;
    }
    printf("  refined corners within 1 px of lattice: %d/%d\n", near, scn);
    CHECK(near >= (scn * 9) / 10, ">=90% of refined corners land within 1 px");
    CHECK(wubu_corner_refine_subpix(&cb3, sc, scn, 0) == -1, "radius 0 -> -1");
    wubu_img_free(&cb2); wubu_img_free(&cb3);


    /* --- true sub-pixel localisation (wave 21) --- */
    wubu_img cb4 = {0};
    wubu_img_alloc(&cb4, 160, 160);
    for (int y = 0; y < 160; y++)
        for (int x = 0; x < 160; x++) {
            int cxcell = (int)floor((x - 0.4) / 20.0);
            int cycell = (int)floor((y - 0.4) / 20.0);
            cb4.data[y * cb4.stride + x] = ((cxcell + cycell) & 1) ? 220 : 30;
        }

    wubu_corner fc[128];
    int fcn = wubu_harris_corners(&cb4, 0.02f, 5, fc, 128);
    wubu_cornerf fout[128];
    CHECK(wubu_corner_refine_centroid(&cb4, fc, fcn, 3, fout) == 0, "float centroid runs");

    double fx_err = 0.0, fy_err = 0.0;
    int fcount = 0;
    for (int i = 0; i < fcn; i++) {
        int mx = (int)((fout[i].x + 10) / 20) * 20;
        int my = (int)((fout[i].y + 10) / 20) * 20;
        if (mx < 20 || mx > 140 || my < 20 || my > 140) continue;
        fx_err += fabs((double)fout[i].x - ((double)mx + 0.4));
        fy_err += fabs((double)fout[i].y - ((double)my + 0.4));
        fcount++;
    }
    if (fcount) { fx_err /= fcount; fy_err /= fcount; }
    printf("  float centroid error vs (lattice+0.4): dx=%.3f dy=%.3f (n=%d)\n", fx_err, fy_err, fcount);
    CHECK(fcount >= 36 && fx_err < 0.30 && fy_err < 0.30,
          "float centroids land within 0.30 px of the true sub-pixel corner");
    CHECK(wubu_corner_refine_centroid(&cb4, fc, fcn, 0, fout) == -1, "radius 0 -> -1");
    wubu_img_free(&cb4);


    /* --- PnP by normalised DLT (wave 22) --- */
    const double P0[12] = {500, 0, 80, 0,  0, 500, 80, 0,  0, 0, 1, 6};
    double p3[24] = {-1,-1,0,  1,-1,0,  1,1,0,  -1,1,0,
                     0,0,1,    0,0,-1,  -1,1,2,   1,-1,2};
    double p2d[16];
    for (int i = 0; i < 8; i++) {
        double X = p3[3*i], Y = p3[3*i+1], Z = p3[3*i+2];
        double w = P0[8]*X + P0[9]*Y + P0[10]*Z + P0[11];
        p2d[2*i]   = (P0[0]*X + P0[1]*Y + P0[2]*Z + P0[3]) / w;
        p2d[2*i+1] = (P0[4]*X + P0[5]*Y + P0[6]*Z + P0[7]) / w;
    }
    double Pest[12];
    CHECK(wubu_pnp_dlt(p3, p2d, 8, Pest) == 0, "DLT PnP solves an 8-point system");
    double rms = wubu_pnp_rmse(p3, p2d, 8, Pest);
    printf("  PnP reprojection RMS: %.3g px\n", rms);
    CHECK(rms < 1e-6, "DLT recovers the camera to machine precision on exact data");
    CHECK(wubu_pnp_dlt(p3, p2d, 5, Pest) == -1, "5 points -> -1");


    /* --- RQ decomposition to K/R/t (wave 23) --- */
    double K9[9], R9[9], t3[3];
    CHECK(wubu_pnp_decompose(Pest, K9, R9, t3) == 0, "RQ decomposition runs");
    const double Kexp[9] = {500, 0, 80,  0, 500, 80,  0, 0, 1};
    double kerr = 0.0;
    for (int i = 0; i < 9; i++) kerr += fabs(K9[i] - Kexp[i]);
    printf("  K error=%.3g  t=(%.3f,%.3f,%.3f)\n", kerr, t3[0], t3[1], t3[2]);
    CHECK(kerr < 1e-6, "intrinsics recovered exactly from the DLT matrix");

    double rerr = 0.0;
    for (int i = 0; i < 9; i++) rerr += fabs(R9[i] - (i % 4 == 0 ? 1.0 : 0.0));
    CHECK(rerr < 1e-6, "rotation is identity for an axis-aligned camera");

    /* P0's translation column is (0,0,6), so K^-1 p4 = (-0.96,-0.96,6). */
    CHECK(fabs(t3[0] + 0.96) < 1e-6 && fabs(t3[1] + 0.96) < 1e-6 &&
          fabs(t3[2] - 6.0) < 1e-6, "translation is K^-1 p4, exactly as defined");
    CHECK(wubu_pnp_decompose(NULL, K9, R9, t3) == -1, "NULL P -> -1");


    /* --- homography + perspective warp (wave 24) --- */
    wubu_img cbw = {0};
    wubu_img_alloc(&cbw, 160, 160);
    for (int y = 0; y < 160; y++)
        for (int x = 0; x < 160; x++)
            cbw.data[y * cbw.stride + x] = ((x / 20 + y / 20) & 1) ? 220 : 30;

    double wsrc[8] = {40, 40, 80, 40, 80, 80, 40, 80};
    double wdst[8] = {0, 0, 40, 0, 40, 40, 0, 40};
    double Hw[9];
    CHECK(wubu_homography(wsrc, wdst, Hw) == 0, "homography from four pairs");
    CHECK(fabs(Hw[0] - 1.0) < 1e-9 && fabs(Hw[2] + 40.0) < 1e-9,
          "H is the expected translation for this quad");

    wubu_img wrp = {0};
    CHECK(wubu_warp_perspective(&cbw, Hw, &wrp, 40, 40) == 0, "warp runs");
    CHECK(wrp.w == 40 && wrp.h == 40, "warp produces the requested size");

    int wbad = 0, wtot = 0;
    for (int y = 5; y < 40; y += 5)
        for (int x = 5; x < 40; x += 5) {
            wtot++;
            if (wubu_img_at(&wrp, x, y) != wubu_img_at(&cbw, x + 40, y + 40)) wbad++;
        }
    printf("  warp mismatches: %d/%d\n", wbad, wtot);
    CHECK(wbad == 0, "every sampled pixel matches the source under H");
    CHECK(wubu_homography(NULL, wdst, Hw) == -1, "NULL src -> -1");
    wubu_img_free(&cbw); wubu_img_free(&wrp);


    /* --- ArUco-style marker decode (wave 25) --- */
    const uint16_t adict[4] = {0x1234, 0xEDCB, 0x0F0F, 0xF0F0};
    wubu_img am = {0};
    wubu_img_alloc(&am, 60, 60);
    wubu_img_fill(&am, 0);
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            if (!((adict[2] >> (15 - (r * 4 + c))) & 1)) continue;
            for (int y = (r + 1) * 10; y < (r + 2) * 10; y++)
                for (int x = (c + 1) * 10; x < (c + 2) * 10; x++)
                    am.data[y * am.stride + x] = 255;
        }
    double aquad[8] = {0, 0, 59, 0, 59, 59, 0, 59};
    wubu_aruco_marker amk;
    CHECK(wubu_aruco_decode(&am, aquad, adict, 4, 0, &amk) == 0, "marker decodes");
    CHECK(amk.id == 2 && amk.hamming == 0 && amk.bits == 0x0F0F,
          "decoded id, bits and zero Hamming distance are exact");

    wubu_img ablank = {0};
    wubu_img_alloc(&ablank, 60, 60);
    wubu_img_fill(&ablank, 0);
    CHECK(wubu_aruco_decode(&ablank, aquad, adict, 4, 0, &amk) == -1,
          "an all-black payload matches nothing at Hamming 0");
    CHECK(wubu_aruco_decode(NULL, aquad, adict, 4, 0, &amk) == -1, "NULL image -> -1");
    wubu_img_free(&am); wubu_img_free(&ablank);

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