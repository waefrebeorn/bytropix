/*
 * test_cv.c -- core + imgproc + features + uplift (solidcv reimplementation).
 */
#include "wubu_cv.h"
#include "wubu_features.h"
#include "wubu_uplift.h"
#include "wubu_flow.h"
#include "wubu_sae.h"
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