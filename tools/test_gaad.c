/*
 * test_gaad.c — tests for the GAAD Golden Aspect Adaptive Decomposition.
 *
 * Port of WuBuOS/src/kernel/wubu_gaad_test.c. Tests golden subdivision,
 * feng shui mirrors, phi-spiral sectoring, resolution translation,
 * and pure-C math helpers.
 *
 * C11, no external deps.
 */
#include "wubu_gaad.h"
#include <stdio.h>
#include <math.h>

static int tests_run = 0;
static int tests_pass = 0;

static void check(const char *name, int cond) {
    tests_run++;
    if (cond) { tests_pass++; printf("  PASS: %s\n", name); }
    else      { printf("  FAIL: %s\n", name); }
}

static int approx(double a, double b, double tol) {
    return fabs(a - b) < tol;
}

int main(void) {
    printf("=== test_gaad: Golden Aspect Adaptive Decomposition ===\n");

    /* ---- Test 1: Golden Subdivision ---- */
    printf("\n--- Test 1: Golden Subdivision ---\n");

    /* Square frame → single square */
    {
        WubuGaadDecomp d;
        wubu_gaad_decompose(640, 640, 6, &d);
        check("square 640x640 produces regions", d.n_regions > 0);
        check("square frame_w set", d.frame_w == 640);
        check("square frame_h set", d.frame_h == 640);
        check("first region is square", d.regions[0].kind == WUBU_GAAD_SQUARE);
        check("first region covers full frame", d.regions[0].w == 640 && d.regions[0].h == 640);
    }

    /* Landscape frame → golden subdivision */
    {
        WubuGaadDecomp d;
        wubu_gaad_decompose(1920, 1080, 6, &d);
        check("landscape produces regions", d.n_regions > 0);
        check("first region is square", d.regions[0].kind == WUBU_GAAD_SQUARE);
        check("first region square w==h", d.regions[0].w == d.regions[0].h);
        check("first region w=1080 (shorter side)", d.regions[0].w == 1080);
        check("regions within max", d.n_regions <= WUBU_GAAD_MAX_REGIONS);

        /* Depth 0 region should have phi_scale = φ^0 = 1 */
        check("depth 0 phi_scale = 1", approx(d.regions[0].phi_scale, 1.0, 1e-9));
    }

    /* Portrait frame */
    {
        WubuGaadDecomp d;
        wubu_gaad_decompose(1080, 1920, 6, &d);
        check("portrait produces regions", d.n_regions > 0);
        check("first region is square", d.regions[0].kind == WUBU_GAAD_SQUARE);
        check("first region h=1080 (shorter side)", d.regions[0].h == 1080);
    }

    /* Small frame → no subdivisions */
    {
        WubuGaadDecomp d;
        wubu_gaad_decompose(2, 2, 6, &d);
        check("2x2 produces 0 regions", d.n_regions == 0);
    }

    /* NULL safety */
    {
        wubu_gaad_decompose(640, 480, 6, NULL);
        check("decompose NULL out is safe", 1);
    }

    /* ---- Test 2: Feng Shui Cardinal Mirrors ---- */
    printf("\n--- Test 2: Feng Shui Mirrors ---\n");

    {
        WubuFengShui fs;
        wubu_gaad_feng_shui_build(1920, 1080, &fs);

        /* All 13 regions should be filled */
        check("north region 0 non-zero", fs.north[0].w > 0);
        check("south region 0 non-zero", fs.south[0].w > 0);
        check("east region 0 non-zero", fs.east[0].w > 0);
        check("west region 0 non-zero", fs.west[0].h > 0);
        check("center non-zero", fs.center.w > 0 && fs.center.h > 0);

        /* All snap targets */
        check("north[0] is_snap_target", fs.north[0].is_snap_target);
        check("south[0] is_snap_target", fs.south[0].is_snap_target);
        check("center is_snap_target", fs.center.is_snap_target);

        /* Cardinals set correctly */
        check("north cardinal=0 (N)", fs.north[0].cardinal == 0);
        check("east cardinal=1 (E)", fs.east[0].cardinal == 1);
        check("south cardinal=2 (S)", fs.south[0].cardinal == 2);
        check("west cardinal=3 (W)", fs.west[0].cardinal == 3);
        check("center cardinal=-1", fs.center.cardinal == -1);

        /* Regions tile the frame: top+middle+bottom rows cover full height */
        int top_h = fs.north[0].h;          /* top row */
        int bot_y = fs.south[0].y;           /* south starts here */
        int bot_h = fs.south[0].h;           /* bottom row height */
        int mid_h = bot_y - top_h;           /* middle (center row) */
        check("vertical tiling: top+middle+bottom = frame_h",
              top_h + mid_h + bot_h == 1080);
    }

    {
        wubu_gaad_feng_shui_build(1920, 1080, NULL);
        check("feng_shui NULL fs is safe", 1);
    }

    /* ---- Test 3: Phi-Spiral Sectoring ---- */
    printf("\n--- Test 3: Phi-Spiral ---\n");

    {
        WubuGaadDecomp d;
        wubu_gaad_decompose(800, 600, 6, &d);
        int before = d.n_regions;
        wubu_gaad_add_spirals(&d, 4, 10);
        check("spirals add regions", d.n_regions > before);
        check("spiral regions are points",
              d.regions[before].kind == WUBU_GAAD_SPIRAL_PT ||
              d.regions[before].w == 1);
        check("spiral within bounds", d.n_regions <= WUBU_GAAD_MAX_REGIONS);
    }

    {
        wubu_gaad_add_spirals(NULL, 4, 10);
        check("spirals NULL is safe", 1);
    }

    /* ---- Test 4: Resolution Translation ---- */
    printf("\n--- Test 4: Resolution Translation ---\n");

    {
        WubuGaadTranslate t;
        wubu_gaad_translate_init(640, 480, 1920, 1080, 6, &t);
        check("translate src_w", t.src_w == 640);
        check("translate src_h", t.src_h == 480);
        check("translate dst_w", t.dst_w == 1920);
        check("translate dst_h", t.dst_h == 1080);
        check("src_decomp has regions", t.src_decomp.n_regions > 0);
        check("dst_decomp has regions", t.dst_decomp.n_regions > 0);

        /* Pixel translation: corner maps to corner */
        int dx, dy;
        wubu_gaad_translate_pixel(&t, 0, 0, &dx, &dy);
        check("origin maps to origin region", dx >= 0 && dy >= 0);

        /* (639, 479) -> maps to (1919, 1079) approximately */
        wubu_gaad_translate_pixel(&t, 639, 479, &dx, &dy);
        check("far corner maps to far area", dx > 1000 && dy > 900);
    }

    {
        /* NULL safety */
        wubu_gaad_translate_pixel(NULL, 0, 0, NULL, NULL);
        wubu_gaad_translate_init(640, 480, 1920, 1080, 6, NULL);
        check("translate NULL safety", 1);
    }

    /* ---- Test 5: Pure C Math ---- */
    printf("\n--- Test 5: Pure C Math ---\n");

    check("isqrt(0) = 0", wubu_isqrt(0) == 0);
    check("isqrt(1) = 1", wubu_isqrt(1) == 1);
    check("isqrt(4) = 2", wubu_isqrt(4) == 2);
    check("isqrt(16) = 4", wubu_isqrt(16) == 4);
    check("isqrt(25) = 5", wubu_isqrt(25) == 5);
    check("isqrt(100) = 10", wubu_isqrt(100) == 10);
    check("isqrt(2) = 1", wubu_isqrt(2) == 1);
    check("isqrt(24) = 4", wubu_isqrt(24) == 4);

    check("dist(0,0,3,4) = 5", wubu_dist(0, 0, 3, 4) == 5);
    check("dist(0,0,0,0) = 0", wubu_dist(0, 0, 0, 0) == 0);

    check("phi_pow(0) = 1", approx(wubu_phi_pow(0), 1.0, 1e-9));
    check("phi_pow(1) = φ", approx(wubu_phi_pow(1), WUBU_PHI, 1e-9));
    check("phi_pow(2) = φ²", approx(wubu_phi_pow(2), WUBU_PHI_SQ, 1e-9));

    check("clamp(5, 0, 10) = 5", wubu_clamp(5, 0, 10) == 5);
    check("clamp(-5, 0, 10) = 0", wubu_clamp(-5, 0, 10) == 0);
    check("clamp(15, 0, 10) = 10", wubu_clamp(15, 0, 10) == 10);

    /* ---- Test 6: snap_pos ---- */
    printf("\n--- Test 6: snap_pos ----\n");
    {
        WubuGaadDecomp d;
        wubu_gaad_decompose(1920, 1080, 6, &d);
        int x, y, w, h;
        wubu_gaad_snap_pos(&d, 0, &x, &y, &w, &h);
        check("snap_pos region 0 x", x == 0);
        check("snap_pos region 0 y", y == 0);
        check("snap_pos region 0 w", w > 0);

        /* NULL safety */
        wubu_gaad_snap_pos(NULL, 0, NULL, NULL, NULL, NULL);
        wubu_gaad_snap_pos(&d, -1, NULL, NULL, NULL, NULL);
        wubu_gaad_snap_pos(&d, 999, NULL, NULL, NULL, NULL);
        check("snap_pos NULL/invalid safe", 1);
    }

    printf("\n=== Results: %d/%d tests passed ===\n", tests_pass, tests_run);
    if (tests_pass == tests_run) {
        printf("ALL GAAD TESTS PASSED\n");
        return 0;
    }
    return 1;
}
