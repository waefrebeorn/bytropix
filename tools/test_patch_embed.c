/*
 * test_patch_embed.c — tests for ViT patch embedding kernel.
 *
 * Verifies:
 * - Patch extraction + projection produces correct output
 * - Positional embeddings are added
 * - Bias terms work
 * - forward_grid matches forward for identity projection
 * - Non-square patch divisibility is rejected
 */
#include "wubu_patch_embed.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0;
static int tests_pass = 0;
static int tests_fail = 0;

static void check(int cond, const char *name) {
    tests_run++;
    if (cond) { tests_pass++; printf("  PASS: %s\n", name); }
    else      { tests_fail++; printf("  FAIL: %s\n", name); }
}

int main(void) {
    printf("=== ViT Patch Embedding Tests ===\n\n");

    /* --- Test 1: Basic patch extraction + projection --- */
    /* Image: 4x4, 1 channel, patch=2, hidden=3
     * Expected: (4/2)^2 = 4 patches, each 2x2=4 elements -> 3-dim output */
    {
        float img[16] = {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        };
        /* proj_w: identity-like (3x4), proj_b: zeros */
        float proj_w[12] = {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0
        };
        float proj_b[3] = {0, 0, 0};
        float out[12];  /* 4 patches * 3 hidden */

        wubu_patch_embed_t *ctx = wubu_patch_embed_init(2, 1, 3, 4, proj_w, proj_b, NULL);
        check(ctx != NULL, "Basic init");
        check(wubu_patch_embed_num_patches(ctx) == 4, "Num patches = 4");

        int np = wubu_patch_embed_forward(ctx, img, out);
        check(np == 4, "Forward returns 4 patches");

        /* Patch 0 = top-left 2x2 = [1,2,5,6] -> proj: out = [1,2,5] */
        check(fabs(out[0] - 1.0f) < 1e-6, "Patch 0 elem 0 = 1");
        check(fabs(out[1] - 2.0f) < 1e-6, "Patch 0 elem 1 = 2");
        check(fabs(out[2] - 5.0f) < 1e-6, "Patch 0 elem 2 = 5");

        /* Patch 1 = top-right 2x2 = [3,4,7,8] -> proj: out = [3,4,7] */
        check(fabs(out[3] - 3.0f) < 1e-6, "Patch 1 elem 0 = 3");

        /* Patch 2 = bottom-left 2x2 = [9,10,13,14] -> proj: out = [9,10,13] */
        check(fabs(out[6] - 9.0f) < 1e-6, "Patch 2 elem 0 = 9");

        /* Patch 3 = bottom-right 2x2 = [11,12,15,16] -> proj: out = [11,12,15] */
        check(fabs(out[9] - 11.0f) < 1e-6, "Patch 3 elem 0 = 11");

        wubu_patch_embed_close(ctx);
    }

    /* --- Test 2: With bias --- */
    {
        float img[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
        float proj_w[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
        float proj_b[3] = {10, 20, 30};
        float out[12];

        wubu_patch_embed_t *ctx = wubu_patch_embed_init(2, 1, 3, 4,
                                                        proj_w, proj_b, NULL);
        check(ctx != NULL, "BIAS init");
        wubu_patch_embed_forward(ctx, img, out);
        check(fabs(out[0] - 11.0f) < 1e-6, "BIAS: elem0 = 1+10 = 11");
        check(fabs(out[1] - 22.0f) < 1e-6, "BIAS: elem1 = 2+20 = 22");
        check(fabs(out[2] - 35.0f) < 1e-6, "BIAS: elem2 = 5+30 = 35");
        wubu_patch_embed_close(ctx);
    }

    /* --- Test 3: With positional embeddings --- */
    {
        float img[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
        float proj_w[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
        float proj_b[3] = {0, 0, 0};
        float pos[12] = {100, 200, 300,   /* patch 0 pos */
                         10,  20,  30,   /* patch 1 pos */
                         1,   2,   3,   /* patch 2 pos */
                         0,   0,   0};  /* patch 3 pos */
        float out[12];

        wubu_patch_embed_t *ctx = wubu_patch_embed_init(2, 1, 3, 4,
                                                        proj_w, proj_b, pos);
        check(ctx != NULL, "POS init");
        wubu_patch_embed_forward(ctx, img, out);
        check(fabs(out[0] - 101.0f) < 1e-6, "POS: patch0 elem0 = 1+100 = 101");
        check(fabs(out[1] - 202.0f) < 1e-6, "POS: patch0 elem1 = 2+200 = 202");
        check(fabs(out[3] - 13.0f) < 1e-6, "POS: patch1 elem0 = 3+10 = 13");
        wubu_patch_embed_close(ctx);
    }

    /* --- Test 4: forward_grid matches forward --- */
    {
        float img[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
        /* proj_w: [3, 4] row-major — identity (first 3 elements of patch) */
        float proj_w[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
        float proj_b[3] = {5, 0, -5};
        float out_forward[12], out_grid[12];
        float grid[16];

        /* Manually flatten 4 patches into grid */
        /* Patch 0: [1,2,5,6], Patch 1: [3,4,7,8], Patch 2: [9,10,13,14], Patch 3: [11,12,15,16] */
        memcpy(&grid[0],  (float[4]){1,2,5,6},  16);
        memcpy(&grid[4],  (float[4]){3,4,7,8},  16);
        memcpy(&grid[8],  (float[4]){9,10,13,14}, 16);
        memcpy(&grid[12], (float[4]){11,12,15,16}, 16);

        wubu_patch_embed_t *ctx = wubu_patch_embed_init(2, 1, 3, 4,
                                                        proj_w, proj_b, NULL);
        check(ctx != NULL, "GRID init");

        wubu_patch_embed_forward(ctx, img, out_forward);
        wubu_patch_embed_forward_grid(ctx, grid, out_grid);

        int match = 1;
        for (int i = 0; i < 12; i++) {
            if (fabs(out_forward[i] - out_grid[i]) > 1e-6) {
                match = 0;
                printf("    Mismatch at [%d]: fwd=%.4f grid=%.4f\n", i,
                       out_forward[i], out_grid[i]);
            }
        }
        check(match, "forward_grid matches forward");
        wubu_patch_embed_close(ctx);
    }

    /* --- Test 5: Reject non-divisible image size --- */
    {
        float proj_w[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
        /* 5x5 image with patch=2 -> 5%2 != 0 -> should fail */
        wubu_patch_embed_t *ctx = wubu_patch_embed_init(2, 1, 3, 5, proj_w, NULL, NULL);
        check(ctx == NULL, "Reject non-divisible image size (5%2!=0)");
        /* Use a valid one to make sure we didn't break anything */
        wubu_patch_embed_t *ctx2 = wubu_patch_embed_init(2, 1, 3, 4, proj_w, NULL, NULL);
        check(ctx2 != NULL, "Valid 4x4 image still works");
        wubu_patch_embed_close(ctx2);
    }

    /* --- Test 6: NULL safety --- */
    {
        check(wubu_patch_embed_forward(NULL, (float*)1, (float*)1) == 0, "NULL ctx -> 0");
        check(wubu_patch_embed_forward((wubu_patch_embed_t*)1, NULL, (float*)1) == 0, "NULL img -> 0");
        check(wubu_patch_embed_forward((wubu_patch_embed_t*)1, (float*)1, NULL) == 0, "NULL out -> 0");
        /* num_patches on NULL */
        check(wubu_patch_embed_num_patches(NULL) == 0, "NULL ctx num_patches -> 0");
    }

    /* --- Test 7: Zero patch_size --- */
    {
        float proj_w[4] = {1, 0, 0, 1};
        wubu_patch_embed_t *ctx = wubu_patch_embed_init(0, 1, 1, 4, proj_w, NULL, NULL);
        check(ctx == NULL, "Reject zero patch_size");
    }

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_pass, tests_run, tests_fail);

    /* Clean up test artifacts */

    return tests_fail > 0 ? 1 : 0;
}
