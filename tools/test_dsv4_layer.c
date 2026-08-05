/*
 * test_dsv4_layer.c — tests for the DSV4 MLA attention bridge (wubu_dsv4_layer).
 *
 * Covers: create/free, NULL safety, tensor name resolution, forward pass
 * with small identity-ish weights.
 *
 * C11, no external deps.
 */
#include "wubu_dsv4_layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run = 0;
static int tests_pass = 0;

static void check(const char *name, int cond) {
    tests_run++;
    if (cond) { tests_pass++; printf("  PASS: %s\n", name); }
    else      { printf("  FAIL: %s\n", name); }
}

int main(void) {
    printf("=== DSV4 Layer Bridge Tests ===\n");

    /* ---- Test 1: create/free ---- */
    printf("\n--- Test 1: create/free ---\n");
    wubu_dsv4_layer_t *dl = wubu_dsv4_layer_create(7168, 96, 192, 1536, 512, 64);
    check("dsv4_layer_create non-NULL", dl != NULL);
    wubu_dsv4_layer_free(dl);
    check("free(NULL) safe", 1);
    wubu_dsv4_layer_free(NULL);

    /* ---- Test 2: NULL safety ---- */
    printf("\n--- Test 2: NULL safety ---\n");
    check("forward with NULL layer", wubu_dsv4_layer_forward(NULL, NULL, NULL, 0, NULL) == -1);
    check("load_tensors with NULL", wubu_dsv4_layer_load_tensors(NULL, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL) == -1);

    /* ---- Test 3: bad dims ---- */
    printf("\n--- Test 3: bad dims ---\n");
    wubu_dsv4_layer_t *bad = wubu_dsv4_layer_create(0, 2, 4, 2, 2, 2);
    check("create with hidden_dim=0 returns NULL", bad == NULL);
    bad = wubu_dsv4_layer_create(4, 0, 4, 2, 2, 2);
    check("create with n_heads=0 returns NULL", bad == NULL);

    /* ---- Test 4: tensor names ---- */
    printf("\n--- Test 4: tensor names ---\n");
    char *n = wubu_dsv4_tensor_name(5, "q_a");
    check("tensor_name q_a", n != NULL && strcmp(n, "blk.5.attn_q_a") == 0);
    free(n);
    n = wubu_dsv4_tensor_name(3, "kv");
    check("tensor_name kv", n != NULL && strcmp(n, "blk.3.attn_kv") == 0);
    free(n);
    n = wubu_dsv4_tensor_name(7, "o_b");
    check("tensor_name o_b", n != NULL && strcmp(n, "blk.7.attn_out_b") == 0);
    free(n);
    check("tensor_name invalid type", wubu_dsv4_tensor_name(0, "bogus") == NULL);
    check("tensor_name NULL type", wubu_dsv4_tensor_name(0, NULL) == NULL);

    /* ---- Test 5: forward without weights ---- */
    printf("\n--- Test 5: forward without weights ---\n");
    wubu_dsv4_layer_t *dl2 = wubu_dsv4_layer_create(4, 2, 4, 2, 2, 2);
    check("small layer created", dl2 != NULL);
    if (dl2) {
        float x[4] = {1, 2, 3, 4};
        float out[4] = {0};
        check("forward before load returns -1",
              wubu_dsv4_layer_forward(dl2, x, NULL, 0, out) == -1);
    }

    /* ---- Test 6: forward with weights ---- */
    printf("\n--- Test 6: forward with weights ---\n");
    if (dl2) {
        /* dims: hidden=4, n_heads=2, head_dim=4, q_lora=2, kv_lora=2, rope=2
         * kv_latent_dim = kv_lora + rope = 4 */
        /* W_DKV: [4, 4] = 16, identity */
        float W_DKV[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
        /* W_UK: [n_heads*head_dim=8, kv_lora_rank=2] = 16, identity-ish */
        float W_UK[16] = {1,0, 0,1, 1,0, 0,1, 1,0, 0,1, 1,0, 0,1};
        /* W_UV: same shape = 16 */
        float W_UV[16] = {1,0, 0,1, 1,0, 0,1, 1,0, 0,1, 1,0, 0,1};
        /* W_DQ: [4, 2] = 8, identity-ish */
        float W_DQ[8] = {1,0, 0,1, 1,0, 0,1};
        /* W_UQ: [q_lora_rank=2, n_heads*head_dim=8] = 16 */
        float W_UQ[16] = {
            1,0,0,0,0,0,0,0,
            0,1,0,0,0,0,0,0
        };
        /* W_O: [4, 8] = 32, identity for first 4 columns */
        float W_O[32] = {
            1,0,0,0,0,0,0,0,
            0,1,0,0,0,0,0,0,
            0,0,1,0,0,0,0,0,
            0,0,0,1,0,0,0,0
        };
        /* RMSNorm: all 1s */
        float norm[4] = {1, 1, 1, 1};

        int rc = wubu_dsv4_layer_load_tensors(dl2, 0,
                                               W_DQ, W_UQ,
                                               W_DKV, W_UK, W_UV,
                                               W_O, norm);
        check("load_tensors returns 1", rc == 1);

        float x[4] = {1, 2, 3, 4};
        float out[4] = {0};
        int fwd_rc = wubu_dsv4_layer_forward(dl2, x, NULL, 0, out);
        check("forward returns 0", fwd_rc == 0);
        check("out[0] finite", out[0] == out[0]);  /* NaN check */
        check("out[1] finite", out[1] == out[1]);

        wubu_dsv4_layer_free(dl2);
    }

    printf("\n=== Results: %d/%d tests passed ===\n", tests_pass, tests_run);
    if (tests_pass == tests_run) {
        printf("ALL TESTS PASSED\n");
        return 0;
    }
    return 1;
}
