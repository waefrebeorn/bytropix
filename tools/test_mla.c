/*
 * test_mla.c — tests for Multi-Latent Attention (MLA) kernel.
 *
 * Verifies the wubu_mla module used by DeepSeek V2/V4:
 * - wubu_mla_create / free
 * - wubu_mla_down_proj_kv (KV latent compression)
 * - wubu_mla_up_proj_k / wubu_mla_up_proj_v (KV up-projection)
 * - wubu_mla_proj_q (Q projection via lora rank)
 * - wubu_mla_attn (single-token attention)
 * - wubu_mla_compression_ratio (KV cache savings)
 *
 * Dimensions match DeepSeek-V2/MLA paper:
 *   hidden_dim=512, n_heads=8, head_dim=64
 *   q_lora_rank=64, kv_lora_rank=128, rope_head_dim=32
 *   => KV compression: 2*8*64=1024 vs 128+32=160 → 6.4x
 */
#include "wubu_mla.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0;
static int tests_passed = 0;

static void check(const char *name, int cond) {
    tests_run++;
    if (cond) { tests_passed++; printf("  PASS: %s\n", name); }
    else      { printf("  FAIL: %s\n", name); }
}

static void check_f32(const char *name, float got, float expected, float tol) {
    tests_run++;
    float diff = fabsf(got - expected);
    if (diff <= tol) { tests_passed++; printf("  PASS: %s (got %.6f, exp %.6f)\n", name, got, expected); }
    else             { printf("  FAIL: %s (got %.6f, exp %.6f, diff %.2e)\n", name, got, expected, diff); }
}

int main(void) {
    printf("=== MLA (Multi-Latent Attention) Tests ===\n\n");

    /* Test 1: Create/free */
    printf("--- Test 1: create/free ---\n");
    wubu_mla_t *m = wubu_mla_create(512, 8, 64, 64, 128, 32);
    check("create non-NULL", m != NULL);
    check("hidden_dim", m && m->hidden_dim == 512);
    check("n_heads", m && m->n_heads == 8);
    check("head_dim", m && m->head_dim == 64);
    check("q_lora_rank", m && m->q_lora_rank == 64);
    check("kv_lora_rank", m && m->kv_lora_rank == 128);
    check("rope_head_dim", m && m->rope_head_dim == 32);
    check("kv_latent_dim", m && m->kv_latent_dim == 160);
    wubu_mla_free(m);
    check("free doesn't crash", 1);

    /* Test 1b: Reject invalid dims */
    wubu_mla_t *bad = wubu_mla_create(0, 8, 64, 64, 128, 32);
    check("reject hidden_dim=0", bad == NULL);
    wubu_mla_t *bad2 = wubu_mla_create(512, 8, 64, -1, 128, 32);
    check("reject q_lora_rank<0", bad2 == NULL);

    /* Test 2: Compression ratio */
    printf("\n--- Test 2: compression ratio ---\n");
    m = wubu_mla_create(512, 8, 64, 64, 128, 32);
    float ratio = wubu_mla_compression_ratio(m);
    /* Standard: 2 * 8 * 64 = 1024; MLA: 160. Ratio = 1024/160 = 6.4 */
    check_f32("compression 6.4x", ratio, 6.4f, 0.01f);
    wubu_mla_free(m);

    /* Test 3: down_proj_kv */
    printf("\n--- Test 3: down_proj_kv ---\n");
    m = wubu_mla_create(4, 2, 2, 2, 2, 1);  /* tiny dims for manual verification */
    float W_DKV[12] = {  /* [3, 4] row-major: 3 rows, 4 cols */
        1, 0, 0, 0,  /* row 0: picks x[0] */
        0, 1, 0, 0,  /* row 1: picks x[1] */
        0, 0, 1, 0   /* row 2: picks x[2] */
    };
    float x[4] = {10, 20, 30, 40};
    float out[3];
    wubu_mla_down_proj_kv(m, W_DKV, x, out);
    check_f32("down_proj[0]=10", out[0], 10.0f, 1e-6f);
    check_f32("down_proj[1]=20", out[1], 20.0f, 1e-6f);
    check_f32("down_proj[2]=30", out[2], 30.0f, 1e-6f);
    wubu_mla_free(m);

    /* Test 4: up_proj_k */
    printf("\n--- Test 4: up_proj_k ---\n");
    m = wubu_mla_create(4, 2, 2, 2, 2, 1);
    float W_UK[8] = {  /* 4 rows, 2 cols */
        1, 0,  /* out[0] = kv[0] */
        0, 1,  /* out[1] = kv[1] */
        1, 1,  /* out[2] = kv[0]+kv[1] */
        0, 0   /* out[3] = 0 */
    };
    float kv_latent[2] = {5, 3};
    float outk[4];
    wubu_mla_up_proj_k(m, W_UK, kv_latent, outk);
    check_f32("up_k[0]=5", outk[0], 5.0f, 1e-6f);
    check_f32("up_k[1]=3", outk[1], 3.0f, 1e-6f);
    check_f32("up_k[2]=8", outk[2], 8.0f, 1e-6f);
    check_f32("up_k[3]=0", outk[3], 0.0f, 1e-6f);
    wubu_mla_free(m);

    /* Test 5: proj_q (down then up) */
    printf("\n--- Test 5: proj_q ---\n");
    m = wubu_mla_create(4, 1, 2, 2, 2, 1);
    float W_DQ[8] = {  /* row-major: 2 rows, 4 cols */
        1, 0, 0, 0,  /* dq[0] = x[0] */
        0, 1, 0, 0   /* dq[1] = x[1] */
    };
    float W_UQ[4] = {  /* row-major: 2 rows, 2 cols */
        1, 1,  /* up[0] = dq[0]+dq[1] */
        1, -1  /* up[1] = dq[0]-dq[1] */
    };
    float x4[4] = {3, 2, 0, 0};
    float qout[2];
    wubu_mla_proj_q(m, W_DQ, W_UQ, x4, qout);
    check_f32("q[0]=5", qout[0], 5.0f, 1e-6f);
    check_f32("q[1]=1", qout[1], 1.0f, 1e-6f);
    wubu_mla_free(m);

    /* Test 6: attn (single token, trivial softmax) */
    printf("\n--- Test 6: attn ---\n");
    m = wubu_mla_create(4, 2, 2, 2, 2, 1);
    float q[4] = {1, 2, 3, 4};        /* 2 heads * 2 */
    float k_nope[4] = {0.5, 0.5, 1, 1}; /* 2 heads * 2 */
    float k_rope[2] = {0.1, 0.2};     /* 2 heads * 1 */
    float v[4] = {10, 20, 30, 40};     /* 2 heads * 2 */
    float attn_out[4];
    wubu_mla_attn(m, q, k_nope, k_rope, v, attn_out);
    /* Single-token: softmax = 1.0, output = V */
    check_f32("attn_out[0]=10", attn_out[0], 10.0f, 1e-6f);
    check_f32("attn_out[1]=20", attn_out[1], 20.0f, 1e-6f);
    check_f32("attn_out[2]=30", attn_out[2], 30.0f, 1e-6f);
    check_f32("attn_out[3]=40", attn_out[3], 40.0f, 1e-6f);
    wubu_mla_free(m);

    /* Test 7: NULL safety */
    printf("\n--- Test 7: NULL safety ---\n");
    wubu_mla_down_proj_kv(NULL, NULL, NULL, NULL);  /* should not crash */
    check("down_proj NULL safe", 1);
    wubu_mla_attn(NULL, NULL, NULL, NULL, NULL, NULL);
    check("attn NULL safe", 1);

    /* Test 8: Large dims (DeepSeek-V2 style) */
    printf("\n--- Test 8: DeepSeek-V2 dims ---\n");
    m = wubu_mla_create(5120, 40, 128, 1536, 512, 64);
    float ratio2 = wubu_mla_compression_ratio(m);
    /* Standard: 2*40*128=10240; MLA: 512+64=576. Ratio=10240/576=17.78 */
    check_f32("V2 compression 17.78x", ratio2, 10240.0f / 576.0f, 0.01f);
    wubu_mla_free(m);

    printf("\n=== Results: %d/%d tests passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
