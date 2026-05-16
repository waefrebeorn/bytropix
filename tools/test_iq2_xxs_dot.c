/**
 * test_iq2_xxs_dot.c — Verify on-the-fly IQ2_XXS dequant dot product
 * against reference dequantize_iq2_xxs_row() + standard dot product.
 *
 * Generates random quantized data, dequantizes to f32 both ways,
 * and compares dot products.
 */
#include "gguf_reader.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define QK_K 256
#define IQ2_XXS_BLOCK_SIZE 66

// Reference: full dequant then dot
static float reference_dot_block(const uint8_t *block, const float *x) {
    float buf[QK_K];
    dequantize_iq2_xxs_block(block, buf);
    float dot = 0.0f;
    for (int i = 0; i < QK_K; i++) {
        dot += buf[i] * x[i];
    }
    return dot;
}

// Test single block
static int test_single_block(void) {
    printf("=== Test: Single IQ2_XXS block dequant ===\n");

    // Generate random block data (66 bytes)
    uint8_t block[IQ2_XXS_BLOCK_SIZE];
    for (int i = 0; i < IQ2_XXS_BLOCK_SIZE; i++) {
        block[i] = (uint8_t)(rand() & 0xFF);
    }

    // Generate random input vector (256 floats)
    float x[QK_K];
    for (int i = 0; i < QK_K; i++) {
        x[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
    }

    // Fused dequant + dot
    float dot_fused = iq2_xxs_dot_block(block, x);

    // Reference: full dequant then dot
    float dot_ref = reference_dot_block(block, x);

    // Compare
    float diff = fabsf(dot_fused - dot_ref);
    float rel_diff = (fabsf(dot_ref) > 1e-10f) ? diff / fabsf(dot_ref) : diff;

    printf("  dot_fused = %.10f\n", dot_fused);
    printf("  dot_ref   = %.10f\n", dot_ref);
    printf("  abs_diff  = %.10f  rel_diff = %.10f\n", diff, rel_diff);

    if (rel_diff > 1e-5f && diff > 1e-5f) {
        printf("  FAIL: dot product mismatch!\n");
        return 0;
    }
    printf("  PASS\n");
    return 1;
}

// Test multiple blocks (as in one row/column of MoE weight)
static int test_multi_block(void) {
    printf("\n=== Test: Multi-block row/column (2048 elems = 8 blocks) ===\n");

    int n_blocks = 8;
    int n_total_bytes = n_blocks * IQ2_XXS_BLOCK_SIZE;
    int n_elems = n_blocks * QK_K;

    // Allocate quantized data
    uint8_t *qdata = (uint8_t *)malloc(n_total_bytes);
    float *x = (float *)malloc(n_elems * sizeof(float));

    // Generate random quantized data
    for (int i = 0; i < n_total_bytes; i++) {
        qdata[i] = (uint8_t)(rand() & 0xFF);
    }

    // Generate random input vector
    for (int i = 0; i < n_elems; i++) {
        x[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
    }

    // Fused dequant + dot over all blocks
    float dot_fused = iq2_xxs_dot_row(qdata, n_elems, x);

    // Reference: dequant whole row then dot
    float *buf = (float *)malloc(n_elems * sizeof(float));
    dequantize_iq2_xxs_row(qdata, buf, n_elems);
    float dot_ref = 0.0f;
    for (int i = 0; i < n_elems; i++) {
        dot_ref += buf[i] * x[i];
    }

    float diff = fabsf(dot_fused - dot_ref);
    float rel_diff = (fabsf(dot_ref) > 1e-10f) ? diff / fabsf(dot_ref) : diff;

    printf("  dot_fused = %.10f\n", dot_fused);
    printf("  dot_ref   = %.10f\n", dot_ref);
    printf("  abs_diff  = %.10f  rel_diff = %.10f\n", diff, rel_diff);

    if (rel_diff > 1e-5f && diff > 1e-5f) {
        printf("  FAIL: dot product mismatch!\n");
        free(qdata); free(x); free(buf);
        return 0;
    }
    printf("  PASS\n");
    free(qdata); free(x); free(buf);
    return 1;
}

// Test full MoE expert gate matmul: [D_MODEL] @ [D_MODEL, D_FF]
// Compare moe_expert_forward_dequant vs dequant-then-matmul
static int test_moe_expert_matmul(void) {
    printf("\n=== Test: Full MoE expert matmul (IQ2_XXS dequant) ===\n");

    // Gate/up weights: [D_MODEL, D_FF] = [2048, 512]
    const int n_rows = D_MODEL;
    const int n_cols = D_FF;
    const int blocks_per_col = n_rows / QK_K;  // 8
    const int col_bytes = blocks_per_col * IQ2_XXS_BLOCK_SIZE;  // 528
    const int total_bytes = n_cols * col_bytes;  // 270336

    // Down weight: [D_FF, D_MODEL] = [512, 2048]
    const int down_n_rows = D_FF;
    const int down_n_cols = D_MODEL;
    const int down_blocks_per_col = down_n_rows / QK_K;  // 2
    const int down_col_bytes = down_blocks_per_col * IQ2_XXS_BLOCK_SIZE;  // 132
    const int down_total_bytes = down_n_cols * down_col_bytes;  // 270336

    // Allocate quantized weights and input
    uint8_t *gate_q = (uint8_t *)malloc(total_bytes);
    uint8_t *up_q = (uint8_t *)malloc(total_bytes);
    uint8_t *down_q = (uint8_t *)malloc(down_total_bytes);
    float *x = (float *)malloc(D_MODEL * sizeof(float));
    float *temp = (float *)malloc(D_FF * 3 * sizeof(float));
    float *output_dequant = (float *)malloc(D_MODEL * sizeof(float));
    float *output_ref = (float *)malloc(D_MODEL * sizeof(float));
    float *buf = (float *)malloc(n_rows * sizeof(float));

    // Generate random quantized data with valid fp16 scale values
    for (int i = 0; i < total_bytes; i += 66) {
        // Set d to a valid fp16 value (around 0.5-2.0 range)
        // fp16(1.0) = 0x3C00, fp16(0.5) = 0x3800
        uint16_t d_val = 0x3800 + (uint16_t)(rand() % 0x0400);  // 0.5 to ~2.0
        gate_q[i] = d_val & 0xFF;
        gate_q[i+1] = (d_val >> 8) & 0xFF;
        up_q[i] = d_val & 0xFF;
        up_q[i+1] = (d_val >> 8) & 0xFF;
        // Fill remaining 64 bytes with random valid grid data
        for (int k = 2; k < 66 && i + k < total_bytes; k++) {
            gate_q[i + k] = (uint8_t)(rand() & 0xFF);
            up_q[i + k] = (uint8_t)(rand() & 0xFF);
        }
    }
    for (int i = 0; i < down_total_bytes; i += 66) {
        uint16_t d_val = 0x3800 + (uint16_t)(rand() % 0x0400);
        down_q[i] = d_val & 0xFF;
        down_q[i+1] = (d_val >> 8) & 0xFF;
        for (int k = 2; k < 66 && i + k < down_total_bytes; k++) {
            down_q[i + k] = (uint8_t)(rand() & 0xFF);
        }
    }

    // Generate random input
    for (int i = 0; i < D_MODEL; i++) {
        x[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
    }

    // === Method 1: On-the-fly dequant ===
    moe_expert_forward_dequant(x, gate_q, up_q, down_q, temp, output_dequant);

    // === Method 2: Full dequant then standard matmul ===
    // Dequantize gate weight column by column
    float *gate_f32 = (float *)malloc(n_rows * n_cols * sizeof(float));
    for (int j = 0; j < n_cols; j++) {
        const uint8_t *qcol = gate_q + j * col_bytes;
        for (int b = 0; b < blocks_per_col; b++) {
            dequantize_iq2_xxs_block(qcol + b * 66, buf + b * QK_K);
        }
        memcpy(gate_f32 + j * n_rows, buf, n_rows * sizeof(float));
    }

    // Dequantize up weight
    float *up_f32 = (float *)malloc(n_rows * n_cols * sizeof(float));
    for (int j = 0; j < n_cols; j++) {
        const uint8_t *qcol = up_q + j * col_bytes;
        for (int b = 0; b < blocks_per_col; b++) {
            dequantize_iq2_xxs_block(qcol + b * 66, buf + b * QK_K);
        }
        memcpy(up_f32 + j * n_rows, buf, n_rows * sizeof(float));
    }

    // Dequantize down weight
    float *down_f32 = (float *)malloc(down_n_rows * down_n_cols * sizeof(float));
    for (int j = 0; j < down_n_cols; j++) {
        const uint8_t *qcol = down_q + j * down_col_bytes;
        for (int b = 0; b < down_blocks_per_col; b++) {
            dequantize_iq2_xxs_block(qcol + b * 66, buf + b * QK_K);
        }
        memcpy(down_f32 + j * down_n_rows, buf, down_n_rows * sizeof(float));
    }

    // Standard matmul: gate then up then act then down
    float gate_ref[D_FF], up_ref[D_FF], act_ref[D_FF];
    for (int j = 0; j < D_FF; j++) {
        float sum = 0.0f;
        for (int k = 0; k < D_MODEL; k++)
            sum += x[k] * gate_f32[k + j * D_MODEL];
        gate_ref[j] = sum;
    }
    for (int j = 0; j < D_FF; j++) {
        float sum = 0.0f;
        for (int k = 0; k < D_MODEL; k++)
            sum += x[k] * up_f32[k + j * D_MODEL];
        up_ref[j] = sum;
    }
    for (int j = 0; j < D_FF; j++) {
        float g = gate_ref[j];
        float silu_g = (g < -80.0f) ? 0.0f : g / (1.0f + expf(-g));
        act_ref[j] = silu_g * up_ref[j];
    }
    for (int j = 0; j < D_MODEL; j++) {
        float sum = 0.0f;
        for (int k = 0; k < D_FF; k++)
            sum += act_ref[k] * down_f32[k + j * D_FF];
        output_ref[j] = sum;
    }

    // === Compare results ===
    float max_err = 0.0f;
    float avg_err = 0.0f;
    int n_bad = 0;
    float ref_norm = 0.0f;
    for (int i = 0; i < D_MODEL; i++) ref_norm += output_ref[i] * output_ref[i];
    ref_norm = sqrtf(ref_norm);
    float scale_tol = fmaxf(1.0f, ref_norm * 1e-5f);

    for (int i = 0; i < D_MODEL; i++) {
        float err = fabsf(output_dequant[i] - output_ref[i]);
        if (err > max_err) max_err = err;
        avg_err += err;
        if (err > scale_tol) n_bad++;
    }
    avg_err /= D_MODEL;

    printf("  ref_l2_norm = %.6f\n", ref_norm);
    printf("  max_abs_err = %.10f (tol=%.10f)\n", max_err, scale_tol);
    printf("  avg_abs_err = %.10f\n", avg_err);
    printf("  elements with err > tolerance: %d / %d\n", n_bad, D_MODEL);

    // Print first 8 elements of each
    printf("  output_dequant[0..7]: ");
    for (int i = 0; i < 8; i++) printf("%+.6f ", output_dequant[i]);
    printf("\n");
    printf("  output_ref[0..7]:      ");
    for (int i = 0; i < 8; i++) printf("%+.6f ", output_ref[i]);
    printf("\n");

    if (n_bad > 3 || (ref_norm > 1e-6f && max_err > ref_norm * 1e-3f)) {
        printf("  FAIL: output mismatch! (max_err=%.6f, ref_norm=%.6f, ratio=%.10f)\n",
               max_err, ref_norm, ref_norm > 0 ? max_err/ref_norm : 0);
        free(gate_q); free(up_q); free(down_q);
        free(x); free(temp); free(output_dequant); free(output_ref); free(buf);
        free(gate_f32); free(up_f32); free(down_f32);
        return 0;
    }
    printf("  PASS\n");

    free(gate_q); free(up_q); free(down_q);
    free(x); free(temp); free(output_dequant); free(output_ref); free(buf);
    free(gate_f32); free(up_f32); free(down_f32);
    return 1;
}

// Test individual block dequant correctness (dequantize_iq2_xxs_block vs dequantize_iq2_xxs_row)
static int test_block_dequant(void) {
    printf("\n=== Test: dequantize_iq2_xxs_block vs dequantize_iq2_xxs_row ===\n");

    uint8_t block[IQ2_XXS_BLOCK_SIZE];
    for (int i = 0; i < IQ2_XXS_BLOCK_SIZE; i++) {
        block[i] = (uint8_t)(rand() & 0xFF);
    }

    float out1[QK_K];
    float out2[QK_K];

    // Block dequant
    dequantize_iq2_xxs_block(block, out1);
    // Row dequant (single block)
    dequantize_iq2_xxs_row(block, out2, QK_K);

    float max_err = 0.0f;
    int n_bad = 0;
    for (int i = 0; i < QK_K; i++) {
        float err = fabsf(out1[i] - out2[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-6f) n_bad++;
    }

    printf("  max_err = %.10f, different elements: %d / %d\n", max_err, n_bad, QK_K);
    if (n_bad > 0) {
        printf("  FAIL: block dequant mismatch!\n");
        return 0;
    }
    printf("  PASS\n");
    return 1;
}

int main(void) {
    srand(42);  // deterministic seed

    int passed = 0, failed = 0;

    if (test_single_block()) passed++; else failed++;
    if (test_multi_block()) passed++; else failed++;
    if (test_block_dequant()) passed++; else failed++;
    if (test_moe_expert_matmul()) passed++; else failed++;

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed out of %d tests\n",
           passed, failed, passed + failed);
    printf("========================================\n");

    return failed > 0 ? 1 : 0;
}
