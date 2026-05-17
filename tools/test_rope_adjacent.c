/**
 * test_rope_adjacent.c — Verify adjacent-pair RoPE against split-half
 * Two binary outputs: one with correct adjacent pairs, one with NEOX split-half
 * Compare against llama.cpp by dumping Q before/after RoPE
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define GQA_HEAD_DIM 256
#define ROPE_THETA 10000000.0f
#define ROTARY_DIM 64
#define N_HEADS 16
#define N_POS 512
#define MAX_CACHE_T 1024

// Adjacent-pair RoPE (CORRECT for Qwen3 MRoPE)
static void rope_adjacent(float *buf, int n_heads, int position, const float *sc) {
    const float *sc_p = sc + (size_t)position * ROTARY_DIM;
    for (int h = 0; h < n_heads; h++) {
        float *head = buf + (size_t)h * GQA_HEAD_DIM;
        for (int i = 0; i < ROTARY_DIM / 2; i++) {
            float cosv = sc_p[i * 2];
            float sinv = sc_p[i * 2 + 1];
            float d0 = head[i * 2];
            float d1 = head[i * 2 + 1];
            head[i * 2]     = d0 * cosv - d1 * sinv;
            head[i * 2 + 1] = d0 * sinv + d1 * cosv;
        }
    }
}

// Split-half RoPE (WRONG for Qwen3 MRoPE — OLD infer_text.c)
static void rope_splithalf(float *buf, int n_heads, int position, const float *sc) {
    const float *sc_p = sc + (size_t)position * ROTARY_DIM;
    int half = ROTARY_DIM / 2;
    for (int h = 0; h < n_heads; h++) {
        float *head = buf + (size_t)h * GQA_HEAD_DIM;
        for (int i = 0; i < half; i++) {
            float cosv = sc_p[i * 2];
            float sinv = sc_p[i * 2 + 1];
            float d0 = head[i];
            float d1 = head[i + half];
            head[i]        = d0 * cosv - d1 * sinv;
            head[i + half] = d0 * sinv + d1 * cosv;
        }
    }
}

int main() {
    // Build sin/cos table (same for both variants)
    float *rope_sc = (float *)malloc((size_t)MAX_CACHE_T * ROTARY_DIM * sizeof(float));
    float theta_scale = powf(ROPE_THETA, -2.0f / ROTARY_DIM);
    for (int p = 0; p < MAX_CACHE_T; p++) {
        float theta = (float)p;
        for (int i = 0; i < ROTARY_DIM / 2; i++) {
            rope_sc[p * ROTARY_DIM + i * 2]     = cosf(theta);
            rope_sc[p * ROTARY_DIM + i * 2 + 1] = sinf(theta);
            theta *= theta_scale;
        }
    }

    // Pre-compute reference sin/cos using direct formula (for comparison)
    float *rope_sc_ref = (float *)malloc((size_t)MAX_CACHE_T * ROTARY_DIM * sizeof(float));
    for (int p = 0; p < MAX_CACHE_T; p++) {
        for (int i = 0; i < ROTARY_DIM / 2; i++) {
            float theta = powf(ROPE_THETA, -2.0f * i / ROTARY_DIM);
            float angle = (float)p * theta;
            rope_sc_ref[p * ROTARY_DIM + i * 2]     = cosf(angle);
            rope_sc_ref[p * ROTARY_DIM + i * 2 + 1] = sinf(angle);
        }
    }

    // Compare recurrence vs direct formula for table
    int table_ok = 1;
    for (int p = 0; p < 10; p++) {
        for (int i = 0; i < ROTARY_DIM; i++) {
            float diff = fabsf(rope_sc[p * ROTARY_DIM + i] - rope_sc_ref[p * ROTARY_DIM + i]);
            if (diff > 1e-6f) {
                printf("TABLE MISMATCH at p=%d i=%d: rec=%.10f ref=%.10f diff=%e\n",
                       p, i, rope_sc[p * ROTARY_DIM + i], rope_sc_ref[p * ROTARY_DIM + i], diff);
                table_ok = 0;
            }
        }
    }
    printf("Sin/cos table (recurrence vs direct): %s\n", table_ok ? "MATCH (1.0)" : "MISMATCH");

    // Test: apply both RoPE variants to random data and compare
    // Create deterministic test data
    float input[N_HEADS * GQA_HEAD_DIM];
    srand(42);
    for (int i = 0; i < N_HEADS * GQA_HEAD_DIM; i++)
        input[i] = (float)(rand() % 1000) / 100.0f - 5.0f;

    float buf_adj[N_HEADS * GQA_HEAD_DIM];
    float buf_split[N_HEADS * GQA_HEAD_DIM];
    memcpy(buf_adj, input, sizeof(input));
    memcpy(buf_split, input, sizeof(input));

    rope_adjacent(buf_adj, N_HEADS, 7, rope_sc);
    rope_splithalf(buf_split, N_HEADS, 7, rope_sc);

    // Compare adjacent vs split-half outputs
    int diff_count = 0;
    double cos_sim_num = 0, cos_sim_a2 = 0, cos_sim_b2 = 0;
    for (int i = 0; i < N_HEADS * GQA_HEAD_DIM; i++) {
        if (fabsf(buf_adj[i] - buf_split[i]) > 1e-6f) diff_count++;
        cos_sim_num += (double)buf_adj[i] * (double)buf_split[i];
        cos_sim_a2  += (double)buf_adj[i] * (double)buf_adj[i];
        cos_sim_b2  += (double)buf_split[i] * (double)buf_split[i];
    }
    double cs = cos_sim_num / (sqrt(cos_sim_a2) * sqrt(cos_sim_b2) + 1e-30);
    printf("\nAdjacent vs SplitHalf at position 7:\n");
    printf("  Different dims: %d / %d (%.1f%%)\n", diff_count, N_HEADS * GQA_HEAD_DIM,
           100.0 * diff_count / (N_HEADS * GQA_HEAD_DIM));
    printf("  Cos-sim: %.6f\n", cs);

    // Dump first head, first 64 dims at positions 0-4 for visual inspection
    printf("\nFirst head first 64 dims at various positions (0/1):\n");
    for (int p = 0; p < 2; p++) {
        float test[N_HEADS * GQA_HEAD_DIM];
        memcpy(test, input, sizeof(input));
        rope_adjacent(test, N_HEADS, p, rope_sc);
        printf("  pos=%d dims[0..7]: ", p);
        for (int d = 0; d < 8; d++) printf("%+.4f ", test[d]);
        printf("\n");
    }

    // Test all 0 position (should be identity for adjacent)
    float zero_test[N_HEADS * GQA_HEAD_DIM];
    memcpy(zero_test, input, sizeof(input));
    rope_adjacent(zero_test, N_HEADS, 0, rope_sc);
    double zero_diff = 0;
    for (int i = 0; i < N_HEADS * GQA_HEAD_DIM; i++)
        zero_diff += fabsf(zero_test[i] - input[i]);
    printf("\n  pos=0 diff (should be ~0): %.10f\n", zero_diff);

    free(rope_sc);
    free(rope_sc_ref);
    return table_ok ? 0 : 1;
}
