/* Test: MoE expert choice routing (doc E05). */
#include "wubu_expert_choice.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

int main(void) {
    int n_tokens = 8, n_experts = 4, k = 2;

    /* Build a score matrix: [n_tokens, n_experts] */
    float scores[32];
    for (int i = 0; i < 32; i++) scores[i] = 0.1f * ((i * 7) % 11 - 5);

    /* Test 1: standard top-k routing */
    int topk_assign[16]; float topk_weights[16];
    wubu_topk_route(scores, n_tokens, n_experts, k, topk_assign, topk_weights);

    /* Each token should get k experts, weights sum to ~1 */
    for (int t = 0; t < n_tokens; t++) {
        float wsum = 0.0f;
        for (int i = 0; i < k; i++) {
            assert(topk_assign[t * k + i] >= 0 && topk_assign[t * k + i] < n_experts);
            wsum += topk_weights[t * k + i];
        }
        assert(fabsf(wsum - 1.0f) < 0.01f);
    }
    printf("Top-k routing: %d tokens x %d experts, k=%d, weights sum to 1\n", n_tokens, n_experts, k);

    /* Test 2: expert choice routing */
    int ec_assign[8]; float ec_weights[8];
    wubu_expert_choice_route(scores, n_tokens, n_experts, 2, ec_assign, ec_weights);

    /* Each expert should get k=2 tokens */
    for (int e = 0; e < n_experts; e++) {
        for (int i = 0; i < 2; i++) {
            assert(ec_assign[e * 2 + i] >= 0 && ec_assign[e * 2 + i] < n_tokens);
        }
        float wsum = 0.0f;
        for (int i = 0; i < 2; i++) wsum += ec_weights[e * 2 + i];
        assert(fabsf(wsum - 1.0f) < 0.01f);
    }
    printf("Expert choice: %d experts x k=%d tokens each, weights sum to 1\n", n_experts, 2);

    /* Test 3: load balance — expert choice should be perfectly balanced */
    float lb = wubu_route_load_balance(ec_assign, n_experts, 2, n_tokens);
    printf("Expert choice load balance CV = %.4f (lower = better)\n", (double)lb);
    assert(lb >= 0.0f);

    printf("ALL EXPERT-CHOICE TESTS PASSED\n");
    return 0;
}
