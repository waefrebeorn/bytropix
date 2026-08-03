/*
 * test_wubu.c -- BarunLM-35M: the mustard seed test.
 * Loads the REAL released checkpoint, verifies the parameter count
 * (35,072,768), runs a forward pass, and generates text.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "wubu.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

/* the release tokenizer is byte-level BPE; for the smoke test we use
 * the first 8 tokens (BOS + 7 arbitrary) to exercise the pipeline. */
static const uint16_t k_prompt[] = { 0, 1, 2, 3, 4, 5, 6, 7 };

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "models/wubu/model.safetensors";
    printf("=== test_wubu (BarunLM-35M, the mustard seed) ===\n");

    wubu_model_t m;
    if (wubu_load(&m, path) != 0) {
        printf("  FAIL: cannot load %s\n", path);
        return 1;
    }
    printf("  loaded %s\n", path);

    /* the released parameter count: 35,072,768 */
    long params = wubu_parameter_count(&m);
    printf("  parameters: %ld (release: %d)\n", params, BARUN_PARAMS);
    CHECK(params == BARUN_PARAMS, "parameter count == 35,072,768");

    /* the buffer + forward */
    wubu_buf_t b;
    CHECK(wubu_buf_alloc(&b, 64) == 0, "buf alloc");
    int rc = wubu_forward(&m, &b, k_prompt, 8);
    CHECK(rc == 0, "forward pass");
    if (rc == 0) {
        /* the logits must be finite */
        const float *lg = b.logits;
        int finite = 1;
        float maxv = -1e30f;
        for (int i = 0; i < 8 * BARUN_VOCAB; i++) {
            if (lg[i] != lg[i]) { finite = 0; break; }
            if (lg[i] > maxv) maxv = lg[i];
        }
        CHECK(finite, "logits finite");
        printf("  forward ok, max logit %.3f\n", maxv);
    }

    /* generation: greedy, 12 tokens */
    uint16_t gen[64];
    memcpy(gen, k_prompt, sizeof(k_prompt));
    size_t made = wubu_generate(&m, &b, gen, 8, 12, 0.0f, 42);
    CHECK(made == 12, "generated 12 tokens");
    printf("  generated %zu tokens: ", made);
    for (size_t i = 0; i < made; i++) printf("%u ", gen[8 + i]);
    printf("\n");

    /* the generated tokens must be valid vocab ids */
    int valid = 1;
    for (size_t i = 0; i < made; i++)
        if (gen[8 + i] >= BARUN_VOCAB) valid = 0;
    CHECK(valid, "tokens in vocab range");

    wubu_free(&m, &b);

    if (failures == 0) printf("ALL BARUN TESTS PASSED -- the seed is alive\n");
    else printf("%d BARUN FAILURES\n", failures);
    return failures ? 1 : 0;
}
