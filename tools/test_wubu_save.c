/*
 * test_wubu_save.c -- the checkpoint round-trip test.
 * Saves the loaded model as real safetensors, reloads it, and checks
 * byte-identical weights. The DA pass found we had a reader but no
 * writer -- this closes the gap and pins it.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "wubu.h"
#include "wubu_save.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "models/wubu/model.safetensors";
    printf("=== test_wubu_save (checkpoint export round-trip) ===\n");
    wubu_model_t m, m2;
    CHECK(wubu_load(&m, path) == 0, "load");
    CHECK(wubu_save_safetensors(&m, "/tmp/wubu_roundtrip.safetensors") == 0,
          "save as safetensors");
    CHECK(wubu_load(&m2, "/tmp/wubu_roundtrip.safetensors") == 0, "reload");

    long n = wubu_parameter_count(&m);
    long bad = 0;
    const float *a = m.embedding;
    const float *b2 = m2.embedding;
    for (long i = 0; i < 16384L * 448; i++)
        if (a[i] != b2[i]) bad++;
    CHECK(bad == 0, "embedding byte-identical");
    printf("  embedding: %ld elems, %ld diffs\n", 16384L * 448, bad);

    for (int i = 0; i < WUBU_LAYERS; i++) {
        const wubu_block_t *x = &m.blocks[i];
        const wubu_block_t *y = &m2.blocks[i];
        if (memcmp(x->q_proj, y->q_proj, 448 * 448 * sizeof(float)) != 0) { bad++; break; }
        if (memcmp(x->gate_up, y->gate_up, 448 * 2456 * sizeof(float)) != 0) { bad++; break; }
        if (memcmp(x->down, y->down, 1228 * 448 * sizeof(float)) != 0) { bad++; break; }
    }
    CHECK(bad == 0, "layer weights byte-identical");
    CHECK(wubu_parameter_count(&m2) == n, "param count preserved");

    wubu_free(&m, NULL);
    wubu_free(&m2, NULL);
    if (failures == 0) printf("ALL WUBU_SAVE TESTS PASSED\n");
    else printf("%d WUBU_SAVE FAILURES\n", failures);
    return failures ? 1 : 0;
}
