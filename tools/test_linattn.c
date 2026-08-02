/* test_linattn.c -- Theme IU batch 1: the linear-attention frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_linattn.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_linattn (IU batch 1) ===\n");

    /* IU01: chunked recurrence == the per-step accumulation */
    {
        float k[6][2] = { {1,0},{0,1},{1,1},{0,1},{1,0},{1,1} };
        float v[6] = { 1, 2, 3, 4, 5, 6 };
        float dec[6] = { 1,1,1,1,1,1 };
        float state[2] = { 0, 0 };
        CHECK(wubu_la_chunk(&k[0][0], v, dec, 6, 2, state, 2) == 3,
              "three chunks");
        NEAR(state[0], 1 + 0 + 3 + 0 + 5 + 6, 1e-4f);
        NEAR(state[1], 0 + 2 + 3 + 4 + 0 + 6, 1e-4f);
    }

    /* IU02: selective state update */
    {
        float x[2] = { 1, 0 }, B[2] = { 1, 0.5f }, C[2] = { 1, 0 }, A[2] = { 0.9f, 0.9f };
        float st[2] = { 1, 1 }, out = 0;
        wubu_la_selective(x, B, C, A, st, &out, 2);
        NEAR(st[0], 0.9f * 1 + 1, 1e-5f);
        NEAR(out, st[0], 1e-5f);
    }

    /* IU03: delta update reduces the error */
    {
        float B[2] = { 1, 0 }, C[2] = { 1, 0 }, tgt[2] = { 5, 0 };
        float st[2] = { 0, 0 };
        wubu_la_delta(B, tgt, C, 0.5f, st, 2);
        /* err = C'st = 0 -> st += g B (tgt-0) = 0.5*1*5 = 2.5 */
        NEAR(st[0], 2.5f, 1e-5f);
    }

    /* IU05: HGRN gated recurrence */
    {
        float x[2] = { 1, 2 }, g1[2] = { 0.5f, 0.5f }, g2[2] = { 1, 1 };
        float st[2] = { 0, 0 }, out[2];
        wubu_la_hgrn(x, g1, g2, st, 2, out);
        NEAR(out[0], 1.0f, 1e-5f);
        NEAR(out[1], 2.0f, 1e-5f);
    }

    /* IU08: tiling == the full accumulation */
    {
        float k[4][1] = { {1},{2},{3},{4} };
        float v[4] = { 1, 1, 1, 1 };
        float st = 0;
        CHECK(wubu_la_tile(&k[0][0], v, 4, 1, 2, &st) == 2, "two tiles");
        NEAR(st, 1 + 2 + 3 + 4, 1e-4f);
    }

    /* IU09: lightning update */
    NEAR(wubu_la_lightning(1.0f, 2.0f, 3.0f, 0.5f), 0.5f + 6.0f, 1e-5f);

    /* IU10: Householder preserves the norm (PaTH-style accumulation) */
    {
        float vec[3] = { 3, 4, 0 };
        float before = sqrtf(25.0f);
        wubu_la_householder(vec, 3, 1);
        float after = sqrtf(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
        NEAR(after, before, 1e-3f);
    }

    /* IU11: hybrid heads return the count */
    {
        float x[1] = { 0 }, out = 0;
        CHECK(wubu_la_hybrid_heads(x, 4, 1, 2, 2, &out) == 4, "head count");
    }

    /* IU13: KV-free recurrence */
    {
        float x[2] = { 1, 0 }, A[2] = { 0.5f, 0.5f };
        float st[2] = { 0, 0 }, out[2];
        wubu_la_kv_free(x, A, st, 2, out);
        NEAR(out[0], 1.0f, 1e-5f);
        NEAR(out[1], 0.0f, 1e-5f);
    }

    /* IU16: stability clamp */
    NEAR(wubu_la_stable(5.0f, 3.0f), 3.0f, 1e-6f);
    NEAR(wubu_la_stable(-5.0f, 3.0f), -3.0f, 1e-6f);
    NEAR(wubu_la_stable(1.0f, 3.0f), 1.0f, 1e-6f);

    /* null guards */
    CHECK(wubu_la_chunk(NULL, NULL, NULL, 4, 2, NULL, 2) == -1, "null chunk");
    CHECK(wubu_la_tile(NULL, NULL, 4, 2, 2, NULL) == -1, "null tile");

    if (failures == 0) printf("ALL LINATTN TESTS PASSED\n");
    else printf("%d LINATTN FAILURES\n", failures);
    return failures ? 1 : 0;
}
