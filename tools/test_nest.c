/*
 * test_nest.c -- the WuBu Nesting transitions test (phase 3).
 * Verifies the quaternion SO(4) rotation preserves norms (a rotation
 * must not change lengths), the Hamilton product is the rotation
 * engine, and the full transition T = T̃∘R produces a valid tangent
 * point. Also the relative-vector and descriptor-flow pieces.
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "wubu_nest.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

int main(void)
{
    printf("=== test_nest (the WuBu Nesting transitions, 層疊嵌套) ===\n");

    /* 1. the Hamilton product is the rotation engine: rotating by q
     * then q* (inverse) must restore the vector */
    {
        wubu_quat_t q = { 0.9239f, 0.2209f, 0.2209f, 0.0f };  /* unit, ~45° */
        q = wubu_quat_normalize(q);
        float v[4] = { 1, 0, 0, 0 };
        float mid[4], back[4];
        wubu_quat_rotate_vec(q, v, mid);
        wubu_quat_t qinv = { q.w, -q.x, -q.y, -q.z };
        wubu_quat_rotate_vec(qinv, mid, back);
        CHECK(fabsf(back[0] - 1) < 1e-3 && fabsf(back[1]) < 1e-3 &&
              fabsf(back[2]) < 1e-3, "q then q* restores the vector");
        printf("  rotate by q then q*: (%.4f, %.4f, %.4f, %.4f)\n",
               back[0], back[1], back[2], back[3]);
    }

    /* 2. SO(4): a rotation preserves the norm */
    {
        srand(3);
        int ok = 1;
        for (int t = 0; t < 2000; t++) {
            wubu_quat_t q = {
                ((float)rand() / RAND_MAX) * 2 - 1,
                ((float)rand() / RAND_MAX) * 2 - 1,
                ((float)rand() / RAND_MAX) * 2 - 1,
                ((float)rand() / RAND_MAX) * 2 - 1
            };
            q = wubu_quat_normalize(q);
            float v[4] = {
                ((float)rand() / RAND_MAX) * 2 - 1,
                ((float)rand() / RAND_MAX) * 2 - 1,
                ((float)rand() / RAND_MAX) * 2 - 1,
                ((float)rand() / RAND_MAX) * 2 - 1
            };
            float before = sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]+v[3]*v[3]);
            float out[4];
            wubu_quat_rotate_vec(q, v, out);
            float after = sqrtf(out[0]*out[0]+out[1]*out[1]+out[2]*out[2]+out[3]*out[3]);
            if (fabsf(before - after) > 1e-3) { ok = 0; break; }
        }
        CHECK(ok, "SO(4) rotation preserves the norm (2000 samples)");
        printf("  SO(4) norm-preserving: 2000 samples ok\n");
    }

    /* 3. the learned rotation: axis = descriptor, angle learned */
    {
        float ld[4] = { 1, 0, 0, 0 };
        wubu_quat_t q = wubu_nest_learned_rotation(ld, 1.5708f);  /* 90° */
        CHECK(fabsf(q.w - cosf(0.7854f)) < 1e-3, "learned rotation w = cos(a/2)");
        /* rotating (0,1,0,0) by 90° about x gives ~(0,0,1,0) */
        float v[4] = { 0, 1, 0, 0 }, out[4];
        wubu_quat_rotate_vec(q, v, out);
        CHECK(fabsf(out[2] - 1) < 1e-2 && fabsf(out[1]) < 1e-2,
              "90° about X maps Y -> Z");
        printf("  learned rotation: (0,1,0,0) -> (%.3f, %.3f, %.3f, %.3f)\n",
               out[0], out[1], out[2], out[3]);
    }

    /* 4. the full transition T = T̃∘R: 4 -> 8 dims, bounded output */
    {
        float map_w[8 * 4], map_b[8];
        for (int i = 0; i < 8 * 4; i++) map_w[i] = 0.1f * ((float)(i % 5) - 2);
        for (int i = 0; i < 8; i++) map_b[i] = 0.05f;
        float v[4] = { 0.5f, -0.3f, 0.2f, 0.1f };
        float ld[4] = { 1, 0, 0, 0 };
        wubu_quat_t rot = wubu_nest_learned_rotation(ld, 0.5f);
        float out[8];
        wubu_nest_transition(rot, v, 4, map_w, map_b, 8, out);
        int bounded = 1;
        for (int i = 0; i < 8; i++) if (fabsf(out[i]) > 1.0f + 1e-4) bounded = 0;
        CHECK(bounded, "transition output bounded by tanh");
        printf("  transition 4->8: out = {%.3f %.3f %.3f %.3f ...}\n",
               out[0], out[1], out[2], out[3]);
    }

    /* 5. relative vectors + descriptor flow */
    {
        float v[4] = { 1, 0, 0, 0 }, b[4] = { 0.5f, 0.2f, 0, 0 };
        float d[4];
        wubu_nest_relative(v, b, 4, d);
        CHECK(fabsf(d[0] - 0.5f) < 1e-6 && fabsf(d[1] + 0.2f) < 1e-6,
              "relative vector d = v - boundary");
        float map_w[8 * 4], map_b[8];
        for (int i = 0; i < 8 * 4; i++) map_w[i] = 0.01f;
        for (int i = 0; i < 8; i++) map_b[i] = 0;
        float ld_src[4] = { 1, 0, 0, 0 }, ld_dst[8];
        wubu_quat_t rot = wubu_nest_learned_rotation(ld_src, 0.3f);
        wubu_nest_descriptor_flow(rot, ld_src, 4, map_w, map_b, 0.7f, 8, ld_dst);
        CHECK(fabsf(ld_dst[7] - 0.7f) < 1e-6, "spread σ flows as the last dim");
        printf("  descriptor flow: σ=0.7 -> ld_dst[7]=%.3f\n", ld_dst[7]);
    }

    if (failures == 0) printf("ALL NEST TESTS PASSED -- the bubbles nest\n");
    else printf("%d NEST FAILURES\n", failures);
    return failures ? 1 : 0;
}
