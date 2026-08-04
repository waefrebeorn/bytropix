/*
 * test_orbits.c -- NESTED-SPHERE MEMORY test (research/060, AN12 wave 3).
 *
 * The DA oracles:
 *   1. write/read round-trip: an item written at its address reads
 *      back exactly
 *   2. the fractal: addresses are self-similar — items with the same
 *      leading pair share the outer level, differ deeper in
 *   3. nesting: nest() grows the depth; a deeper address distinguishes
 *      items the outer levels could not
 *   4. ring-bounded: writing past the per-level capacity recycles the
 *      oldest (the hive freelist recycles; the trace cannot bloat)
 *   5. spheres in orbits: the angle distinguishes same-radius items
 *      (the orbit position = specialization)
 *   6. spheres inside spheres: the radius distinguishes same-angle
 *      items (the depth = hierarchy)
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "wubu_orbits.h"
#include "wubu_hive.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } else { printf("  ok: %s\n", m); } } while (0)

int main(void)
{
    printf("=== test_orbits (nested-sphere memory, AN12 wave 3) ===\n");

    wubu_hive_t hive;
    wubu_hive_init(&hive);
    wubu_orbits_cfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.max_depth = 6;
    cfg.R_max = 1.0;
    cfg.cap_per_level = 8;
    wubu_orbits_t *o = wubu_orbits_init(&hive, &cfg);
    CHECK(o != NULL, "init");

    /* --- oracle 1: write/read round-trip --- */
    {
        printf("[oracle 1] write/read round-trip\n");
        float x[8] = {0.5f, 0.3f, -0.2f, 0.9f, 0.1f, -0.7f, 0.4f, 0.6f};
        int marker = 42;
        CHECK(wubu_orbits_write(o, x, 8, &marker) == 0, "write item");
        wubu_orbit_addr_t *a = wubu_orbits_addr(o, x, 8);
        CHECK(a != NULL, "address computed");
        void *got = NULL;
        if (a) got = wubu_orbits_read(o, a);
        CHECK(got == &marker, "read returns the same item pointer");
        printf("  (depth=%d addr levels=%d r0=%.3f t0=%.3f)\n",
               wubu_orbits_depth(o), a ? a->n_levels : -1,
               a ? a->r[0] : -1, a ? a->theta[0] : -1);
        wubu_orbits_addr_free(a);
    }

    /* --- oracle 2: the fractal — same outer, different inner ---
     * needs depth >= 2 (an inner sphere must exist to differ in) */
    {
        printf("[oracle 2] fractal self-similarity\n");
        CHECK(wubu_orbits_nest(o) == 0, "nest to depth 2 for the fractal check");
        float xa[8] = {0.5f, 0.3f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
        float xb[8] = {0.5f, 0.3f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f};
        wubu_orbit_addr_t *aa = wubu_orbits_addr(o, xa, 8);
        wubu_orbit_addr_t *ab = wubu_orbits_addr(o, xb, 8);
        CHECK(aa && ab, "both addresses computed");
        if (aa && ab) {
            CHECK(fabs(aa->r[0] - ab->r[0]) < 1e-9 && fabs(aa->theta[0] - ab->theta[0]) < 1e-9,
                  "same leading pair -> same OUTER sphere");
            int differs = 0;
            for (int l = 1; l < aa->n_levels && l < 4; l++)
                if (fabs(aa->r[l] - ab->r[l]) > 1e-6) differs = 1;
            CHECK(differs, "different residuals -> different INNER spheres");
        }
        wubu_orbits_addr_free(aa);
        wubu_orbits_addr_free(ab);
    }

    /* --- oracle 5: spheres in orbits (angle = specialization) --- */
    {
        printf("[oracle 5] same radius, different angle (orbit position)\n");
        float same_r_a[6] = {0.4f, 0.4f, 0.0f, 0.0f, 0.0f, 0.0f};   /* r=0.566 */
        float same_r_b[6] = {-0.4f, 0.4f, 0.0f, 0.0f, 0.0f, 0.0f};  /* r=0.566, θ different */
        wubu_orbit_addr_t *a1 = wubu_orbits_addr(o, same_r_a, 6);
        wubu_orbit_addr_t *a2 = wubu_orbits_addr(o, same_r_b, 6);
        CHECK(a1 && a2, "addresses computed");
        if (a1 && a2) {
            CHECK(fabs(a1->r[0] - a2->r[0]) < 1e-9, "same radius (same depth)");
            CHECK(fabs(a1->theta[0] - a2->theta[0]) > 1e-3,
                  "different angle (different orbit = specialization)");
        }
        wubu_orbits_addr_free(a1);
        wubu_orbits_addr_free(a2);
    }

    /* --- oracle 6: spheres inside spheres (radius = depth) --- */
    {
        printf("[oracle 6] same angle, different radius (nesting depth)\n");
        float same_t_small[6] = {0.2f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f};   /* r=0.283 */
        float same_t_big[6]   = {0.9f, 0.9f, 0.0f, 0.0f, 0.0f, 0.0f};   /* r=1.27→clamped */
        wubu_orbit_addr_t *a1 = wubu_orbits_addr(o, same_t_small, 6);
        wubu_orbit_addr_t *a2 = wubu_orbits_addr(o, same_t_big, 6);
        CHECK(a1 && a2, "addresses computed");
        if (a1 && a2) {
            CHECK(fabs(a1->theta[0] - a2->theta[0]) < 1e-9, "same angle (same orbit)");
            CHECK(fabs(a1->r[0] - a2->r[0]) > 1e-3,
                  "different radius (different nesting depth = hierarchy)");
        }
        wubu_orbits_addr_free(a1);
        wubu_orbits_addr_free(a2);
    }

    /* --- oracle 3: nest() grows the depth --- */
    {
        printf("[oracle 3] nesting grows the recursion\n");
        CHECK(wubu_orbits_depth(o) == 2, "depth 2 after oracle-2's nest");
        CHECK(wubu_orbits_nest(o) == 0, "nest to 3");
        CHECK(wubu_orbits_depth(o) == 3, "depth 3 after the third nest");
        /* deeper addresses now have more levels */
        float x[10] = {0.3f, 0.2f, 0.1f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.2f, 0.1f};
        wubu_orbit_addr_t *a = wubu_orbits_addr(o, x, 10);
        CHECK(a && a->n_levels == 3, "address has 3 levels at depth 3");
        wubu_orbits_addr_free(a);
    }

    /* --- oracle 4: ring-bounded (the hive cannot bloat) --- */
    {
        printf("[oracle 4] ring-bounded per level\n");
        /* write cap_per_level+10 items; the live count stays at cap */
        for (int i = 0; i < (int)cfg.cap_per_level + 10; i++) {
            float x[6];
            x[0] = (float)(0.1 + 0.01 * i);
            x[1] = (float)(0.2 + 0.01 * i);
            x[2] = x[3] = x[4] = x[5] = 0.0f;
            wubu_orbits_write(o, x, 6, NULL);
        }
        /* the hive holds at most cap_per_level + the earlier writes;
         * the freelist recycles: live count <= cap + a few early items */
        size_t live = wubu_hive_live(&hive);
        printf("  (hive live=%zu, cap_per_level=%zu)\n", live, cfg.cap_per_level);
        CHECK(live <= cfg.cap_per_level + 3,
              "live count bounded (the ring recycles the oldest)");
    }

    wubu_orbits_free(o);
    wubu_hive_clear(&hive);

    printf("\n%s (%d failures)\n",
           failures == 0 ? "=== test_orbits PASSED ===" : "=== test_orbits FAILED ===",
           failures);
    return failures == 0 ? 0 : 1;
}
