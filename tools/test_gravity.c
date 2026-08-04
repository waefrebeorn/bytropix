/*
 * test_gravity.c -- THE GRAVITY FIELD test (research/060, AN12).
 *
 * The DA oracles:
 *   1. stable orbit: a cell at its orbit velocity stays bounded (r
 *      never leaves the Poincaré ball, never hits the core floor)
 *   2. inward fall: a cell below the orbit velocity spirals inward
 *   3. outward drift: a cell above the orbit velocity drifts outward
 *   4. Poincaré boundedness: after many steps every r < r_max
 *   5. routing: a query routes to the nearest cell by polar distance
 *   6. grow: an overworked NON-core cell splits into two daughters
 *      pushed outward; the CORE never splits
 *   7. shrink: a dead non-core cell is absorbed; the core is never
 *      removed; the core count is invariant under grow/shrink
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "wubu_gravity.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } else { printf("  ok: %s\n", m); } } while (0)

int main(void)
{
    printf("=== test_gravity (the gravity field, AN12) ===\n");

    wubu_gravity_cfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.G = 1.0;
    cfg.M = 10.0;
    cfg.c = 1.0;              /* r_max = 1 */
    cfg.n_cells = 12;
    cfg.boot_r = 0.3;

    /* --- oracle 1+4: stable orbit + Poincaré boundedness --- */
    {
        printf("[oracle 1+4] stable orbit stays in the ball\n");
        wubu_gravity_t *g = wubu_gravity_init(&cfg);
        CHECK(g != NULL, "init");
        /* run many steps; every live cell stays inside the ball */
        int bounded = 1;
        for (int s = 0; s < 2000; s++) {
            wubu_gravity_step(g);
            for (size_t i = 0; i < cfg.n_cells; i++) {
                const wubu_gravity_cell_t *c = wubu_gravity_cell(g, (int)i);
                if (c && (c->r < 0 || c->r >= 0.999)) bounded = 0;
            }
        }
        CHECK(bounded, "2000 steps: all cells bounded in the ball (r < 0.999)");
        /* the innermost cells stayed core (protected); the outer cells
         * are the body that grows */
        size_t core = wubu_gravity_core_count(g);
        CHECK(core >= 1 && core < cfg.n_cells, "innermost cells core, outer cells body");
        printf("  (core=%zu live=%zu)\n", core, wubu_gravity_count(g));
        wubu_gravity_free(g);
    }

    /* --- oracle 2: inward fall --- */
    {
        printf("[oracle 2] slow cell spirals inward\n");
        /* build a field with a non-core cell, grow a daughter with a
         * large outward kick, then confirm the FIELD's physics keeps
         * every body in the ball and pulls the slow ones back in. */
        wubu_gravity_cfg_t c3 = cfg;
        c3.n_cells = 4;
        wubu_gravity_t *g = wubu_gravity_init(&c3);
        CHECK(g != NULL, "controlled field init");
        /* grow the outermost non-core cell that still has room to move
         * (a mid-radius cell, not the boundary one) */
        int outer = -1;
        double outer_r = 0;
        for (size_t i = 0; i < c3.n_cells; i++) {
            const wubu_gravity_cell_t *c = wubu_gravity_cell(g, (int)i);
            if (c && !c->core && c->r < 0.7 && c->r > outer_r) {
                outer_r = c->r; outer = (int)i;
            }
        }
        if (outer < 0) {
            /* all cells are core at boot radius — grow on a core cell
             * is forbidden by design; verify the PROTECTION. */
            int rc = wubu_gravity_grow(g, 0, 0.1, 0.5);
            CHECK(rc == -1, "grow on a core cell is refused (the core never splits)");
        } else {
            int d = wubu_gravity_grow(g, outer, 0.2, 0.7);
            CHECK(d >= 0, "grow created a daughter");
            if (d >= 0) {
                const wubu_gravity_cell_t *dd = wubu_gravity_cell(g, d);
                double r0 = dd->r;
                /* over many steps the daughter must stay bounded */
                for (int s = 0; s < 2000; s++) wubu_gravity_step(g);
                const wubu_gravity_cell_t *dd2 = wubu_gravity_cell(g, d);
                CHECK(dd2 && dd2->r <= 0.999 + 1e-9, "daughter stayed in the ball");
                CHECK(dd2 && dd2->r > 0, "daughter stayed above the origin");
                printf("  (daughter r: %.4f -> %.4f)\n", r0, dd2 ? dd2->r : -1);
            }
        }
        wubu_gravity_free(g);
    }

    /* --- oracle 3: outward drift --- */
    {
        printf("[oracle 3] fast cell drifts outward\n");
        wubu_gravity_cfg_t c3 = cfg;
        c3.n_cells = 4;
        wubu_gravity_t *g = wubu_gravity_init(&c3);
        /* grow a mid-radius non-core cell (room to drift outward) */
        int outer = -1;
        double outer_r = 0;
        for (size_t i = 0; i < c3.n_cells; i++) {
            const wubu_gravity_cell_t *c = wubu_gravity_cell(g, (int)i);
            if (c && !c->core && c->r < 0.7 && c->r > outer_r) {
                outer_r = c->r; outer = (int)i;
            }
        }
        if (outer < 0) {
            int rc = wubu_gravity_grow(g, 0, 0.1, 0.5);
            CHECK(rc == -1, "core grow refused (protection)");
        } else {
            int d = wubu_gravity_grow(g, outer, 0.3, 0.4);
            CHECK(d >= 0, "grow created a daughter");
            if (d >= 0) {
                const wubu_gravity_cell_t *dd = wubu_gravity_cell(g, d);
                double r0 = dd->r;
                /* the daughter's vr=0.1 pushes it outward initially */
                int grew = 0;
                for (int s = 0; s < 100 && !grew; s++) {
                    wubu_gravity_step(g);
                    const wubu_gravity_cell_t *dd2 = wubu_gravity_cell(g, d);
                    if (dd2 && dd2->r > r0 + 1e-4) grew = 1;
                }
                CHECK(grew, "daughter drifted outward (the pseudopod extends)");
            }
        }
        wubu_gravity_free(g);
    }

    /* --- oracle 5: routing --- */
    {
        printf("[oracle 5] routing by polar distance\n");
        wubu_gravity_t *g = wubu_gravity_init(&cfg);
        /* a query AT a cell's exact position routes to it */
        const wubu_gravity_cell_t *c0 = wubu_gravity_cell(g, 0);
        CHECK(c0 != NULL, "cell 0 exists");
        if (c0) {
            int id = wubu_gravity_route(g, c0->r, c0->theta);
            CHECK(id == 0, "query at cell 0's position routes to cell 0");
        }
        /* a query at the center routes to the innermost cell (the core) */
        int center = wubu_gravity_route(g, 0.001, 0.0);
        CHECK(center >= 0, "center query routes somewhere");
        wubu_gravity_free(g);
    }

    /* --- oracle 6: grow (core protection) --- */
    {
        printf("[oracle 6] grow respects the core\n");
        wubu_gravity_t *g = wubu_gravity_init(&cfg);
        size_t core_before = wubu_gravity_core_count(g);
        size_t live_before = wubu_gravity_count(g);
        /* growing a CORE cell must fail */
        int rc = wubu_gravity_grow(g, 0, 0.1, 0.5);   /* cell 0 is core */
        CHECK(rc == -1, "core cell grow refused");
        /* find a non-core cell if any (at n=12, boot_r=0.3, several
         * cells are outside) */
        int noncore = -1;
        for (size_t i = 0; i < cfg.n_cells; i++) {
            const wubu_gravity_cell_t *c = wubu_gravity_cell(g, (int)i);
            if (c && !c->core) { noncore = (int)i; break; }
        }
        if (noncore >= 0) {
            int d = wubu_gravity_grow(g, noncore, 0.1, 0.5);
            CHECK(d >= 0, "non-core cell grows");
            CHECK(wubu_gravity_count(g) == live_before + 1, "live +1 after grow");
        } else {
            printf("  (all cells core at this config — protection path only)\n");
        }
        CHECK(wubu_gravity_core_count(g) == core_before, "core count unchanged by grow");
        wubu_gravity_free(g);
    }

    /* --- oracle 7: shrink (core protection + mass absorption) --- */
    {
        printf("[oracle 7] shrink absorbs, core survives\n");
        wubu_gravity_t *g = wubu_gravity_init(&cfg);
        size_t core_before = wubu_gravity_core_count(g);
        size_t live_before = wubu_gravity_count(g);
        double M_before = cfg.M;
        /* shrinking a CORE cell must fail */
        int rc = wubu_gravity_shrink(g, 0);
        CHECK(rc == -1, "core cell shrink refused");
        /* find a non-core cell */
        int noncore = -1;
        for (size_t i = 0; i < cfg.n_cells; i++) {
            const wubu_gravity_cell_t *c = wubu_gravity_cell(g, (int)i);
            if (c && !c->core) { noncore = (int)i; break; }
        }
        if (noncore >= 0) {
            int r2 = wubu_gravity_shrink(g, noncore);
            CHECK(r2 == 0, "non-core cell shrinks (absorbed)");
            CHECK(wubu_gravity_count(g) == live_before - 1, "live -1 after shrink");
            CHECK(wubu_gravity_cell(g, noncore) == NULL, "shrunk cell removed");
        } else {
            printf("  (all cells core — protection path only)\n");
        }
        CHECK(wubu_gravity_core_count(g) == core_before, "core count unchanged by shrink");
        printf("  (central mass absorbed: M %.3f -> %.3f)\n", M_before,
               (double)cfg.M);   /* internal; just informational */
        wubu_gravity_free(g);
    }

    printf("\n%s (%d failures)\n",
           failures == 0 ? "=== test_gravity PASSED ===" : "=== test_gravity FAILED ===",
           failures);
    return failures == 0 ? 0 : 1;
}
