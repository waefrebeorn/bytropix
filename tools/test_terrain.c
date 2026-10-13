/* test_terrain.c -- WORLDSIM #259 terrain heightfield + erosion.
 * Proves: init/flat, get/set bounds, gaussian bump, erosion moves sediment,
 * erosion is deterministic (same seed -> same result), sediment accumulates,
 * fail-closed on bad dims/NULL. */
#include "wubu_terrain.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)
#define NEAR(a,b,t) (fabs((a)-(b)) < (t))

int main(void) {
    wubu_terrain_t t;
    CHECK(wubu_terrain_init(&t, 32, 32, 10.0) == 0, "init 32x32 flat at 10");
    CHECK(NEAR(wubu_terrain_get(&t, 5, 5), 10.0, 1e-9), "flat height 10");
    CHECK(wubu_terrain_get(&t, -1, 0) == 0.0, "out-of-range get -> 0");
    CHECK(wubu_terrain_set(&t, 100, 0, 5) == -1, "out-of-range set -> -1");

    /* bad dims */
    CHECK(wubu_terrain_init(&t, 0, 32, 0) == -1, "zero width rejected");
    CHECK(wubu_terrain_init(&t, 200, 200, 0) == -1, "oversize rejected");

    /* gaussian bump raises the center */
    wubu_terrain_set(&t, 16, 16, 10.0);
    wubu_terrain_bump(&t, 16, 16, 6.0, 20.0);
    CHECK(wubu_terrain_get(&t, 16, 16) > 25.0, "bump raises center");
    CHECK(wubu_terrain_get(&t, 0, 0) == 10.0, "bump leaves far corner flat");

    /* erosion moves sediment */
    wubu_terrain_t e;
    wubu_terrain_init(&e, 48, 48, 5.0);
    wubu_terrain_bump(&e, 24, 24, 15.0, 30.0);   /* a hill to erode */
    double sed0 = wubu_terrain_sediment(&e);
    CHECK(sed0 == 0.0, "no sediment before erosion");
    wubu_terrain_erode(&e, 200, 64);
    CHECK(wubu_terrain_sediment(&e) > 0.0, "erosion moves sediment");
    /* erosion reshapes the hill: some cells change height (carves slopes) */
    int changed = 0;
    wubu_terrain_t ref;
    wubu_terrain_init(&ref, 48, 48, 5.0);
    wubu_terrain_bump(&ref, 24, 24, 15.0, 30.0);
    for (int y = 0; y < 48 && !changed; y++)
        for (int x = 0; x < 48; x++)
            if (fabs(wubu_terrain_get(&e,x,y) - wubu_terrain_get(&ref,x,y)) > 1e-6) { changed = 1; break; }
    CHECK(changed == 1, "erosion reshapes the hill (cells change height)");

    /* DETERMINISM: two terrains, same seed, same erosion -> identical */
    wubu_terrain_t d1, d2;
    wubu_terrain_init(&d1, 40, 40, 5.0);
    wubu_terrain_init(&d2, 40, 40, 5.0);
    wubu_terrain_bump(&d1, 20, 20, 12.0, 25.0);
    wubu_terrain_bump(&d2, 20, 20, 12.0, 25.0);
    wubu_terrain_erode(&d1, 100, 50);
    wubu_terrain_erode(&d2, 100, 50);
    int identical = 1;
    for (int y = 0; y < 40 && identical; y++)
        for (int x = 0; x < 40; x++)
            if (fabs(wubu_terrain_get(&d1,x,y) - wubu_terrain_get(&d2,x,y)) > 1e-12) { identical = 0; break; }
    CHECK(identical == 1, "DETERMINISTIC: same seed -> identical terrain");

    /* roughness is finite and positive */
    CHECK(wubu_terrain_roughness(&e) > 0.0, "roughness > 0");

    /* fail-closed on NULL */
    CHECK(wubu_terrain_init(NULL, 4, 4, 0) == -1, "NULL init -> -1");
    CHECK(wubu_terrain_erode(NULL, 1, 1) == -1, "NULL erode -> -1");
    CHECK(wubu_terrain_sediment(NULL) == 0.0, "NULL sediment -> 0");

    printf("\n=== test_terrain: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
