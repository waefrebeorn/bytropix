/* test_ns_world.c -- WORLDSIM #262 worldsim state at /n/world.
 * Proves: /n/world recognized in ns_tree, stat files render (steps/bodies/ke/
 * dt/restitution/gravity), per-body files render (body/{i}/{field}), unknown
 * paths fail-closed, content matches live world state. */
#include "wubu_ns_world.h"
#include "wubu_ns_tree.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    /* ns_tree recognizes /n/world */
    wubu_ns_resolved_t r;
    CHECK(wubu_ns_tree_resolve("/n/world", &r) == 0 && r.subsys == WUBU_NS_WORLD, "/n/world recognized");
    CHECK(r.is_dir == 1, "/n/world is a dir");
    CHECK(strcmp(wubu_ns_tree_subsys_name(WUBU_NS_WORLD), "world") == 0, "subsys name world");

    /* build a world, step it */
    wubu_pforces_t grav = {10.0, 0.0, 0.0};
    wubu_world_t w;
    wubu_world_init(&w, 2, grav, 0.5, 1.0/60.0, 1000);
    wubu_world_add(&w, 0, 100, 0, 0, 1, 1);
    wubu_world_add(&w, 5, 100, 3, 0, 2, 1);
    wubu_world_step(&w, 0.5);

    char buf[128];
    /* stat files */
    CHECK(wubu_ns_world_read(&w, "steps", buf, sizeof buf) > 0, "read /n/world/steps");
    CHECK(strtol(buf, NULL, 10) > 0, "steps > 0");
    CHECK(wubu_ns_world_read(&w, "bodies", buf, sizeof buf) > 0, "read /n/world/bodies");
    CHECK(strcmp(buf, "2\n") == 0, "bodies == 2");
    CHECK(wubu_ns_world_read(&w, "ke", buf, sizeof buf) > 0, "read /n/world/ke");
    CHECK(wubu_ns_world_read(&w, "dt", buf, sizeof buf) > 0, "read /n/world/dt");
    CHECK(wubu_ns_world_read(&w, "restitution", buf, sizeof buf) > 0, "read /n/world/restitution");
    CHECK(wubu_ns_world_read(&w, "gravity", buf, sizeof buf) > 0, "read /n/world/gravity");
    CHECK(strncmp(buf, "10.0", 4) == 0, "gravity == 10.0");

    /* per-body files */
    CHECK(wubu_ns_world_read(&w, "body/0/y", buf, sizeof buf) > 0, "read /n/world/body/0/y");
    double y0 = atof(buf);
    CHECK(y0 < 100.0, "body 0 fell (y < 100)");
    CHECK(wubu_ns_world_read(&w, "body/1/vx", buf, sizeof buf) > 0, "read /n/world/body/1/vx");
    CHECK(atof(buf) > 2.9, "body 1 kept vx ~3 (no drag)");
    CHECK(wubu_ns_world_read(&w, "body/1/mass", buf, sizeof buf) > 0, "read /n/world/body/1/mass");
    CHECK(atof(buf) == 2.0, "body 1 mass == 2");

    /* fail-closed: unknown paths */
    CHECK(wubu_ns_world_read(&w, "bogus", buf, sizeof buf) == -1, "unknown stat -> -1");
    CHECK(wubu_ns_world_read(&w, "body/99/x", buf, sizeof buf) == -1, "out-of-range body -> -1");
    CHECK(wubu_ns_world_read(&w, "body/0/bogus", buf, sizeof buf) == -1, "unknown field -> -1");
    CHECK(wubu_ns_world_read(&w, "body/x/y", buf, sizeof buf) == -1, "non-numeric index -> -1");
    CHECK(wubu_ns_world_read(NULL, "steps", buf, sizeof buf) == -1, "NULL world -> -1");
    CHECK(wubu_ns_world_read(&w, NULL, buf, sizeof buf) == -1, "NULL relpath -> -1");

    printf("\n=== test_ns_world: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
