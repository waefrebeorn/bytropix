/* Test: ECS component store (doc C06). register / snapshot / restore. */
#include "wubu_ecs.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

typedef struct { int foo; float bar; } comp_a_t;
typedef struct { double big[4]; } comp_b_t;

int main(void) {
    wubu_ecs_t *e = wubu_ecs_create(8);
    assert(e);

    /* Register two typed components. */
    int id_a = wubu_ecs_add(e, "kv_cache_state", sizeof(comp_a_t), NULL);
    int id_b = wubu_ecs_add(e, "ssm_states", sizeof(comp_b_t), NULL);
    assert(id_a >= 0 && id_b >= 0);
    assert(wubu_ecs_find(e, "kv_cache_state") == id_a);
    assert(wubu_ecs_find(e, "nope") == -1);
    assert(wubu_ecs_count(e) == 2);

    /* Write distinct data into each. */
    comp_a_t *a = (comp_a_t *)wubu_ecs_get(e, id_a);
    comp_b_t *b = (comp_b_t *)wubu_ecs_get(e, id_b);
    assert(a && b);
    a->foo = 12345; a->bar = 3.14159f;
    for (int i = 0; i < 4; i++) b->big[i] = 1.5 * i + 0.25;

    /* Snapshot. */
    size_t sz;
    uint8_t *snap = wubu_ecs_snapshot(e, &sz);
    assert(snap && sz > 0);

    /* Mutate, then restore. */
    a->foo = -1; a->bar = 0.0f;
    b->big[0] = 999.0;
    int rc = wubu_ecs_restore(e, snap, sz);
    assert(rc == 0);
    assert(a->foo == 12345);
    assert(fabsf(a->bar - 3.14159f) < 1e-6f);
    assert(fabs(b->big[0] - 0.25) < 1e-9);
    assert(fabs(b->big[3] - 4.75) < 1e-9);

    /* Restore with wrong size must fail. */
    uint8_t *bad = (uint8_t *)malloc(sz + 4);
    memcpy(bad, snap, sz);
    assert(wubu_ecs_restore(e, bad, sz + 4) == -1);

    free(snap); free(bad);
    wubu_ecs_free(e);
    printf("ALL ECS COMPONENT-STORE TESTS PASSED\n");
    return 0;
}
