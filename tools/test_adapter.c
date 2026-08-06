/* test_adapter.c — tests for the hot-swappable AGI adapter framework
 *
 * Verifies:
 *   T1: register + lookup adapter works
 *   T2: duplicate registration is rejected
 *   T3: type-filtered lookup works
 *   T4: list returns all adapters of a type
 *   T5: hot-swap replaces a running adapter
 *   T6: compat shim redirects old name to new
 *   T7: compat check works (same major version)
 *
 * WaefreBeorn Umbrella License v3.0
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "wubu_adapter.h"
#include "wubu_adapter_compat.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { tests_failed++; printf("FAIL: %s\n", msg); } while(0)

/* Dummy forward function for test adapters */
static int dummy_forward(void *self, wubu_adapter_ctx_t *ctx,
                          const float *x, size_t n_in,
                          float *out, size_t n_out) {
    (void)self; (void)ctx;
    if (n_in < n_out) return -1;
    for (size_t i = 0; i < n_out; i++) out[i] = x[i] * 2.0f;
    return 0;
}
static void dummy_free(void *self) { (void)self; }

/* Build a test adapter on the stack (opaque to callers via ops) */
static wubu_adapter_t make_adapter(const char *name, const char *ver,
                                    wubu_component_type_t type) {
    static wubu_adapter_ops_t ops[16];  /* static pool for test */
    static int slot = 0;
    if (slot >= 16) slot = 0;
    wubu_adapter_ops_t *o = &ops[slot++];
    o->type = type;
    o->name = name;
    o->version = ver;
    o->init = NULL;
    o->free_fn = dummy_free;
    o->forward = dummy_forward;
    o->backward = NULL;
    wubu_adapter_t a;
    a.ops = o;
    return a;
}

int main(void) {
    wubu_adapter_t attn_local = make_adapter("attn.local", "1.0.0", WUBU_COMP_ATTN);
    wubu_adapter_t attn_sliding = make_adapter("attn.sliding", "1.1.0", WUBU_COMP_ATTN);
    wubu_adapter_t ffn_relu = make_adapter("ffn.relu", "1.0.0", WUBU_COMP_FFN);

    /* T1: register + lookup */
    printf("  [t1_register_lookup] ... ");
    fflush(stdout);
    if (wubu_adapter_register(&attn_local) != 0) { FAIL("register failed"); goto cleanup; }
    if (wubu_adapter_register(&attn_sliding) != 0) { FAIL("register failed"); goto cleanup; }
    if (wubu_adapter_register(&ffn_relu) != 0) { FAIL("register failed"); goto cleanup; }
    wubu_adapter_t *found = wubu_adapter_lookup("attn.local");
    if (!found) { FAIL("lookup returned NULL"); goto cleanup; }
    if (found->ops->forward(NULL, NULL, NULL, 0, NULL, 0) != 0)
        { FAIL("forward returned nonzero"); goto cleanup; }
    PASS();

    /* T2: duplicate registration rejected */
    printf("  [t2_dup_rejected] ... ");
    fflush(stdout);
    wubu_adapter_t attn_local_again = make_adapter("attn.local", "1.0.0", WUBU_COMP_ATTN);
    if (wubu_adapter_register(&attn_local_again) == 0)
        { FAIL("duplicate registration should fail"); goto cleanup; }
    PASS();

    /* T3: type-filtered lookup */
    printf("  [t3_type_lookup] ... ");
    fflush(stdout);
    wubu_adapter_t *attn_s = wubu_adapter_lookup_type(WUBU_COMP_ATTN, "attn.sliding");
    if (!attn_s) { FAIL("lookup_type returned NULL"); goto cleanup; }
    wubu_adapter_t *ffn_as_attn = wubu_adapter_lookup_type(WUBU_COMP_ATTN, "ffn.relu");
    if (ffn_as_attn) { FAIL("ffn.relu found as ATTN (wrong type)"); goto cleanup; }
    PASS();

    /* T4: list returns all adapters of a type */
    printf("  [t4_list] ... ");
    fflush(stdout);
    char names[4][64];
    memset(names, 0, sizeof(names));
    int count = wubu_adapter_list(WUBU_COMP_ATTN, names, 4);
    if (count != 2) {
        char buf[128]; snprintf(buf, sizeof(buf), "expected 2 attn, got %d", count);
        FAIL(buf); goto cleanup;
    }
    int found_local = 0, found_sliding = 0;
    for (int i = 0; i < count; i++) {
        if (strcmp(names[i], "attn.local") == 0) found_local = 1;
        if (strcmp(names[i], "attn.sliding") == 0) found_sliding = 1;
    }
    if (!found_local || !found_sliding)
        { FAIL("missing adapter in list"); goto cleanup; }
    PASS();

    /* T5: hot-swap (same major version so compat passes) */
    printf("  [t5_hot_swap] ... ");
    fflush(stdout);
    wubu_adapter_t attn_new = make_adapter("attn.local", "1.2.0", WUBU_COMP_ATTN);
    int rc = wubu_adapter_swap("attn.local", &attn_new);
    if (rc != 0) { FAIL("swap returned nonzero"); goto cleanup; }
    wubu_adapter_t *after = wubu_adapter_lookup("attn.local");
    if (!after || strcmp(after->ops->version, "1.2.0") != 0)
        { FAIL("swap didn't update version"); goto cleanup; }
    PASS();

    /* T6: compat shim redirects old name */
    printf("  [t6_compat_shim] ... ");
    fflush(stdout);
    wubu_adapter_t attn_v3 = make_adapter("attn.v3", "3.0.0", WUBU_COMP_ATTN);
    if (wubu_adapter_register(&attn_v3) != 0) { FAIL("register v3 failed"); goto cleanup; }
    /* Register a shim: attn.v2 → attn.v3 */
    if (wubu_adapter_register_shim("attn.v2", "attn.v3") != 0)
        { FAIL("register_shim failed"); goto cleanup; }
    /* attn.v2 doesn't exist, but the shim should redirect */
    wubu_adapter_t *via_shim = wubu_adapter_lookup("attn.v2");
    if (via_shim) { FAIL("exact lookup should be NULL for missing"); goto cleanup; }
    wubu_adapter_t *via_compat = wubu_adapter_lookup_compat("attn.v2");
    if (!via_compat) { FAIL("lookup_compat returned NULL via shim"); goto cleanup; }
    PASS();

    /* T7: compat check — same major vs different major */
    printf("  [t7_compat_check] ... ");
    fflush(stdout);
    wubu_adapter_ops_t v1_ops;
    v1_ops.type = WUBU_COMP_ATTN;
    v1_ops.name = "test.v1";
    v1_ops.version = "1.0.0";
    if (!wubu_adapter_compat(&v1_ops, "1.5.0"))
        { FAIL("same major should be compat"); goto cleanup; }
    if (wubu_adapter_compat(&v1_ops, "2.0.0"))
        { FAIL("different major should be incompat"); goto cleanup; }
    if (!wubu_adapter_compat(&v1_ops, NULL))
        { FAIL("NULL target should be compat"); goto cleanup; }
    PASS();

    printf("\n=== %d passed, %d failed ===\n", tests_passed, tests_failed);
    /* Don't shutdown — that would free our static ops */
    return tests_failed > 0 ? 1 : 0;

cleanup:
    printf("\n=== %d passed, %d failed ===\n", tests_passed, tests_failed);
    return 1;
}
