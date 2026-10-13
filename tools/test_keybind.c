/* test_keybind.c -- OMARCHY #292 tests: Omarchy-style keybind-driven UX. */
#include "wubu_keybind.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

/* illustrative key codes */
enum { K_SUPER=133, K_T=116, K_W=119, K_Q=113, K_B=98, K_1=49, K_2=50, K_X=120 };

int main(void)
{
    wubu_keybind_t k;

    /* init: root layer */
    CHECK(wubu_kb_init(&k) == 0);
    CHECK(strcmp(k.active_layer, "root") == 0);
    CHECK(wubu_kb_pending(&k) == 0);

    /* layers */
    CHECK(wubu_kb_add_layer(&k, "root") == -1);       /* dup */
    CHECK(wubu_kb_add_layer(&k, "media") == 0);
    CHECK(wubu_kb_set_layer(&k, "media") == 0);
    CHECK(strcmp(k.active_layer, "media") == 0);
    CHECK(wubu_kb_set_layer(&k, "root") == 0);
    CHECK(wubu_kb_set_layer(&k, "nope") == -1);       /* unknown */

    /* bind single-key commands in root */
    int t1[] = {K_Q};
    CHECK(wubu_kb_bind(&k, "root", t1, 1, "close-window") == 0);
    int t2[] = {K_W};
    CHECK(wubu_kb_bind(&k, "root", t2, 1, "toggle-float") == 0);

    /* bind a leader chord: SUPER T -> launch-terminal */
    int lt[] = {K_SUPER, K_T};
    CHECK(wubu_kb_bind(&k, "root", lt, 2, "launch-terminal") == 0);
    /* SUPER B -> launch-browser */
    int lb[] = {K_SUPER, K_B};
    CHECK(wubu_kb_bind(&k, "root", lb, 2, "launch-browser") == 0);
    /* SUPER 1 -> workspace-1 (shares the SUPER prefix) */
    int l1[] = {K_SUPER, K_1};
    CHECK(wubu_kb_bind(&k, "root", l1, 2, "workspace-1") == 0);

    /* single-key fire */
    CHECK(wubu_kb_key(&k, K_Q) == 1);
    CHECK(strcmp(k.last_command, "close-window") == 0);
    CHECK(wubu_kb_pending(&k) == 0);

    /* leader chord: SUPER arms, T fires */
    CHECK(wubu_kb_key(&k, K_SUPER) == 0);             /* consumed as prefix */
    CHECK(wubu_kb_pending(&k) == 1);
    CHECK(wubu_kb_key(&k, K_T) == 1);                 /* fires */
    CHECK(strcmp(k.last_command, "launch-terminal") == 0);
    CHECK(wubu_kb_pending(&k) == 0);

    /* same SUPER prefix, different second key */
    CHECK(wubu_kb_key(&k, K_SUPER) == 0);
    CHECK(wubu_kb_key(&k, K_B) == 1);
    CHECK(strcmp(k.last_command, "launch-browser") == 0);
    CHECK(wubu_kb_key(&k, K_SUPER) == 0);
    CHECK(wubu_kb_key(&k, K_1) == 1);
    CHECK(strcmp(k.last_command, "workspace-1") == 0);

    /* unknown key after SUPER aborts (fail-safe), no fire */
    CHECK(wubu_kb_key(&k, K_SUPER) == 0);
    CHECK(wubu_kb_pending(&k) == 1);
    CHECK(wubu_kb_key(&k, K_X) == -1);                /* SUPER X unbound -> abort */
    CHECK(wubu_kb_pending(&k) == 0);

    /* a lone unbound key aborts */
    CHECK(wubu_kb_key(&k, K_X) == -1);

    /* layer-scoped binding: media layer gets its own SUPER T */
    int mt[] = {K_SUPER, K_T};
    CHECK(wubu_kb_bind(&k, "media", mt, 2, "media-playpause") == 0);
    /* in root, SUPER T is still launch-terminal */
    CHECK(wubu_kb_key(&k, K_SUPER) == 0);
    CHECK(wubu_kb_key(&k, K_T) == 1);
    CHECK(strcmp(k.last_command, "launch-terminal") == 0);
    /* switch to media: SUPER T now fires media-playpause */
    CHECK(wubu_kb_set_layer(&k, "media") == 0);
    CHECK(wubu_kb_key(&k, K_SUPER) == 0);
    CHECK(wubu_kb_key(&k, K_T) == 1);
    CHECK(strcmp(k.last_command, "media-playpause") == 0);
    /* root-only binding (K_Q) is inactive in media layer */
    CHECK(wubu_kb_key(&k, K_Q) == -1);                /* not bound in media */

    /* switching layer aborts a pending chord */
    CHECK(wubu_kb_set_layer(&k, "root") == 0);
    CHECK(wubu_kb_key(&k, K_SUPER) == 0);
    CHECK(wubu_kb_pending(&k) == 1);
    CHECK(wubu_kb_set_layer(&k, "media") == 0);
    CHECK(wubu_kb_pending(&k) == 0);

    /* explicit reset aborts pending */
    CHECK(wubu_kb_set_layer(&k, "root") == 0);
    CHECK(wubu_kb_key(&k, K_SUPER) == 0);
    CHECK(wubu_kb_reset(&k) == 0);
    CHECK(wubu_kb_pending(&k) == 0);

    /* bad args */
    CHECK(wubu_kb_bind(&k, "root", NULL, 1, "x") == -1);
    int z[1] = {0};
    CHECK(wubu_kb_bind(&k, "root", z, 0, "x") == -1);   /* n_keys 0 */
    CHECK(wubu_kb_key(NULL, K_T) == -1);

    if (fails == 0) printf("ALL KEYBIND TESTS PASSED\n");
    else printf("%d KEYBIND TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
