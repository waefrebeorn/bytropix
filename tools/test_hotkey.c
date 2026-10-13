/* test_hotkey.c -- UX #238 tests: global summon hotkey (Ctrl+/). */
#include "wubu_hotkey.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

enum { K_SLASH = 47, K_T = 116, K_SPACE = 32, K_N = 110 };

int main(void)
{
    wubu_hotkey_t h;

    CHECK(wubu_hk_init(&h) == 0);
    CHECK(h.enabled == 1);

    /* the flagship: Ctrl+/ summons the command palette */
    CHECK(wubu_hk_bind(&h, WUBU_HK_CTRL, K_SLASH, "summon-palette") == 0);
    /* Ctrl+T terminal, Super+Space launcher */
    CHECK(wubu_hk_bind(&h, WUBU_HK_CTRL, K_T, "summon-terminal") == 0);
    CHECK(wubu_hk_bind(&h, WUBU_HK_SUPER, K_SPACE, "summon-launcher") == 0);

    /* conflict: same chord rejected */
    CHECK(wubu_hk_bind(&h, WUBU_HK_CTRL, K_SLASH, "other") == -1);
    CHECK(wubu_hk_has(&h, WUBU_HK_CTRL, K_SLASH) == 1);
    CHECK(wubu_hk_has(&h, WUBU_HK_CTRL, K_N) == 0);

    /* dispatch exact chord */
    CHECK(wubu_hk_dispatch(&h, WUBU_HK_CTRL, K_SLASH) == 1);
    CHECK(strcmp(h.last_command, "summon-palette") == 0);
    CHECK(h.n_fired == 1);

    /* modifier must match exactly: Ctrl+/ != Alt+/ != / */
    CHECK(wubu_hk_dispatch(&h, WUBU_HK_ALT, K_SLASH) == 0);
    CHECK(wubu_hk_dispatch(&h, 0, K_SLASH) == 0);
    CHECK(wubu_hk_dispatch(&h, WUBU_HK_CTRL | WUBU_HK_SHIFT, K_SLASH) == 0);

    /* other bound chords fire */
    CHECK(wubu_hk_dispatch(&h, WUBU_HK_CTRL, K_T) == 1);
    CHECK(strcmp(h.last_command, "summon-terminal") == 0);
    CHECK(wubu_hk_dispatch(&h, WUBU_HK_SUPER, K_SPACE) == 1);
    CHECK(strcmp(h.last_command, "summon-launcher") == 0);

    /* unbound key: no fire */
    CHECK(wubu_hk_dispatch(&h, WUBU_HK_CTRL, K_N) == 0);
    CHECK(h.last_fired == 0);

    /* unbind then dispatch fails */
    CHECK(wubu_hk_unbind(&h, WUBU_HK_CTRL, K_T) == 0);
    CHECK(wubu_hk_has(&h, WUBU_HK_CTRL, K_T) == 0);
    CHECK(wubu_hk_dispatch(&h, WUBU_HK_CTRL, K_T) == 0);
    CHECK(wubu_hk_unbind(&h, WUBU_HK_CTRL, K_T) == -1);   /* already gone */

    /* after unbind, the chord can be re-bound */
    CHECK(wubu_hk_bind(&h, WUBU_HK_CTRL, K_T, "summon-terminal-2") == 0);
    CHECK(wubu_hk_dispatch(&h, WUBU_HK_CTRL, K_T) == 1);
    CHECK(strcmp(h.last_command, "summon-terminal-2") == 0);

    /* disable: no hotkey fires while disabled */
    CHECK(wubu_hk_set_enabled(&h, 0) == 0);
    CHECK(wubu_hk_dispatch(&h, WUBU_HK_CTRL, K_SLASH) == 0);
    CHECK(h.n_fired == 4);   /* unchanged since disable */
    /* re-enable: fires again */
    CHECK(wubu_hk_set_enabled(&h, 1) == 0);
    CHECK(wubu_hk_dispatch(&h, WUBU_HK_CTRL, K_SLASH) == 1);

    /* bad args */
    CHECK(wubu_hk_bind(&h, WUBU_HK_CTRL, K_SLASH, NULL) == -1);
    CHECK(wubu_hk_bind(NULL, WUBU_HK_CTRL, K_SLASH, "x") == -1);
    CHECK(wubu_hk_dispatch(NULL, WUBU_HK_CTRL, K_SLASH) == -1);
    CHECK(wubu_hk_init(NULL) == -1);

    if (fails == 0) printf("ALL HOTKEY TESTS PASSED\n");
    else printf("%d HOTKEY TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
