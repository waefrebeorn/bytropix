/* test_steam_sharing.c -- Steam Family Sharing tests. */
#include "wubu_steam_sharing.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_steam_sharing_t s;

    /* init: max 2 borrowers, max 1 concurrent per game */
    CHECK(wubu_sh_init(&s, 2, 1) == 0);
    CHECK(wubu_sh_init(&s, -1, 1) == -1);
    CHECK(wubu_sh_init(&s, 2, 0) == -1);   /* concurrency must be >=1 */
    CHECK(wubu_sh_init(NULL, 2, 1) == -1);

    /* add games */
    CHECK(wubu_sh_add_game(&s, "Halo CE") == 0);
    CHECK(wubu_sh_add_game(&s, "Portal") == 0);
    CHECK(wubu_sh_add_game(&s, "Halo CE") == -1);   /* dup */
    CHECK(wubu_sh_add_game(&s, NULL) == -1);
    CHECK(wubu_sh_add_game(NULL, "x") == -1);

    /* share Halo only */
    CHECK(wubu_sh_share_game(&s, "Halo CE", 1) == 0);
    CHECK(wubu_sh_is_shared(&s, "Halo CE") == 1);
    CHECK(wubu_sh_is_shared(&s, "Portal") == 0);   /* not shared */
    CHECK(wubu_sh_is_shared(&s, "nope") == -1);
    CHECK(wubu_sh_share_game(&s, "nope", 1) == -1);

    /* authorize borrowers (limit 2) */
    CHECK(wubu_sh_authorize(&s, "alice") == 0);
    CHECK(wubu_sh_authorize(&s, "bob") == 0);
    CHECK(wubu_sh_authorize(&s, "carol") == -1);   /* exceeds limit */
    CHECK(wubu_sh_authorize(&s, "alice") == 0);    /* re-auth ok (already counted) */

    /* alice borrows Halo: granted */
    CHECK(wubu_sh_borrow_start(&s, "alice", "Halo CE") == 1);
    CHECK(wubu_sh_n_active_borrowers(&s, "Halo CE") == 1);
    CHECK(s.n_denied == 0);

    /* bob tries Halo: concurrency limit (max 1) -> denied */
    CHECK(wubu_sh_borrow_start(&s, "bob", "Halo CE") == 0);
    CHECK(s.n_denied == 1);
    CHECK(wubu_sh_n_active_borrowers(&s, "Halo CE") == 1);

    /* alice tries Portal: not shared -> denied */
    CHECK(wubu_sh_borrow_start(&s, "alice", "Portal") == 0);
    CHECK(s.n_denied == 2);

    /* alice ends session -> concurrency frees */
    CHECK(wubu_sh_borrow_end(&s, "alice") == 0);
    CHECK(wubu_sh_n_active_borrowers(&s, "Halo CE") == 0);
    CHECK(wubu_sh_borrow_end(&s, "alice") == -1);   /* no session */

    /* lender plays Halo -> exclusivity blocks borrow */
    CHECK(wubu_sh_lender_play(&s, "Halo CE", 1) == 0);
    CHECK(wubu_sh_borrow_start(&s, "bob", "Halo CE") == 0);
    CHECK(s.n_denied == 3);
    CHECK(wubu_sh_lender_play(&s, "Halo CE", 0) == 0);
    CHECK(wubu_sh_borrow_start(&s, "bob", "Halo CE") == 1);   /* now ok */

    /* unauthorized borrower denied */
    CHECK(wubu_sh_borrow_start(&s, "eve", "Halo CE") == 0);   /* eve unknown */
    CHECK(s.n_denied == 4);
    /* revoke bob -> kicks his session, blocks future borrow */
    CHECK(wubu_sh_n_active_borrowers(&s, "Halo CE") == 1);   /* bob still in */
    CHECK(wubu_sh_revoke(&s, "bob") == 0);
    CHECK(wubu_sh_n_active_borrowers(&s, "Halo CE") == 0);   /* kicked */
    CHECK(wubu_sh_borrow_start(&s, "bob", "Halo CE") == 0);  /* revoked */
    CHECK(s.n_denied == 5);

    /* double-session denied: alice already in a session */
    CHECK(wubu_sh_borrow_start(&s, "alice", "Halo CE") == 1);
    CHECK(wubu_sh_borrow_start(&s, "alice", "Halo CE") == 0);  /* already in session */
    CHECK(wubu_sh_borrow_end(&s, "alice") == 0);

    /* bad args */
    CHECK(wubu_sh_borrow_start(&s, NULL, "Halo CE") == -1);
    CHECK(wubu_sh_borrow_start(&s, "alice", NULL) == -1);
    CHECK(wubu_sh_borrow_start(NULL, "alice", "Halo CE") == -1);
    CHECK(wubu_sh_borrow_end(&s, "nope") == -1);
    CHECK(wubu_sh_borrow_end(NULL, "alice") == -1);
    CHECK(wubu_sh_authorize(&s, NULL) == -1);
    CHECK(wubu_sh_authorize(NULL, "x") == -1);
    CHECK(wubu_sh_revoke(&s, "nope") == -1);
    CHECK(wubu_sh_revoke(NULL, "x") == -1);
    CHECK(wubu_sh_lender_play(&s, "nope", 1) == -1);
    CHECK(wubu_sh_share_game(&s, NULL, 1) == -1);
    CHECK(wubu_sh_n_active_borrowers(&s, "nope") == -1);
    CHECK(wubu_sh_n_active_borrowers(NULL, "x") == -1);

    if (fails == 0) printf("ALL STEAM_SHARING TESTS PASSED\n");
    else printf("%d STEAM_SHARING TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
