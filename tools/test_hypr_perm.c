/* test_hypr_perm.c -- OMARCHY #281 tests: Hyprland permission management. */
#include "wubu_hypr_perm.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_hypr_perm_t hp;
    CHECK(wubu_hp_init(&hp) == 0);
    CHECK(hp.n_prompts == 0);
    CHECK(hp.next_id == 1);

    /* fresh screenshot request: raises a prompt, denied (default fail-closed) */
    int r = wubu_hp_request(&hp, WUBU_HP_OP_SCREENSHOT, "grim", "eDP-1");
    CHECK(r == 0);                          /* denied -> 0 */
    CHECK(hp.n_prompts == 1);
    CHECK(wubu_hp_pending(&hp) == 1);
    CHECK(hp.prompts[0].active == 1);
    CHECK(hp.prompts[0].id == 1);
    CHECK(hp.prompts[0].decision == WUBU_HP_DECISION_NONE);
    CHECK(hp.n_denied == 1);

    /* re-request same while pending: still denied, no new prompt */
    CHECK(wubu_hp_request(&hp, WUBU_HP_OP_SCREENSHOT, "grim", "eDP-1") == 0);
    CHECK(hp.n_prompts == 1);
    CHECK(hp.n_denied == 2);

    /* user answers NO: denied, prompt closed */
    CHECK(wubu_hp_answer(&hp, 1, WUBU_HP_DECISION_DENY) == 0);
    CHECK(hp.prompts[0].active == 0);
    CHECK(hp.prompts[0].decision == WUBU_HP_DECISION_DENY);
    CHECK(wubu_hp_pending(&hp) == 0);

    /* same request again: not always-granted, so re-prompts and denies */
    CHECK(wubu_hp_request(&hp, WUBU_HP_OP_SCREENSHOT, "grim", "eDP-1") == 0);
    CHECK(hp.n_prompts == 2);

    /* --- always grant persists --- */
    CHECK(wubu_hp_init(&hp) == 0);
    CHECK(wubu_hp_request(&hp, WUBU_HP_OP_SCREENSHOT, "grim", "eDP-1") == 0);
    int aid = hp.prompts[0].id;
    CHECK(wubu_hp_answer(&hp, aid, WUBU_HP_DECISION_ALLOW_ALWAYS) == 0);
    CHECK(hp.prompts[0].always == 1);
    /* now admitted immediately, no new prompt */
    CHECK(wubu_hp_request(&hp, WUBU_HP_OP_SCREENSHOT, "grim", "eDP-1") == 1);
    CHECK(hp.n_prompts == 1);
    CHECK(hp.n_admitted == 2);   /* 1 from YES-always answer + 1 from immediate admit */
    /* has_grant reflects the persisted grant */
    CHECK(wubu_hp_has_grant(&hp, WUBU_HP_OP_SCREENSHOT, "grim", "eDP-1") == 1);
    /* different resource still prompts */
    CHECK(wubu_hp_request(&hp, WUBU_HP_OP_SCREENSHOT, "grim", "HDMI-A-1") == 0);
    CHECK(hp.n_prompts == 2);
    /* different requester still prompts */
    CHECK(wubu_hp_request(&hp, WUBU_HP_OP_SCREENSHOT, "slurp", "eDP-1") == 0);
    CHECK(hp.n_prompts == 3);
    /* different op still prompts */
    CHECK(wubu_hp_request(&hp, WUBU_HP_OP_INJECT, "grim", "eDP-1") == 0);
    CHECK(hp.n_prompts == 4);

    /* revoke grim's grants */
    CHECK(wubu_hp_revoke(&hp, "grim") == 1);
    CHECK(wubu_hp_has_grant(&hp, WUBU_HP_OP_SCREENSHOT, "grim", "eDP-1") == 0);
    /* after revoke, re-request prompts (fail-closed) */
    CHECK(wubu_hp_request(&hp, WUBU_HP_OP_SCREENSHOT, "grim", "eDP-1") == 0);

    /* slurp never had an always grant, so still prompts */
    CHECK(wubu_hp_has_grant(&hp, WUBU_HP_OP_SCREENSHOT, "slurp", "eDP-1") == 0);
    CHECK(wubu_hp_request(&hp, WUBU_HP_OP_SCREENSHOT, "slurp", "eDP-1") == 0);

    /* answer YES admits once but does not persist */
    CHECK(wubu_hp_init(&hp) == 0);
    CHECK(wubu_hp_request(&hp, WUBU_HP_OP_GLOBAL_HK, "term", "Ctrl+Shift+T") == 0);
    int yid = hp.prompts[0].id;
    CHECK(wubu_hp_answer(&hp, yid, WUBU_HP_DECISION_ALLOW) == 0);
    CHECK(hp.prompts[0].always == 0);
    CHECK(hp.prompts[0].decision == WUBU_HP_DECISION_ALLOW);
    /* next request re-prompts (allow not persisted) */
    CHECK(wubu_hp_request(&hp, WUBU_HP_OP_GLOBAL_HK, "term", "Ctrl+Shift+T") == 0);
    CHECK(hp.n_prompts == 2);

    /* invalid ops / bad args */
    CHECK(wubu_hp_answer(&hp, 99999, WUBU_HP_DECISION_ALLOW) == -1);
    CHECK(wubu_hp_request(&hp, (wubu_hp_op_t)99, "x", "y") == -1);
    CHECK(wubu_hp_request(NULL, WUBU_HP_OP_SCREENSHOT, "x", "y") == -1);
    CHECK(wubu_hp_request(&hp, WUBU_HP_OP_SCREENSHOT, NULL, "y") == -1);
    CHECK(wubu_hp_has_grant(NULL, WUBU_HP_OP_SCREENSHOT, "x", "y") == 0);
    CHECK(wubu_hp_init(NULL) == -1);
    CHECK(wubu_hp_revoke(NULL, "x") == -1);
    CHECK(wubu_hp_pending(&hp) >= 0);

    if (fails == 0) printf("ALL HYPR_PERM TESTS PASSED\n");
    else printf("%d HYPR_PERM TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
