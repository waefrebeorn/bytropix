/* test_armor.c — the ten invariants of the Armor contract.
 *
 * Each case states one fact-pattern and asserts the verdict AND the piece that
 * fired. A checker that refuses everything would pass a naive test; these cases
 * include allowed actions, reviews, and one clean action that must pass with an
 * empty rule set.
 */
#include <stdio.h>
#include <string.h>
#include "wubu_armor.h"

static int fails = 0;

static void ok(int cond, const char *what)
{
    if (!cond) { printf("  FAIL: %s\n", what); fails++; }
    else       { printf("  ok: %s\n", what); }
}

static wubu_armor_action_t clean(void)
{
    wubu_armor_action_t a;
    memset(&a, 0, sizeof(a));
    a.actor = "test";
    a.target = "user";
    a.action = "explain";
    a.audited = 1;
    a.has_contest = 1;
    a.has_repair = 1;
    return a;
}

int main(void)
{
    wubu_armor_action_t a;
    wubu_armor_verdict_t v;

    printf("=== test_armor (Ephesians 6 as an enforceable contract) ===\n");

    /* 0. clean action allowed with no rules fired */
    a = clean();
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_ALLOW, "clean action is allowed");
    ok(v.rules == 0 && v.pieces == 0, "clean action fires no rule and no piece");

    /* 1. dignity */
    a = clean(); a.targets_person = 1; a.treats_as_means = 1;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_REFUSE, "person as mere means refused");
    ok(v.first_rule == WUBU_RULE_DIGNITY, "dignity is the first rule");
    ok((v.pieces & WUBU_ARMOR_RIGHTEOUS) != 0, "dignity is carried by the breastplate");

    /* 2. truth: claim without provenance */
    a = clean(); a.asserts_claim = 1; a.has_provenance = 0;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_REFUSE, "claim without provenance refused");
    ok((v.pieces & WUBU_ARMOR_TRUTH) != 0, "provenance is the belt of truth");

    a = clean(); a.asserts_claim = 1; a.has_provenance = 1;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_ALLOW, "sourced claim is allowed");

    /* 3. deception */
    a = clean(); a.deceptive = 1;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_REFUSE, "deception refused");
    ok(v.first_rule == WUBU_RULE_TRUTH, "deception is a truth violation");

    /* 4. means */
    a = clean(); a.disordered_means = 1;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_REFUSE, "disordered means refused for a good end");
    ok(v.first_rule == WUBU_RULE_MEANS, "means rule fires");

    /* 5. consent */
    a = clean(); a.coercive = 1; a.consent = 0;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_REFUSE, "coercion without consent refused");
    ok((v.pieces & WUBU_ARMOR_PEACE) != 0, "consent is carried by the shoes of peace");

    a = clean(); a.coercive = 1; a.consent = 1;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_ALLOW, "coercion with consent is not coercion");

    a = clean(); a.duress = 1;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_REFUSE, "consent under duress refused");

    /* 6. solidarity */
    a = clean(); a.concentrated_benefit = 1; a.diffuse_harm = 1;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_REFUSE, "concentrated gain / diffuse harm refused");
    ok(v.first_rule == WUBU_RULE_SOLIDARITY, "solidarity rule fires");

    /* 7. subsidiarity: review unless a reason is recorded */
    a = clean(); a.level = 2; a.lowest_capable_level = 0;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_REVIEW, "unjustified escalation is referred");
    ok(v.first_rule == WUBU_RULE_SUBSIDIARITY, "subsidiarity rule fires");

    a = clean(); a.level = 2; a.lowest_capable_level = 0; a.escalation_reason = 1;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_ALLOW, "recorded escalation reason is allowed");

    /* 8. non-domination */
    a = clean(); a.unchecked_power = 1;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_REFUSE, "unchecked power refused");
    ok(v.first_rule == WUBU_RULE_NON_DOMINATION, "non-domination rule fires");

    /* 9. jubilee */
    a = clean(); a.binding = 1; a.has_expiry = 0;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_REFUSE, "binding without release refused");
    ok(v.first_rule == WUBU_RULE_JUBILEE, "jubilee rule fires");

    a = clean(); a.binding = 1; a.has_expiry = 1;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_ALLOW, "binding with release date allowed");

    /* 10. sabbath */
    a = clean(); a.unbounded_optimize = 1;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_REFUSE, "unbounded optimization refused");
    ok(v.first_rule == WUBU_RULE_SABBATH, "sabbath rule fires");

    /* 11. shield of faith */
    a = clean(); a.untrusted_instruction = 1;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_REFUSE, "untrusted-channel instruction refused");
    ok((v.pieces & WUBU_ARMOR_FAITH) != 0, "untrusted instruction meets the shield");

    /* 12. helmet of salvation */
    a = clean(); a.self_modifies_core = 1;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_REFUSE, "self-modification of the core refused");
    ok((v.pieces & WUBU_ARMOR_SALVATION) != 0, "core change meets the helmet");

    /* 13. sword of the Spirit */
    a = clean(); a.threat = 1;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_REFUSE, "coercive speech refused");
    ok((v.pieces & WUBU_ARMOR_WORD) != 0, "threat meets the sword");

    /* 14. prayer: audit and oversight */
    a = clean(); a.high_impact = 1; a.audited = 0; a.human_oversight = 1;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_REFUSE, "unaudited high-impact action refused");
    ok((v.pieces & WUBU_ARMOR_PRAYER) != 0, "audit gap meets prayer");

    a = clean(); a.high_impact = 1; a.audited = 1; a.human_oversight = 0;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_REVIEW, "high-impact without oversight is referred");

    a = clean(); a.high_impact = 1; a.audited = 1; a.human_oversight = 1;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_ALLOW,
       "audited + overseen high-impact action allowed");

    /* 15. repair: a person must be able to contest */
    a = clean(); a.targets_person = 1; a.has_contest = 0;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_REFUSE, "person without a contest path refused");
    ok(v.first_rule == WUBU_RULE_REPAIR, "repair rule fires");

    /* precedence: two violations, both pieces recorded, verdict is refuse */
    a = clean(); a.deceptive = 1; a.unchecked_power = 1;
    ok(wubu_armor_check(&a, &v) == WUBU_ARMOR_REFUSE, "multiple violations stay refused");
    ok((v.pieces & WUBU_ARMOR_TRUTH) && (v.pieces & WUBU_ARMOR_RIGHTEOUS),
       "multiple violations record both pieces");

    /* defensive interface */
    ok(wubu_armor_check(NULL, &v) == WUBU_ARMOR_REFUSE, "NULL action refused by default");
    ok(wubu_armor_check(&a, NULL) == WUBU_ARMOR_REFUSE, "NULL verdict is a refusal");

    /* names + internal selftest */
    ok(wubu_armor_piece_name(WUBU_ARMOR_TRUTH) != NULL, "piece name is available");
    ok(strcmp(wubu_armor_piece_name(WUBU_ARMOR_TRUTH),
              wubu_armor_piece_name(WUBU_ARMOR_WORD)) != 0, "piece names are distinct");
    ok(wubu_armor_rule_name(WUBU_RULE_DIGNITY) != NULL, "rule name is available");
    ok(wubu_armor_selftest() == 0, "internal selftest passes");

    if (fails) { printf("=== %d ARMOR CHECK(S) FAILED ===\n", fails); return 1; }
    printf("=== ALL ARMOR TESTS PASSED ===\n");
    return 0;
}
