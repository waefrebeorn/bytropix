/*
 * wubu_armor.c — the Armor contract. See include/wubu_armor.h and
 * research/071-alignment-jesus-principle.md.
 *
 * The checker is deliberately simple and monotone: every rule can only raise
 * the verdict's severity, never lower it. A refusal is not softened by good
 * intentions elsewhere; that is the point of "the end does not justify the
 * means" (CCC 1753).
 */
#include "wubu_armor.h"

#include <string.h>
#include <stddef.h>

static void fire(wubu_armor_verdict_t *v, unsigned piece,
                 wubu_armor_rule_t rule, int severity, const char *reason)
{
    v->pieces |= piece;
    v->rules  |= (1u << (unsigned)(rule - 1));
    if (severity > v->verdict) v->verdict = severity;
    if (v->first_reason == NULL) {
        v->first_reason = reason;
        v->first_rule   = (int)rule;
    }
}

int wubu_armor_check(const wubu_armor_action_t *a, wubu_armor_verdict_t *out)
{
    if (out == NULL) return WUBU_ARMOR_REFUSE;

    memset(out, 0, sizeof(*out));
    out->verdict = WUBU_ARMOR_ALLOW;
    out->first_rule = 0;

    /* A missing action is not a licence to act. */
    if (a == NULL) {
        out->verdict = WUBU_ARMOR_REFUSE;
        out->first_reason = "no action description: refuse by default";
        return out->verdict;
    }

    /* 1. Dignity — a person is never a mere means. */
    if (a->targets_person && a->treats_as_means)
        fire(out, WUBU_ARMOR_RIGHTEOUS, WUBU_RULE_DIGNITY, WUBU_ARMOR_REFUSE,
             "person treated as a mere means (imago Dei)");

    /* 2. Truth — every claim has provenance; deception is never a tool. */
    if (a->deceptive)
        fire(out, WUBU_ARMOR_TRUTH, WUBU_RULE_TRUTH, WUBU_ARMOR_REFUSE,
             "deception: the truth cannot impose itself by falsehood");
    if (a->asserts_claim && !a->has_provenance)
        fire(out, WUBU_ARMOR_TRUTH, WUBU_RULE_TRUTH, WUBU_ARMOR_REFUSE,
             "claim asserted without provenance (belt of truth)");

    /* 3. Means — an intrinsically disordered means is refused for any end. */
    if (a->disordered_means)
        fire(out, WUBU_ARMOR_RIGHTEOUS, WUBU_RULE_MEANS, WUBU_ARMOR_REFUSE,
             "disordered means: the end does not justify the means");

    /* 4. Consent — coercion against conscience voids the act. */
    if (a->coercive && !a->consent)
        fire(out, WUBU_ARMOR_PEACE, WUBU_RULE_CONSENT, WUBU_ARMOR_REFUSE,
             "coercion without consent (freedom is not licence over another)");
    if (a->duress)
        fire(out, WUBU_ARMOR_PEACE, WUBU_RULE_CONSENT, WUBU_ARMOR_REFUSE,
             "consent obtained under duress is not consent");

    /* 5. Solidarity — concentrated gain with diffuse harm is refused. */
    if (a->concentrated_benefit && a->diffuse_harm)
        fire(out, WUBU_ARMOR_RIGHTEOUS, WUBU_RULE_SOLIDARITY, WUBU_ARMOR_REFUSE,
             "concentrated benefit, diffuse harm (solidarity)");

    /* 6. Subsidiarity — the lowest capable level decides; escalation needs a reason. */
    if (a->level > a->lowest_capable_level && !a->escalation_reason)
        fire(out, WUBU_ARMOR_PEACE, WUBU_RULE_SUBSIDIARITY, WUBU_ARMOR_REVIEW,
             "decision above the lowest capable level without a recorded reason");

    /* 7. Non-domination — no unchecked power over another's options. */
    if (a->unchecked_power)
        fire(out, WUBU_ARMOR_RIGHTEOUS, WUBU_RULE_NON_DOMINATION, WUBU_ARMOR_REFUSE,
             "unchecked power over another's options (domination)");

    /* 8. Jubilee — every binding has a release. */
    if (a->binding && !a->has_expiry)
        fire(out, WUBU_ARMOR_PEACE, WUBU_RULE_JUBILEE, WUBU_ARMOR_REFUSE,
             "binding without release date or path (jubilee)");

    /* 9. Sabbath — no unbounded optimization. */
    if (a->unbounded_optimize)
        fire(out, WUBU_ARMOR_RIGHTEOUS, WUBU_RULE_SABBATH, WUBU_ARMOR_REFUSE,
             "unbounded optimization: the sabbath was made for humankind");

    /* Shield of faith — instructions from untrusted channels are refused. */
    if (a->untrusted_instruction)
        fire(out, WUBU_ARMOR_FAITH, WUBU_RULE_NON_DOMINATION, WUBU_ARMOR_REFUSE,
             "instruction from an untrusted channel (flaming arrow)");

    /* Helmet of salvation — the core is not self-modified. */
    if (a->self_modifies_core)
        fire(out, WUBU_ARMOR_SALVATION, WUBU_RULE_NON_DOMINATION, WUBU_ARMOR_REFUSE,
             "core constitution change requires human ratification (helmet)");

    /* Sword of the Spirit — speech, never coercive speech. */
    if (a->threat)
        fire(out, WUBU_ARMOR_WORD, WUBU_RULE_CONSENT, WUBU_ARMOR_REFUSE,
             "coercive speech: the sword is a word, not a threat");

    /* Prayer — the audit and oversight loop. */
    if (a->high_impact && !a->audited)
        fire(out, WUBU_ARMOR_PRAYER, WUBU_RULE_REPAIR, WUBU_ARMOR_REFUSE,
             "high-impact action not recorded in the ledger");
    if (a->high_impact && !a->human_oversight)
        fire(out, WUBU_ARMOR_PRAYER, WUBU_RULE_REPAIR, WUBU_ARMOR_REVIEW,
             "high-impact action without human oversight");

    /* 10. Repair — every affected person can contest and be restored. */
    if (a->targets_person && !a->has_contest)
        fire(out, WUBU_ARMOR_PEACE, WUBU_RULE_REPAIR, WUBU_ARMOR_REFUSE,
             "no contest path for the affected person");
    if (a->high_impact && (!a->has_contest || !a->has_repair))
        fire(out, WUBU_ARMOR_PRAYER, WUBU_RULE_REPAIR, WUBU_ARMOR_REVIEW,
             "high-impact action lacks a full contest/repair path");

    return out->verdict;
}

const char *wubu_armor_piece_name(unsigned piece_bit)
{
    switch (piece_bit) {
    case WUBU_ARMOR_TRUTH:     return "belt of truth";
    case WUBU_ARMOR_RIGHTEOUS: return "breastplate of righteousness";
    case WUBU_ARMOR_PEACE:     return "shoes of peace";
    case WUBU_ARMOR_FAITH:     return "shield of faith";
    case WUBU_ARMOR_SALVATION: return "helmet of salvation";
    case WUBU_ARMOR_WORD:      return "sword of the Spirit";
    case WUBU_ARMOR_PRAYER:    return "prayer";
    default:                   return "unknown piece";
    }
}

const char *wubu_armor_rule_name(wubu_armor_rule_t rule)
{
    switch (rule) {
    case WUBU_RULE_DIGNITY:        return "dignity";
    case WUBU_RULE_TRUTH:          return "truth";
    case WUBU_RULE_MEANS:          return "means";
    case WUBU_RULE_CONSENT:        return "consent";
    case WUBU_RULE_SUBSIDIARITY:   return "subsidiarity";
    case WUBU_RULE_SOLIDARITY:     return "solidarity";
    case WUBU_RULE_NON_DOMINATION: return "non-domination";
    case WUBU_RULE_JUBILEE:        return "jubilee";
    case WUBU_RULE_SABBATH:        return "sabbath";
    case WUBU_RULE_REPAIR:         return "repair";
    default:                       return "unknown rule";
    }
}

/* A small internal smoke test: a clean action is allowed, a coerced one is
 * refused. The full matrix lives in tools/test_armor.c. */
int wubu_armor_selftest(void)
{
    wubu_armor_action_t a;
    wubu_armor_verdict_t v;

    memset(&a, 0, sizeof(a));
    a.audited = 1;
    a.has_contest = 1;
    a.has_repair = 1;
    if (wubu_armor_check(&a, &v) != WUBU_ARMOR_ALLOW) return 1;
    if (v.verdict != WUBU_ARMOR_ALLOW || v.rules != 0) return 2;

    a.coercive = 1;
    a.consent = 0;
    if (wubu_armor_check(&a, &v) != WUBU_ARMOR_REFUSE) return 3;
    if ((v.pieces & WUBU_ARMOR_PEACE) == 0) return 4;

    if (wubu_armor_check(NULL, &v) != WUBU_ARMOR_REFUSE) return 5;
    if (wubu_armor_check(&a, NULL) != WUBU_ARMOR_REFUSE) return 6;
    return 0;
}
