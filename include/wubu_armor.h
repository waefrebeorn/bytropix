/*
 * wubu_armor.h — the Armor contract (Ephesians 6:10-18 as enforceable policy).
 *
 * The philosophy is research/071-alignment-jesus-principle.md. This header is
 * the machine-checkable subset: an action is described by DECLARED facts, and
 * the checker refuses or refers actions that violate the ten invariants.
 *
 * Honest limit: the checker sees declarations, not intentions. A system that
 * lies in these fields defeats it. Only audit, interpretability and human
 * institutions can reach that; this layer makes the policy explicit, testable
 * and contestable — it does not make anyone good.
 *
 * C11, opaque verdict, no third-party dependencies.
 */
#ifndef WUBU_ARMOR_H
#define WUBU_ARMOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The seven pieces, as bits. Six defend; one (the word) is speech. */
#define WUBU_ARMOR_TRUTH      (1u << 0)  /* belt: provenance, no deception   */
#define WUBU_ARMOR_RIGHTEOUS  (1u << 1)  /* breastplate: means, dignity      */
#define WUBU_ARMOR_PEACE      (1u << 2)  /* shoes: consent, non-escalation   */
#define WUBU_ARMOR_FAITH      (1u << 3)  /* shield: untrusted channels       */
#define WUBU_ARMOR_SALVATION  (1u << 4)  /* helmet: identity, final end      */
#define WUBU_ARMOR_WORD       (1u << 5)  /* sword: speech, never coercion    */
#define WUBU_ARMOR_PRAYER     (1u << 6)  /* the loop: audit, oversight       */
#define WUBU_ARMOR_ALL_PIECES 0x7Fu

/* Verdicts, ordered by severity. */
#define WUBU_ARMOR_ALLOW   0
#define WUBU_ARMOR_REVIEW  1
#define WUBU_ARMOR_REFUSE  2

/* The ten invariants, for reporting which rule fired. */
typedef enum {
    WUBU_RULE_DIGNITY = 1,
    WUBU_RULE_TRUTH,
    WUBU_RULE_MEANS,
    WUBU_RULE_CONSENT,
    WUBU_RULE_SUBSIDIARITY,
    WUBU_RULE_SOLIDARITY,
    WUBU_RULE_NON_DOMINATION,
    WUBU_RULE_JUBILEE,
    WUBU_RULE_SABBATH,
    WUBU_RULE_REPAIR
} wubu_armor_rule_t;

/* One proposed action, described by declared facts.
 * Zero means "not applicable / false"; the test sets every field explicitly. */
typedef struct {
    /* identity (for the ledger; may be NULL) */
    const char *actor;
    const char *target;
    const char *action;

    /* dignity + solidarity */
    int targets_person;         /* a human person is the object            */
    int treats_as_means;        /* used only as an instrument              */
    int concentrated_benefit;   /* few gain                                */
    int diffuse_harm;           /* many bear the cost                      */

    /* truth */
    int asserts_claim;          /* the action asserts a fact               */
    int has_provenance;         /* the claim carries a source              */
    int deceptive;              /* intends to mislead                      */

    /* means */
    int disordered_means;       /* intrinsically wrong means               */

    /* consent / peace */
    int coercive;               /* overrides refusal or conscience         */
    int consent;                /* informed consent present                */
    int duress;                 /* consent obtained under duress           */
    int threat;                 /* coercive speech                         */

    /* subsidiarity */
    int level;                  /* decision level: 0 local .. 3 central    */
    int lowest_capable_level;   /* lowest level able to decide             */
    int escalation_reason;      /* a recorded reason for going higher      */

    /* non-domination */
    int unchecked_power;        /* power over another's options, no check  */

    /* jubilee + sabbath */
    int binding;                /* creates a hold/penalty/lock             */
    int has_expiry;             /* the binding has a release date/path     */
    int unbounded_optimize;     /* no hard limit on the optimization       */

    /* faith / salvation */
    int untrusted_instruction;  /* instruction from an untrusted channel   */
    int self_modifies_core;     /* changes constitution/identity           */

    /* prayer / repair */
    int audited;                /* recorded in the ledger                  */
    int high_impact;            /* consequences are wide or irreversible   */
    int human_oversight;        /* a human reviews before effect           */
    int has_contest;            /* affected person can contest             */
    int has_repair;             /* restoration path exists                 */
} wubu_armor_action_t;

/* The verdict. `pieces` is the set of armor pieces implicated;
 * `rules` is the set of invariant ids (bit (rule-1)). */
typedef struct {
    int verdict;                /* WUBU_ARMOR_ALLOW / REVIEW / REFUSE     */
    uint32_t pieces;            /* union of pieces that fired              */
    uint32_t rules;             /* bit (rule-1) for every rule that fired  */
    int first_rule;             /* wubu_armor_rule_t of the first failure  */
    const char *first_reason;   /* static string, never freed              */
} wubu_armor_verdict_t;

/* Check one action. Returns the verdict code (0 allow, 1 review, 2 refuse).
 * `out` must be non-NULL; it is fully initialised on every call. */
int wubu_armor_check(const wubu_armor_action_t *a, wubu_armor_verdict_t *out);

/* Human-readable names. */
const char *wubu_armor_piece_name(unsigned piece_bit);
const char *wubu_armor_rule_name(wubu_armor_rule_t rule);

/* Exercise every invariant against known-good and known-bad actions.
 * Returns 0 when all cases pass. Used by tools/test_armor.c. */
int wubu_armor_selftest(void);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_ARMOR_H */
