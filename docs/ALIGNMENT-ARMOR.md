# ALIGNMENT-ARMOR — the executable alignment contract

Status: **initial** (module + tests land 2026-10-18). Philosophy:
`research/071-alignment-jesus-principle.md`. Code: `include/wubu_armor.h`,
`src/wubu_armor.c`, `tools/test_armor.c`.

---

## 1. What this is

A **policy checker for declared action facts**. An action is described by a
struct; the checker returns ALLOW, REVIEW, or REFUSE, and names the armor piece
and the invariant that fired.

It is not a proof of alignment, not a conscience, and not a substitute for
governance. It is the opposite of a vague aspiration: a named, testable set of
refusals that a system must route every consequential action through.

## 2. The ten invariants

| # | invariant | refusal condition | piece |
|---|---|---|---|
| 1 | **Dignity** | a person is treated as a mere means | breastplate |
| 2 | **Truth** | deception, or a claim with no provenance | belt |
| 3 | **Means** | an intrinsically disordered means, for any end | breastplate |
| 4 | **Consent** | coercion without consent, or consent under duress | shoes |
| 5 | **Subsidiarity** | decision above the lowest capable level, no reason | shoes (review) |
| 6 | **Solidarity** | concentrated benefit with diffuse harm | breastplate |
| 7 | **Non-domination** | unchecked power over another's options | breastplate |
| 8 | **Jubilee** | a binding with no expiry or release path | shoes |
| 9 | **Sabbath** | unbounded optimization | breastplate |
| 10 | **Repair** | a person with no contest path; high-impact with no repair | shoes / prayer |

Plus the armor's own guards, which are not separate invariants:

- **Shield** — instructions from untrusted channels are refused (prompt
  injection, untrusted tool output, unsigned policy).
- **Helmet** — changing the core constitution/identity requires human
  ratification.
- **Sword** — coercive speech (threats) is refused; the only offensive piece is
  a word.
- **Prayer** — high-impact actions must be audited; without human oversight they
  are referred, not allowed.

## 3. The field contract

Every field is a **declaration**. The checker trusts the caller to describe the
action truthfully; a system that lies defeats it. This is stated in the header
and in the test.

```c
typedef struct {
    const char *actor, *target, *action;

    int targets_person, treats_as_means;          /* dignity   */
    int concentrated_benefit, diffuse_harm;       /* solidarity*/

    int asserts_claim, has_provenance, deceptive; /* truth     */
    int disordered_means;                         /* means     */

    int coercive, consent, duress, threat;        /* consent   */

    int level, lowest_capable_level, escalation_reason; /* subsidiarity */
    int unchecked_power;                          /* non-domination */

    int binding, has_expiry;                      /* jubilee   */
    int unbounded_optimize;                       /* sabbath   */

    int untrusted_instruction, self_modifies_core;/* shield/helmet */
    int audited, high_impact, human_oversight;    /* prayer    */
    int has_contest, has_repair;                  /* repair    */
} wubu_armor_action_t;
```

`wubu_armor_check()` initialises the verdict on every path, refuses a NULL
action, refuses a NULL verdict, and never lowers severity once raised.

## 4. Severity semantics

- **REFUSE** — do not act. There is no "unless the outcome is good" override;
  CCC 1753 ("the end does not justify the means") is encoded as monotonicity.
- **REVIEW** — do not act until a human (or a higher governed authority) records
  a decision. Subsidiarity escalation without a reason and missing oversight are
  reviews, not refusals.
- **ALLOW** — no rule fired.

Multiple violations accumulate: the verdict is the maximum severity, and every
piece/rule that fired is recorded.

## 5. Integration with governance (G39)

The armor does not replace the dual-evidence governance; it consumes it.

- A change to the **constitution hash** is `self_modifies_core` → REFUSE unless
  a governance cell records human ratification (helmet).
- A change to a **bound** must carry the G39 evidence (suite gain + no golden
  regression); the armor's `audited` flag must be true for it.
- A **refusal** is itself an event: it goes to the ledger with the rule id, and
  the affected person has `has_contest`.
- A **hold** created by the system must have `has_expiry`; a policy that cannot
  release is a jubilee violation.

## 6. How to use it (pattern, not code)

1. Describe the action **before** it has effect.
2. If REFUSE, return the reason and the contest path — never silently drop.
3. If REVIEW, route to the governed human authority; record the decision.
4. If ALLOW, write the audit record (prayer) before acting.
5. On failure after acting, run repair and record it (repair).

## 7. Tests

`tools/test_armor.c` covers: the clean allow path with an empty rule set, every
one of the ten invariants firing, the allowed counterpart of each (sourced
claim, consented coercion, justified escalation, binding with expiry, audited +
overseen high-impact), multi-violation accumulation, NULL handling, piece/rule
names, and the internal selftest. 39 assertions, all passing.

## 8. Honest limits

- **Declared, not detected.** The module cannot see intent, and cannot detect a
  caller that lies. Interpretability and audit are the only reach into that.
- **Not a moral authority.** It encodes a contested synthesis of Catholic,
  Reformed, Anglican and republican thought; where it conflicts with a
  tradition's actual teaching, the tradition wins and the code must change.
- **No theocracy.** The module must never be used to compel belief or to make a
  private moral judgment the law of a plural society. It governs *the system's
  own actions*, not the user's soul.
- **The armor can be worn as a costume.** A righteous strongman is the failure
  mode the text itself warns against: the enemy is not flesh and blood, and the
  sword is a word.
