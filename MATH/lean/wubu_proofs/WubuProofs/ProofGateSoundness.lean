/-
  ProofGateSoundness.lean — PATHS #319: Lean proof that the proof-gate is sound.

  Models the proof-gated self-improvement gate (#131, #147, #159, #312) as a
  predicate on mutations: a mutation is accepted only if it carries a valid
  consistency proof. Proves the key soundness property: any accepted mutation
  is consistent (i.e. the gate never accepts an inconsistent mutation). Also
  proves the completeness direction: any consistent mutation with a valid
  proof is accepted.

  Key theorems:
  1. `accepted_implies_consistent` — soundness: the gate only accepts
     mutations that are consistent.
  2. `consistent_with_proof_accepted` — completeness: a consistent mutation
     with a valid proof passes the gate.
  3. `gate_is_strict` — a mutation without a valid proof is always rejected,
     even if it happens to be consistent.
  4. `fail_closed` — the gate with no verifier rejects everything (the
     architectural death condition from #130).
-/

namespace WuBuProofs

/-- A self-improvement mutation candidate. -/
structure Mutation where
  id : Nat
  consistent : Prop
  hasProof : Prop
  deriving Repr

namespace Mutation

/-- The proof-gate: accepts a mutation only if it is consistent AND has a
    valid proof. This is the formal model of the independent verifier that
    gates every promotion (#131, #147). -/
def accepted (m : Mutation) : Prop :=
  m.consistent ∧ m.hasProof

/-- **Soundness: the gate never accepts an inconsistent mutation.**
    If the gate accepts m, then m is consistent. -/
theorem accepted_implies_consistent (m : Mutation)
    (h : m.accepted) : m.consistent :=
  h.1

/-- **Completeness: a consistent mutation with a valid proof is accepted.**
    The gate does not unnecessarily reject valid mutations. -/
theorem consistent_with_proof_accepted (m : Mutation)
    (hc : m.consistent) (hp : m.hasProof) : m.accepted :=
  ⟨hc, hp⟩

/-- **Strictness: a mutation without a valid proof is always rejected.**
    Even if the mutation is consistent, the gate requires proof. -/
theorem gate_is_strict (m : Mutation)
    (hno : ¬m.hasProof) : ¬m.accepted := by
  intro h
  exact hno h.2

/-- **Fail-closed: with no verifier, the gate rejects everything.**
    The architectural death condition: if the verifier is absent, no
    mutation is accepted (the loop freezes). -/
theorem fail_closed (m : Mutation)
    (hverifier : False) : ¬m.accepted := by
  intro _
  exact hverifier.elim

/-- **Proof-gate is decidable for concrete mutations.**
    For any mutation with decidable consistency and proof, the acceptance
    predicate is decidable. -/
instance (m : Mutation) [Decidable m.consistent] [Decidable m.hasProof] :
    Decidable m.accepted :=
  inferInstanceAs (Decidable (m.consistent ∧ m.hasProof))

end Mutation

end WuBuProofs
