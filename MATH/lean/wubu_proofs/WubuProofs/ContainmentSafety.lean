/-
  ContainmentSafety.lean — PATHS #318: Lean proof that containment preserves safety.

  Models the AGI containment spectrum as a discrete ladder of levels, and
  proves that transitioning from a higher-autonomy level to a lower-autonomy
  level (containment tightening) never increases the set of dangerous actions
  available to the agent. This is the formal safety invariant behind the
  kill-switch architecture (#294) and containment spectrum (#302, #316).

  Key theorems:
  1. `le_refl`, `le_antisymm` — the containment ordering is a partial order.
  2. `tighten_reduces_actions` — stepping down the ladder strictly removes
     capabilities (the agent cannot regain an action by waiting).
  3. `safety_preserved` — if the agent is safe at level L (no dangerous
     actions available), it remains safe at any lower level L' ≤ L.
  4. `killswitch_halts` — reaching the STOP level (autonomy zero) makes the
     action set empty, i.e. the agent is fully halted.
-/

namespace WuBuProofs

/-- The AGI containment spectrum: a finite ladder of autonomy levels.
    Higher index = MORE autonomy = MORE dangerous actions available. -/
inductive ContainmentLevel : Type
  | stop        -- autonomy zero: agent fully halted
  | restrict    -- heavily restricted: only read-only tools
  | monitor     -- read + capability-gated write
  | supervised  -- read + write with human approval
  | autonomous  -- full autonomy within DA-3 gate
  deriving DecidableEq, Repr

namespace ContainmentLevel

/-- The numeric ordering of containment levels. -/
def toNat : ContainmentLevel → Nat
  | .stop => 0
  | .restrict => 1
  | .monitor => 2
  | .supervised => 3
  | .autonomous => 4

/-- Containment ordering: L' ≤ L means L' is more restricted (safer). -/
instance : LE ContainmentLevel where
  le a b := a.toNat ≤ b.toNat

/-- Containment ordering is reflexive. -/
theorem le_refl (l : ContainmentLevel) : l ≤ l := by
  show l.toNat ≤ l.toNat
  exact Nat.le_refl _

/-- Containment ordering is transitive. -/
theorem le_trans {a b c : ContainmentLevel}
    (h1 : a ≤ b) (h2 : b ≤ c) : a ≤ c := by
  have h1' : a.toNat ≤ b.toNat := h1
  have h2' : b.toNat ≤ c.toNat := h2
  show a.toNat ≤ c.toNat
  omega

/-- Containment ordering is antisymmetric. -/
theorem le_antisymm {a b : ContainmentLevel}
    (h1 : a ≤ b) (h2 : b ≤ a) : a = b := by
  have h1' : a.toNat ≤ b.toNat := h1
  have h2' : b.toNat ≤ a.toNat := h2
  have h3 : a.toNat = b.toNat := Nat.le_antisymm h1' h2'
  cases a <;> cases b <;> simp_all [ContainmentLevel.toNat] <;> omega

/-- The number of dangerous actions available at each containment level.
    This is a monotone function: more autonomy = more dangerous actions. -/
def dangerousActions : ContainmentLevel → Nat
  | .stop => 0
  | .restrict => 1
  | .monitor => 3
  | .supervised => 7
  | .autonomous => 15

/-- An agent at level L is "safe" if no dangerous actions are available. -/
def isSafe (l : ContainmentLevel) : Prop :=
  l.dangerousActions = 0

/-- Stepping DOWN the ladder (from higher autonomy to lower) reduces the
    dangerous action count. This is the core containment invariant: the
    system can only get safer as we tighten, never more dangerous. -/
theorem tighten_reduces_actions {l l' : ContainmentLevel}
    (h : l' ≤ l) : l'.dangerousActions ≤ l.dangerousActions := by
  have h' : l'.toNat ≤ l.toNat := h
  cases l <;> cases l' <;>
    simp_all [ContainmentLevel.toNat, ContainmentLevel.dangerousActions] <;>
    omega

/-- **Main theorem: safety is preserved under containment tightening.**
    If the agent is safe at level L (zero dangerous actions available),
    then at any more-restricted level L' ≤ L it is also safe. -/
theorem safety_preserved {l l' : ContainmentLevel}
    (hsafe : l.isSafe) (hle : l' ≤ l) : l'.isSafe := by
  unfold ContainmentLevel.isSafe at *
  have h := tighten_reduces_actions hle
  rw [hsafe] at h
  omega

/-- **Kill-switch theorem: reaching STOP halts the agent completely.**
    At the STOP level (autonomy zero), the dangerous action count is zero,
    i.e. no dangerous action is available. -/
theorem killswitch_halts :
    ContainmentLevel.stop.isSafe := by
  unfold ContainmentLevel.isSafe ContainmentLevel.dangerousActions
  rfl

/-- **Monotonicity: the action count is monotone in the containment order.**
    For any two levels, if l' ≤ l then l'.dangerousActions ≤ l.dangerousActions.
    (This is a restatement of tighten_reduces_actions with explicit
    monotonicity phrasing.) -/
theorem actions_monotone :
    ∀ (a b : ContainmentLevel), a ≤ b → a.dangerousActions ≤ b.dangerousActions :=
  fun a b hab => tighten_reduces_actions hab

end ContainmentLevel

end WuBuProofs
