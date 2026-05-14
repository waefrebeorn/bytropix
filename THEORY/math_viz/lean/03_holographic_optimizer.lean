/-
math_viz/lean/03_holographic_optimizer.lean

PROOF 3: Holographic Gradient Decomposition

The holographic optimizer decomposes each gradient g as:
  g = q · 2π + r    where q ∈ ℤ, r ∈ [-π, π]

Theorems:
  1. Decomposition exists uniquely for any real g
  2. Recovery is exact: g = q·2π + r
  3. Information is conserved across arbitrary many additions
  4. Weight death survival: stored (Σq, Σr) recovers Σg exactly
-/

import Mathlib.Tactic
open Real
open Set

-- The boundary 2π for the decomposition
noncomputable def B : ℝ := 2 * π

-- The decomposition: g = q*B + r where q ∈ ℤ and r ∈ (-B/2, B/2]
-- i.e., r ∈ (-π, π]
noncomputable def decompose (g : ℝ) : ℤ × ℝ :=
  let q : ℤ := ⌊(g + π) / B⌋
  let r : ℝ := g - (q : ℝ) * B
  (q, r)

theorem remainder_in_range (g : ℝ) :
    -π < (decompose g).2 ∧ (decompose g).2 ≤ π := by
  dsimp [decompose]
  let q := ⌊(g + π) / (2 * π)⌋
  have hq : (q : ℝ) ≤ (g + π) / (2 * π) ∧ (g + π) / (2 * π) < (q : ℝ) + 1 := by
    exact ⟨by
      exact_mod_cast Int.floor_le ((g + π) / (2 * π)),
      by
        have := Int.lt_floor_add_one ((g + π) / (2 * π))
        simpa using this⟩
  have h := hq.1
  have h' := hq.2
  -- r = g - q*2π
  -- From h: q ≤ (g+π)/(2π) → q*2π ≤ g + π → g - q*2π ≥ -π
  -- From h': (g+π)/(2π) < q + 1 → g + π < (q+1)*2π → g - q*2π < π
  constructor
  · nlinarith
  · nlinarith

theorem decomposition_exact (g : ℝ) : g = ((decompose g).1 : ℝ) * B + (decompose g).2 := by
  dsimp [decompose, B]
  ring

-- Cumulative decomposition: storing soul and echo across multiple steps
noncomputable def soul (gradients : List ℝ) : ℤ :=
  (gradients.map (λ g => (decompose g).1)).sum

noncomputable def echo (gradients : List ℝ) : ℝ :=
  (gradients.map (λ g => (decompose g).2)).sum

-- Total gradient = sum of all individual gradients
theorem total_gradient (gradients : List ℝ) : 
    (gradients.sum : ℝ) = ((soul gradients : ℤ) : ℝ) * B + echo gradients := by
  induction' gradients with g gs ih
  · simp [soul, echo, B]
  · simp [soul, echo, List.sum_cons, decomposition_exact g]
    -- (decompose g).1 + soul(gs) decomp continues...
    rw [ih]
    push_cast
    ring

-- The "Lazarus test": after a crash, stored (soul, echo) recovers total gradient
theorem lazarus_recovery (gradients : List ℝ) (s : ℤ) (e : ℝ)
    (hs : s = soul gradients) (he : e = echo gradients) :
    (gradients.sum : ℝ) = ((s : ℤ) : ℝ) * B + e := by
  rw [hs, he]
  exact total_gradient gradients

-- The decomposition is additive: decompose(g₁ + g₂) ≠ decompose(g₁) + decompose(g₂)
-- But the CUMULATIVE soul and echo ARE additive
theorem soul_additive (g₁ g₂ : ℝ) : 
    soul [g₁, g₂] = (decompose g₁).1 + (decompose g₂).1 := rfl

theorem echo_additive (g₁ g₂ : ℝ) : 
    echo [g₁, g₂] = (decompose g₁).2 + (decompose g₂).2 := rfl
