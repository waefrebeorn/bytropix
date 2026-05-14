/-
math_viz/lean/01_golden_ratio.lean

PROOF 1: Golden Ratio Identities
φ = (1 + √5) / 2

Theorems:
  1. φ² = φ + 1
  2. 1/φ = φ - 1
  3. φ⁻¹ + φ⁻² = 1
-/

import Mathlib.Tactic
open Real

-- Define φ as the positive root of x² - x - 1 = 0
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

-- The defining equation of φ
theorem phi_def_eq : φ ^ 2 - φ - 1 = 0 := by
  dsimp [φ]
  ring
  -- φ² = ((1+√5)/2)² = (1 + 2√5 + 5)/4 = (6 + 2√5)/4 = (3 + √5)/2
  -- φ + 1 = (1+√5)/2 + 1 = (3+√5)/2
  -- So φ² = φ + 1, meaning φ² - φ - 1 = 0
  nlinarith [show (Real.sqrt 5) ^ 2 = (5 : ℝ) from Real.pow_sqrt_eq_abs 5, 
    by positivity]

theorem phi_sq_eq_phi_plus_one : φ ^ 2 = φ + 1 := by
  -- From φ² - φ - 1 = 0, add φ + 1 to both sides
  linarith [phi_def_eq]

theorem phi_inv_eq_phi_minus_one : φ⁻¹ = φ - 1 := by
  calc
    φ⁻¹ = 1 / φ := by ring
    _ = 1 / ((1 + Real.sqrt 5) / 2) := rfl
    _ = 2 / (1 + Real.sqrt 5) := by ring
    _ = 2 * (1 - Real.sqrt 5) / ((1 + Real.sqrt 5) * (1 - Real.sqrt 5)) := by
      field_simp
      ring
    _ = 2 * (1 - Real.sqrt 5) / (1 - 5) := by ring
    _ = 2 * (1 - Real.sqrt 5) / (-4) := by ring
    _ = (1 - Real.sqrt 5) / (-2) := by ring
    _ = (Real.sqrt 5 - 1) / 2 := by ring
    _ = (1 + Real.sqrt 5) / 2 - 1 := by ring
    _ = φ - 1 := rfl

theorem phi_inv_sq_sum : φ⁻¹ + φ⁻² = 1 := by
  calc
    φ⁻¹ + φ⁻² = φ⁻¹ * (1 + φ⁻¹) := by ring
    _ = φ⁻¹ * (1 + (φ - 1)) := by rw [phi_inv_eq_phi_minus_one]
    _ = φ⁻¹ * φ := by ring
    _ = 1 := by
      field_simp [show φ ≠ 0 from by
        nlinarith [show Real.sqrt 5 > 0 from by positivity]]

-- Verify numerically that φ ≈ 1.618...
theorem phi_approx : φ > 1.618 := by
  have h : Real.sqrt 5 > 2.236 := by
    calc
      (2.236 : ℝ) ^ 2 = 4.999696 := by norm_num
      _ < 5 := by norm_num
    nlinarith
  nlinarith

theorem phi_lt_162 : φ < 1.62 := by
  have h : Real.sqrt 5 < 2.237 := by
    calc
      (2.237 : ℝ) ^ 2 = 5.004169 := by norm_num
      _ > 5 := by norm_num
    nlinarith
  nlinarith
