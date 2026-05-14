import Mathlib
open Real

set_option maxHeartbeats 0

/-
PoincareBall.lean — exp_0^c(log_0^c(y)) = y in 1D Poincaré ball.
-/

theorem poincare_exp_log_identity_nonzero (c : ℝ) (hc : c > 0) (y : ℝ) (hy0 : y ≠ 0)
    (hy : |y| < 1 / Real.sqrt c) :
    (Real.tanh (Real.artanh (Real.sqrt c * |y|)) *
     (Real.artanh (Real.sqrt c * |y|) * y / (Real.sqrt c * |y|)) /
     (Real.sqrt c * |Real.artanh (Real.sqrt c * |y|) * y / (Real.sqrt c * |y|)|)) = y := by
  have hs : Real.sqrt c > 0 := Real.sqrt_pos.mpr hc
  have hpos : Real.sqrt c * |y| > 0 := mul_pos hs (abs_pos.mpr hy0)
  have h_lt_one : Real.sqrt c * |y| < 1 := by
    calc
      Real.sqrt c * |y| < Real.sqrt c * (1 / Real.sqrt c) := mul_lt_mul_of_pos_left hy hs
      _ = 1 := by field_simp [ne_of_gt hs]
  have ha_pos : Real.artanh (Real.sqrt c * |y|) > 0 :=
    Real.artanh_pos (Set.mem_Ioo.mpr ⟨by positivity, h_lt_one⟩)
  have h_abs_v : |Real.artanh (Real.sqrt c * |y|) * y / (Real.sqrt c * |y|)|
    = Real.artanh (Real.sqrt c * |y|) / Real.sqrt c := by
    rw [abs_div, abs_mul, abs_of_pos ha_pos, abs_of_pos hpos]
    field_simp [hpos.ne.symm]
  have h_tanh : Real.tanh (Real.artanh (Real.sqrt c * |y|)) = Real.sqrt c * |y| :=
    Real.tanh_artanh (Set.mem_Ioo.mpr ⟨by nlinarith, h_lt_one⟩)
  -- Now compute the main expression
  rw [h_tanh, h_abs_v]
  field_simp [hs.ne.symm, hpos.ne.symm, ha_pos.ne.symm]
