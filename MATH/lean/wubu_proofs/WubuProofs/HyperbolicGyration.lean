import Mathlib
import WubuProofs.MobiusAdd
open Real

set_option maxHeartbeats 0

/-
HyperbolicGyration.lean — 1D gyration is identity
Ball condition: x^2 < 1/c

Uses mobius_add_1d and denom_nonzero from MobiusAdd.lean.
-/

lemma mobius_ball_preserve (c x y : ℝ) (hc : c > 0) (hx : x ^ 2 < 1 / c) (hy : y ^ 2 < 1 / c) :
    (mobius_add_1d c x y) ^ 2 < 1 / c := by
  -- Use the theorem from MobiusAdd
  exact mobius_add_preserves_ball c x y hc hx hy

lemma gyroassoc_1d (c u v w : ℝ) (hc : c > 0) (hu : u^2 < 1/c) (hv : v^2 < 1/c) (hw : w^2 < 1/c) :
    mobius_add_1d c u (mobius_add_1d c v w) = mobius_add_1d c (mobius_add_1d c u v) w := by
  have huv_sq : (mobius_add_1d c u v) ^ 2 < 1 / c := mobius_ball_preserve c u v hc hu hv
  have hvw_sq : (mobius_add_1d c v w) ^ 2 < 1 / c := mobius_ball_preserve c v w hc hv hw
  -- Define the 4 relevant denominator polynomials
  let Duv : ℝ := 1 + 2*c*u*v + c^2*u^2*v^2
  let Dvw : ℝ := 1 + 2*c*v*w + c^2*v^2*w^2
  have hDuv : Duv ≠ 0 := denom_nonzero c u v hc hu hv
  have hDvw : Dvw ≠ 0 := denom_nonzero c v w hc hv hw
  set uv := ((1 + 2*c*u*v + c*v^2)*u + (1 - c*u^2)*v) / Duv with huv_def
  set vw := ((1 + 2*c*v*w + c*w^2)*v + (1 - c*v^2)*w) / Dvw with hvw_def
  have huv_sq' : uv^2 < 1/c := huv_sq
  have hvw_sq' : vw^2 < 1/c := hvw_sq
  let D1 : ℝ := 1 + 2*c*u*vw + c^2*u^2*vw^2
  let D2 : ℝ := 1 + 2*c*uv*w + c^2*uv^2*w^2
  have hD1 : D1 ≠ 0 := denom_nonzero c u vw hc hu hvw_sq'
  have hD2 : D2 ≠ 0 := denom_nonzero c uv w hc huv_sq' hw
  have hLHS : mobius_add_1d c u (mobius_add_1d c v w) = ((1 + 2*c*u*vw + c*vw^2)*u + (1 - c*u^2)*vw) / D1 := rfl
  have hRHS : mobius_add_1d c (mobius_add_1d c u v) w = ((1 + 2*c*uv*w + c*w^2)*uv + (1 - c*uv^2)*w) / D2 := rfl
  rw [hLHS, hRHS]
  have hN1D2_eq_N2D1 : ((1 + 2*c*u*vw + c*vw^2)*u + (1 - c*u^2)*vw) * D2 = ((1 + 2*c*uv*w + c*w^2)*uv + (1 - c*uv^2)*w) * D1 := by
    dsimp [D1, D2, uv, vw]
    field_simp [hDuv, hDvw]
    ring
  exact ((div_eq_div_iff hD1 hD2).mpr hN1D2_eq_N2D1)

lemma neg_add_id (c x : ℝ) (hc : c > 0) (hx : x^2 < 1/c) : mobius_add_1d c (-x) x = 0 := by
  dsimp [mobius_add_1d]
  have den_nonzero : 1 + 2*c*(-x)*x + c^2*(-x)^2*x^2 ≠ 0 := by
    have : 1 + 2*c*(-x)*x + c^2*(-x)^2*x^2 = (1 - c*x^2)^2 := by ring
    rw [this]
    refine pow_ne_zero 2 ?_
    intro hzero
    have : c*x^2 = 1 := by nlinarith
    have : c*x^2 < 1 := by
      have : x^2 < 1/c := hx
      calc
        c*x^2 < c*(1/c) := mul_lt_mul_of_pos_left this hc
        _ = 1 := by field_simp [hc.ne.symm]
    nlinarith
  field_simp [den_nonzero]
  ring

lemma zero_add_id (c w : ℝ) : mobius_add_1d c 0 w = w := by
  dsimp [mobius_add_1d]; field_simp; ring

theorem gyration_1d_is_identity (c u v w : ℝ) (hc : c > 0) (hu : u ^ 2 < 1 / c) (hv : v ^ 2 < 1 / c) (hw : w ^ 2 < 1 / c) :
    mobius_add_1d c (-(mobius_add_1d c u v)) (mobius_add_1d c u (mobius_add_1d c v w)) = w := by
  have huv_ball : (mobius_add_1d c u v) ^ 2 < 1 / c := mobius_ball_preserve c u v hc hu hv
  have hneg_ball : (-(mobius_add_1d c u v)) ^ 2 < 1 / c := by simpa using huv_ball
  calc
    mobius_add_1d c (-(mobius_add_1d c u v)) (mobius_add_1d c u (mobius_add_1d c v w))
        = mobius_add_1d c (-(mobius_add_1d c u v)) (mobius_add_1d c (mobius_add_1d c u v) w) := by
          rw [gyroassoc_1d c u v w hc hu hv hw]
    _ = mobius_add_1d c (mobius_add_1d c (-(mobius_add_1d c u v)) (mobius_add_1d c u v)) w := by
      rw [gyroassoc_1d c (-(mobius_add_1d c u v)) (mobius_add_1d c u v) w hc hneg_ball huv_ball hw]
    _ = mobius_add_1d c 0 w := by rw [neg_add_id c (mobius_add_1d c u v) hc huv_ball]
    _ = w := zero_add_id c w
