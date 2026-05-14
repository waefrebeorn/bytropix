import Mathlib
open Real

set_option maxHeartbeats 0

/-
MLACompression.lean — KV Compression Factor Identity

H·W_K - H·W_DKV·U_K = H·(W_K - W_DKV·U_K)

The Frobenius norm submultiplicativity (‖AB‖ ≤ ‖A‖·‖B‖) is a
standard result for the Euclidean/Frobenius norm on matrices,
and mathlib4 provides this via norm_mul_le on SemiNormedRing
for square matrices under the operator norm.

For non-square matrices, the Frobenius norm also satisfies
this inequality, but mathlib4 restricts norm instances to
square matrices. We prove the algebraic factor identity here.
-/

theorem mla_compression_factor {T D d_c K : ℕ}
    (H_states : Matrix (Fin T) (Fin D) ℝ)
    (W_DKV : Matrix (Fin D) (Fin d_c) ℝ)
    (U_K : Matrix (Fin d_c) (Fin K) ℝ)
    (W_K_full : Matrix (Fin D) (Fin K) ℝ) :
    H_states * W_K_full - H_states * W_DKV * U_K = H_states * (W_K_full - W_DKV * U_K) := by
  calc
    H_states * W_K_full - H_states * W_DKV * U_K
        = H_states * W_K_full - H_states * (W_DKV * U_K) := by simp [Matrix.mul_assoc]
    _ = H_states * (W_K_full - W_DKV * U_K) := by rw [Matrix.mul_sub]
