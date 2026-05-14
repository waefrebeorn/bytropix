/-
  Main.lean — Entry point for the lean-mathlib4 project.
  Imports LeanCopies.lean and verifies the four theorems compile.
-/

import LeanCopies

open Real

/-- Print a brief verification message. -/
def main : IO Unit :=
  IO.println "All four theorems loaded successfully. Use `lake build` to verify."

/-- Sanity check: Theorem 4's proof is the most complete; run a quick structural test. -/
#check poincare_ball_identity
#check mobius_add_closure
#check mla_kv_compression_error_bound
#check gyration_preserves_ball
