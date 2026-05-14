import Mathlib

example (M : Matrix (Fin 3) (Fin 4) ℝ) : True := by
  have : ‖M‖ = ‖M‖ := rfl
  trivial
