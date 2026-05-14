import Lake
open Lake DSL

package lean

require mathlib4 from git
  "https://github.com/leanprover-community/mathlib4.git"

lean_lib LeanCopies

@[default_target]
lean_exe lean where
  root := `Main
