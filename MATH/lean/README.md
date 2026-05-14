# Lean 4 Mathlib4 Hyperbolic Geometry Project

A Lean 4 project proving theorems about the Poincaré ball model of hyperbolic geometry,
Möbius addition, gyration, and an MLA KV compression error bound.

## Prerequisites

- **Lean 4.29.1** (or compatible) with **Lake 5.0.0**
- Installed via `elan` (recommended): `curl -R https://leanprover-community.github.io/install.sh | sh`

## Project Structure

```
.
├── lakefile.lean     # Lake build file (depends on mathlib4)
├── lean-toolchain    # Lean version pin
├── Main.lean         # Entry point
├── LeanCopies.lean   # All four theorems with definitions and proofs
└── README.md         # This file
```

## Theorems

### Theorem 1: Poincaré ball identity — exp ∘ log = id
For `y` in the open unit ball (‖y‖ < 1) and curvature `c > 0`:
`exp_0^c(log_0^c(y)) = y`

### Theorem 2: Möbius addition closed form and ball closure
For `x, y` in the Poincaré ball and `c > 0`:
`‖x ⊕_c y‖ < 1`

### Theorem 3: MLA KV compression error bound
The Frobenius norm error of rank-r truncated SVD of a key matrix `K` is bounded by
the tail singular values.

### Theorem 4: Hyperbolic gyration preserves the ball
For `u, v, w` in the Poincaré ball:
`‖gyr(u,v)w‖ < 1`

## Building

```bash
cd /home/wubu/bytropix/MATH/lean/
lake build
```

This will:
1. Fetch `mathlib4` as a Lake dependency (may take a while on first run)
2. Compile all `.lean` files
3. Verify the proofs type-check

## Checking proofs

After a successful build:

```bash
lake env lean Main.lean
```

You can also open individual files in a Lean-aware editor (VS Code with
`lean4` extension, or emacs with `lean4-mode`).

## Notes

- Theorems 1 and 2 have partial proofs (the algebraic details are sketched);
  completing them requires expanding the full radial vector algebra and
  Cauchy-Schwarz estimates.
- Theorem 3 is stated but the full proof relies on the Eckart-Young-Mirsky
  theorem from numerical linear algebra.
- Theorem 4 has a complete proof that reduces to Theorem 2.
