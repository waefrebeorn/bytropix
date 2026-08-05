# ADR-002: Opaque Structs at Every Module Seam

- **Status:** accepted
- **Date:** 2026-08-05

## Context

119 files include `wubu_model.h`; 66 headers expose raw struct layouts;
changing one private field forces recompilation across the tree (and the
agent to re-read everything). The research (066, topics A1/E1/B5)
converged: hide internals at every module boundary.

## Decision

Every public API type is opaque (`typedef struct wubu_X wubu_X_t;` in the
header; `struct wubu_X { ... };` in the owning `.c`). Access to private
state goes through accessor functions. God headers are split into a
public API header + an `_internal.h` implementation header.

## Consequences

- **Positive:** compile firewall (change a private field → recompile one
  TU), ABI stability, agents read one module without the whole tree.
- **Negative:** heap/opaque handles cost an allocation at some seams —
  acceptable at module boundaries; hot inner loops keep their layout
  internal to the owning TU.
- **Compatibility:** old code that touched struct fields must migrate to
  accessors. This is a deliberate breaking change, done incrementally
  (Strangler Fig: new modules opaque, old ones migrate as touched).

## Verification

Header-touch regression test: changing one private field in a `.c` must
recompile exactly that TU (per-object `.d` tracking).
