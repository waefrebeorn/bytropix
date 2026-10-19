# MODEL IDENTITY — the body, not a parameter count

Status: canonical. Where this page and any other doc disagree, this page wins and
the other doc is the bug. Verified 2026-10-18 against the code and its tests.

## The rule

**WuBu is a runtime-dimensioned amoeba body. It is never named by a parameter
count.** "35M" is the size of one *geometry* the body has worn, not the model.

The design doctrine (`wubu-design-philosophy`, THEORY/06 §"No fixed geometry")
is explicit: dims are runtime (`WUBU_DIMS`), and a `#define` describing model
shape is a violation (audit item S1). A model whose identity is a constant
cannot grow, shrink, or rotate — which is the whole point of the body.

## The two geometries (both real, both measured)

| geometry | parameters | where the number comes from |
|---|---:|---|
| **checkpoint-native** (the legacy `WuBu-35M` release) | **35,073,216** | `tools/test_runtime_dims.c:75-77` prints `params=%ld (aligned 56376832, ckpt native 35073216)` |
| **aligned** (current default) | **56,376,832** | same test, asserted: `CHECK(params == 56376832L, ...)` |

Aligned geometry is `dim 512 · ffn 2048 · 12 layers · 8 heads · kv_heads 1 ·
head_dim 64 · vocab 16384 · 3 selectors`, tiling evenly into `QK_K=256` (no
zero-fill guards fire) and dividing 64 (AVX-512 cache line) and 16 (VNNI tile).

`wubu_parameter_count()` is a **pure function of the runtime geometry** — it
reads `WUBU_RUNTIME_DIMS`, not the `#define`s. Two different layers isolated from
the same body therefore report different totals, correctly.

## What 35M actually was

`WuBu-35M` is the **checkpoint-native** geometry: the tensor layout of the first
published checkpoint (`safetensors` + `.st`). The README of that release, the
model card and several docs treat that count as the model's name. That is the
inaccuracy: it froze one instance of a movable body and made the frozen number
the identity.

It remains a legitimate *artefact* — a released checkpoint with a known geometry,
loading 35,073,216 parameters. It is not the architecture, and it is not the
current default.

## What the docs must say instead

- Name: **WuBu** (the body). Qualify by geometry when a number is needed:
  "WuBu, aligned geometry (56,376,832)" or "the checkpoint-native geometry
  (35,073,216)".
- Never as a headline claim: "WuBu is a 35M model."
- Fresh training uses aligned geometry (`WUBU_RUNTIME_DIMS` defaults) and can
  rotate under the amoeba loop without renaming itself.

## Known code debt (tracked, not fixed here)

`include/wubu.h` still carries prototype constants:

```
#define WUBU_DIM      448
#define WUBU_LAYERS   12
#define WUBU_FFN_DIM  1228
```

They describe the retired prototype geometry and **disagree with the runtime
defaults (512 / 2048)**. Nothing that calls `wubu_parameter_count()` or the
runtime-dims path uses them for model shape, but they remain a trap: a reader —
or a new file — can take them as truth. Fixing them means auditing every
`#if`/array bound that still uses them, which is why they are flagged rather than
silently changed.

## Where the truth lives

- `include/wubu_dims.h` / `WUBU_RUNTIME_DIMS` — the runtime geometry
- `src/wubu.c:806 wubu_parameter_count()` — the count, derived not asserted
- `tools/test_runtime_dims.c` — the two numbers above, asserted
- `docs/MANIFOLD-TRAINING.md` — how the body trains at whatever geometry it wears
- `wubu-design-philosophy` skill — the doctrine (geometry IS the architecture)
