# 044 — The Amoeba Full-Body: all parts grow AND shrink

> Status: `open` (plan at `.hermes/plans/2026-08-03-amoeba-grow-shrink-expanded-tokenization.md`)
> Date: 2026-08-03. Sources: ShortGPT (arXiv:2403.03853), LaCo (arXiv:2402.11187),
> Net2Net (Chen et al. 2015), Gate-Zero Growth (arXiv:2607.14571).

## The problem

WuBu's model body only GROWS. `wubu_grow_insert_block` (zero-insert) and
`wubu_grow_stack_block` (G_stack copy) shift blocks up; `wubu_train_grow`
shifts the grad/momentum arrays; `wubu_grow_schedule` is monotonic
(Zhiqi Bu). There is NO shrink operator, NO `wubu_train_shrink`, and the
amoeba module (`wubu_amoeba`) only recycles *expert hive slots* — it can
never shrink the model body itself. The amoeba is one-directional: it can
extend a pseudopod but cannot retract it.

## The fix: symmetric operators, every part

| Part | GROW (function-preserving) | SHRINK (fitness-gated) | Train-state pair |
|---|---|---|---|
| depth (layers) | zero-insert / G_stack copy (exists) | **ShortGPT BI-score removal** (NEW) / **LaCo merge** (NEW) | `wubu_train_grow` (exists) / **`wubu_train_shrink`** (NEW) |
| width (dim) | Net2Net row duplication | low-norm column prune | realloc all train arrays |
| FFN dim | Net2Net gate_up/down | low-norm prune | realloc |
| vocab | embedding row add | corpus-count prune (tied head) | — |
| selectors | zero score-vector add | least-used drop | — |
| block slots | freelist pop (hive) | freelist push (hive) | — |

## The DA oracles

1. **GROW** must be function-preserving: forward-before == forward-after
   (tolerance 1e-6). Zero-init residual branch = exact identity.
2. **SHRINK** is NOT function-preserving — its oracle is the amoeba
   fitness gate (held-out loss within `loss_tol`, prover passes) + the
   BI-score-informed layer choice + the FD backward check.
3. **Train state follows**: every model op has a matching `wubu_train_*`
   op (reverse SHIFT_ARR for shrink). ASan-clean free at teardown.

## Key numbers / facts

- ShortGPT: removing layers by BI score (mean hidden-state-norm change
  per layer) removes ~25% of layers with minimal degradation; shallow
  layers are more important than deep ones (pre-norm models).
- LaCo: merges the most similar adjacent layers deep→shallow, thresholded.
- Net2Net: widen by copying neurons (replication factor 1/2 on fan-in);
  deepen by identity insertion. Gate-zero growth: zero-init gate makes
  residual insertion exactly function-preserving.
- BLT's scaling axis (§ the sister doc 045): if the body is shallower,
  patch size compensates — depth is no longer the only lever.

## Implementation plan (from the master plan)

1.5 dim-runtime refactor FIRST (every buffer is `#define`-sized today) →
1.1 `wubu_shrink_remove_block` → 1.2 `wubu_train_shrink` →
1.3 BI-score oracle → 1.4 LaCo merge → 1.9 hive-backed slots →
1.6/1.7/1.8 width/FFN/vocab/selectors → amoeba integration.

## Test oracle (the gate)

`make test_grow test_amoeba test_width test_backprop` all green, ASan
clean; grow 2→5 then shrink 5→2 round-trip: forward finite, live count
correct, teardown clean.
