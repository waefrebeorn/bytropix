# 046 — Heterogeneous precision: the Escha axis (bit-width is an amoeba parameter)

> Status: `open`. Date: 2026-08-03.
> Source: llm.ciru.ai/research/escha-vs-35b/ (Ciru Inference Lab, 2026-08-03) —
> "2-bit Escha W2 vs Released 35B Models".

## The result (quality density)

A 2-bit-class Qwen3.6-35B-A3B MoE ("Escha W2") at **12.3 GB of weights** —
35% smaller than the next-smallest 19.0 GB entry, ~1/3 of the 36.9 GB Q8 —
posts the **best released-model HermesAgent-20 score (90/100, rank #1 of 12)**
and stays within:
- 2.4 pp of the best HumanEval+ (90.9% vs 93.3% best)
- 1.9 pp of the best MBPP+ (75.7% vs 77.5% best)
- 2.0 pp of the best BigCodeBench Hard (29.7% vs 31.8% best)
- Tool Eval 87/100 standard, 80/100 hard-15

## The mechanism (why this matters to the amoeba)

The format is **NOT one uniform bit-width** — it is **heterogeneous per matrix family**:
`2b gate/up · 3b down · INT8 dense`.

Why that split works:
- **gate/up**: outputs are filtered downstream (router / SwiGLU gate), so 2 bits are
  tolerable — small errors don't compound into the residual stream.
- **down**: writes back into the residual stream, needs 3 bits to stay accurate.
- **dense** (q/k/v/o, embedding, and any non-MoE weight): INT8 — full precision budget.

This is exactly the amoeba doctrine at the precision level: **measure which parts are
sensitive, spend bits where they matter, starve the rest**. And it is what makes a
35B-class model fit a 12.3 GB budget — i.e., **"run on all hardware"** is a precision
plan, not a hardware feature.

## What WuBu already has (the raw material)

- `src/wubu_gemv_tune.c` — cpuid AVX512/AVX2 detection + tile selection (the hardware
  profile seed).
- `src/wubu_quant_selector.c` — batch-aware quant switch, ctx-length precision ladder,
  PMC roofline (the adaptive-quant seed).
- `quantized_matmul.c`, `quantized_dot_generic.c`, `wubu_smoothquant.c`, `wubu_awq.c`,
  `wubu_turboquant.c`, IQ2 grids (`iq2xxs_grid_data.inc`, `dequant_iq2_xxs.c`) — the
  low-bit GEMV machinery.
- `src/wubu_ewc.c` — Fisher importance (the priority-intelligence seed; 15-dim sweep
  space today, needs per-block/per-family extension; logits-reversal already applied
  per arXiv 2603.18596).
- `src/wubu_awq.c` — activation-aware 1% salient-channel protection (which dense
  channels survive low bits).

## The plan mapping (Phase 4 of the amoeba master plan)

1. **`wubu_precision_plan`** — per-family bit table: gate/up 2b, down 3b, dense INT8,
   norms/selectors fp32, embedding INT8. Escha defaults, overridable via config.json.
2. **`wubu_hw_profile`** — detect CPU SIMD / RAM / GPU (/dev/dxg, CUDA) / storage once;
   ladder = pure function of profile → every device gets a working precision
   (big box: full Escha; small box: dense→INT4 via AWQ; no SIMD: fp32 fallback).
3. **per-family GEMV dispatch** — INT8 block-32 (exists), 2-bit IQ2-class pack,
   3-bit pack; FD-checked against fp32 matmul per family tolerance.
4. **quality-density gate** — a precision plan is accepted only if quality/byte beats
   the current plan (the Escha result reproduced at 35M scale).

## The priority store (Phase 5)

The amoeba REMEMBERS: per-block BI, per-family Fisher, per-family precision deltas
(which matrices survived 2b, which needed 3b), the mutation ledger — persisted as a
safetensors sidecar (`priority.safetensors`). The immune system consults it before
the next mutation: never re-shrink a layer that proved critical, never 2-bit a family
whose 3-bit was empirically required. That is "stores priority intelligence as an AGI".

## The honest caveat

Escha's win is measured at 35B-A3B MoE scale on coding/tool suites. At 35M dense the
absolute headroom is smaller — but the mechanism (per-family sensitivity → bit
allocation) is scale-free, and the density gate (4.4) decides honestly. The dense
embedding at 16K vocab × 448 = 21% of our params is the first family to fight over.

## Test oracles

1. precision plan round-trips; family overrides don't leak.
2. hw profile on this box (AVX512?, /dev/dxg GPU, 6GB) → ladder fits the device.
3. packed GEMV ≈ fp32 within family tolerance (2b 1e-1, 3b 5e-2, INT8 1e-2).
4. Escha plan beats uniform INT8 on quality-per-byte (density ↑).
5. priority store round-trips; a rejected shrink marks the layer protected; the next
   diagnose never re-proposes it.
