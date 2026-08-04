# research/055 — The Architecture Matrix: 7-hop through OUR OWN project

> 2026-08-04. The user's directive: "we were missing architecture in our
> wizard. We need to add all missing architectures. Do a seven-step Kevin
> Bacon throughout all of our project to see what we already have said and
> planned, re-organize, get oriented, and start making improvements while
> the training runs. We are doing the basics right now — we need to be more
> advanced. We are the hive-mind absorption of all of everything."

## The method

Seven hops through OUR OWN corpus (code + research + docs + skills) instead
of the web — the project's own memory IS the literature to converge:

1. **src/ + include/ sweep** — what architectures EXIST in the engine (grep
   markers across 277 modules).
2. **STATUS.md + README model matrix** — what the bridge actually loads.
3. **research/INDEX.md themes WB/BL/A-AA + AM** — what we claim `wired`.
4. **docs/WUBUWIZARD_100_* + OPTIMIZATIONS_100** — what we PLANNED (the
   un-built frontier).
5. **research/041-054 + THEORY/** — the training/arch doctrine + the
   hyperbolic family.
6. **skills** — wubuwizard-inference / multimodel-c11 / design-philosophy /
   model-zoo: what the procedural layer remembers.
7. **the big-brother zoo** (the model-zoo ladder) — the arch each tier
   demands (deepseek4 = the KAHUNA; Escha = precision; Qwen3.6 = hybrid).

## THE MATRIX — have / said / lack

### HAVE (in code, `wired`)
| Architecture | Module | Notes |
|---|---|---|
| Linear-attn family (Mamba-2 / RetNet / HGRN2 / GLA / Mamba) | `wubu_linear_attn.c` | S-state update fns, FD-verified |
| Gated-DeltaNet (3:1 hybrid) | `wubu_delta_net.c` | oracle-matched (research/008) |
| MLA (multi-head latent attention) | `wubu_mla.c`, `wubu_kda.c` | DeepSeek-family KV compression |
| MEGA | `wubu_mega.c` | exponential-gated chunking |
| Titans (neuro working memory) | `wubu_wm_kv.c` | bounded ring |
| MoE (256-exp) / grouped / expert-choice / LatentMoE | `wubu_moe*.c` | KAT-Coder real-weights |
| MoBA / NSA / sparse-attn / H2O | `wubu_sparse_attn.c` + | the sparse-attention cousins of DSA |
| BitNet/ternary | `wubu_ternary.c`, `wubu_dn2.c` | ±1/0 GEMV |
| Cross-layer residual (AttnRes) | `wubu_attnres.c` | cross-layer residual read/write |
| **mHC hyper-connections** | `wubu_mhc.c` + `test_mhc` | **EXISTS + wired** (Round-3, sigmoid-constrained non-neg mixing, identity oracle passes) — verified 2026-08-04 |
| Hyperbolic: nest / poincare / mobius / hopfield | `wubu_nest*`, `wubu_poincare*`, `wubu_mobius*`, `wubu_hopfield*` | OUR unique sauce, Lean-verified |
| Gemma4 | `wubu_gemma4_model.c`, `gpu_gemma4.cu` | partial |
| Qwen3.x hybrid (SSM+GQA) / KAT MoE / LoRA | bridge + `wubu_lora.c` | the loaded zoo |
| KV: Q8/KIVI/Ecco/4KV/TurboQuant/PolarQuant/Q8K-PQV | `wubu_kv*.c` | research/001-015 |

### SAID (planned in docs, NOT built)
| Planned | Where said | Status |
|---|---|---|
| **Hash-based expert routing** (DeepSeek V3.2/V4) | WUBUWIZARD_100_MORE_MORE (DeepSeek-V4 hub) | **0 files — BUILDING (this wave)** |
| **DSA — DeepSeek Sparse Attention indexer** | same doc | **0 files — BUILDING (this wave)** |
| Gemma 4 CLA (cross-layer attention) + PLE (per-layer embeddings) | same doc | 0 files (AttnRes is the sibling) |
| Mamba-2 fused matmul+scan (#52) | WUBUWIZARD_100_INFERENCE | linear_attn has the update; fused kernel missing |
| MIMO SSM / Mamba-3 (#54) + hybrid cadence (#56) | same | 0 files |
| RWKV-7 | — | 0 files |
| AM01 full-body grow+shrink (ShortGPT/LaCo/Net2Net) | research/044, INDEX AM01 | `open` |
| AM02 BLT expanded tokenization | research/045, INDEX AM02 | `open` |
| AM03 Escha heterogeneous precision (wubu_precision_plan/hw_profile) | research/046, INDEX AM03 | `open` |
| dim-runtime refactor (`m->dim` instead of `#define`) | research/044 (highest-risk) | `open` |

### LACK (the deepseek4 KAHUNA requirement, all 0 files before this wave)
1. **mHC** — the V3.2/V4 residual upgrade (group of hidden states + learned
   manifold-constrained mixing + gated write). AttnRes is the sibling —
   mHC extends it to multiple parallel streams.
2. **Hash router** — learner-free deterministic expert assignment
   (the KAHUNA's 256-exp top-6 uses hash routing, no load-balance loss).
3. **DSA indexer** — coarse-to-fine block index → sparse attention
   (we have NSA/MoBA; DSA is the DeepSeek-specific indexer head).

## Convergence statement

The wizard already absorbed the 2025 attention/KV/quant frontier (themes
A-AA ~241 wired). The 2026 frontier the zoo demands is the **architecture
layer**: DeepSeek's residual + routing + sparse-index upgrades (mHC, hash
router, DSA) and the linear/SSM next-gen (Mamba-2 fused, MIMO, RWKV-7).
The hive-mind absorption = port each big-brother's NEW mechanism into our
own C11, verified by oracle, then wire it into the amoeba's morph axes.

## This wave (parallel, 2026-08-04)

| Module | Mechanism | Oracle |
|---|---|---|
| `wubu_mhc.c` | multi-head hyper-connections | identity init == plain residual (1e-6) |
| `wubu_hashrouter.c` | deterministic hash expert routing | determinism + balance (2.5x uniform) |
| `wubu_dsa.c` | DSA coarse-to-fine block indexer | top-k exact + dominant-block fidelity (1e-2) |

Each: standalone C11 + `test_<x>` + CORE_OBJ + INDEX flip (`wired`).

## Next wave candidates (the un-built frontier, in priority order)

1. **RWKV-7** (`wubu_rwkv7.c`) — the state-tracking linear RNN (0 files).
2. **MIMO SSM / Mamba-3** (`wubu_mimo.c`) — multimodal state (planned #54).
3. **Gemma CLA + PLE** (`wubu_cla2.c`, `wubu_ple.c`) — cross-layer attn +
   per-layer embeddings (planned in the 100-MORE-MORE doc).
4. **Mamba-2 fused matmul+scan** — the #52 kernel (have the update fn).
5. **AM01 depth shrink** (`wubu_shrink.c`) — ShortGPT BI removal + LaCo
   merge — the amoeba's missing SHRINK direction.
6. **AM03 precision plan** (`wubu_precision_plan.c`) — the Escha ladder.

## Registration

- INDEX theme **AN** (this doc): AN01 mHC / AN02 hash router / AN03 DSA
  → `wired` when the tests land; AN04 RWKV-7 / AN05 MIMO / AN06 CLA+PLE
  → `open` (next wave).
