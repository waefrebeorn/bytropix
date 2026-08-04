# wubuwizard

C11 inference engine ("the Colonel") — **the BRAIN of the WuBu AGI**. No
external ML dependencies; quantization, matmul, tokenizer, and model loaders
are implemented in-tree. The Body (`wubuos`) hosts this engine on metal via the
Live Colonel. Doctrine: the standing loop
`corpus → train → diagnose → mutate → validate → archive → RLHF oracle → repeat`
([TOPOLOGY](docs/TOPOLOGY.md), [ARCHITECTURE](docs/ARCHITECTURE.md),
[BUILDING](docs/BUILDING.md)).

## Layout

| Path | Contents |
|------|----------|
| `src/` | Core engine: model loaders, SSM/GQA/MLA/MoE forward, quant matmul, GGUF + safetensors readers, tensor store, CUDA kernels (305 C + 21 CUDA). |
| `include/` | Public headers (opaque structs, minimal includes) — 302 headers. |
| `tools/` | `gen_text` (CPU generation), 348 test tools, CLIs, API server, `repodoc/` doc generator. |
| `research/` | The gap ledger: `INDEX.md` (AN01-AN11) + 45 numbered notes, each with Triple-DA + `wired` status. |
| `THEORY/` `MATH/` | Our papers + Lean-verified proofs (Poincaré ball, Möbius, gyration, MLA compression). |
| `docs/` | The docs: [TOPOLOGY](docs/TOPOLOGY.md), [ARCHITECTURE](docs/ARCHITECTURE.md), [BUILDING](docs/BUILDING.md), [MODULES](docs/MODULES.md), model blueprint + card. |
| `models/wubu/` | The WuBu-35M seed (weights untracked in git — canonical on HF). |

## Build

```bash
make gen_text          # CPU inference binary
make wubu_train        # the trainer
make test_all          # full test gate (299 targets)
make api_server        # OpenAI-compatible HTTP server
```

C11, GCC/Clang. CUDA paths require `nvcc` (optional). Full details:
[docs/BUILDING.md](docs/BUILDING.md). Module table: [docs/MODULES.md](docs/MODULES.md).

## Model support

Loaders: safetensors, GGUF (incl. TurboQuant Q2_0/TQ3_1S/TQ4_1S + multi-split),
`.st` dumps — all through the catalog doctrine (`wubu_tensor_store.c`: a format
is a catalog over the same bytes). Per-model status is in [STATUS.md](STATUS.md).

### Supported architectures

| Architecture | What it is | Key files |
|---|---|---|
| **Gated-DeltaNet hybrid** (SSM + GQA + MLP) | Qwen3.x / Agents-A1; per-layer `layer_types` | `src/wubu_ssm.c`, `src/wubu_model_safetensors_bridge.c` |
| **Qwen3.x MoE** (256 experts) | Deep MoE + shared expert; SSD-paged | `src/wubu_moe.c`, `src/wubu_ssd_moe.c` |
| **MLA + MoE + mHC + DSA** (DeepSeek-V4-Flash Config-I) | 43L, 284.3B, 256×top-6, hash router (blk 0-2), mHC, DSA indexer, KV compressor — **load gate PASSED**; forward next | `src/wubu_mla.c`, `wubu_moe*.c`, `wubu_hashrouter.c`, `wubu_dsa.c`, `wubu_mhc_mh.c` |
| **Dense hybrid** (Qwen3.6-27B) | 64-layer dense, SSM+GQA per layer | `src/wubu_ssm.c` |
| **LoRA adapters** | BTL-3 rank-32 alpha-64 on Qwen3.6-27B base | `src/wubu_lora.c` |

### Model status matrix

| Model | Form | Status |
|---|---|---|
| **WuBu-35M (the seed)** | safetensors / .st | training loop runs (SFT 8.04→7.32); mixed export 5.12× |
| **Agents-A1-4B** | safetensors (SD) | real SSM forward verified ✅ |
| **KAT-Coder-V2.5-Dev** | safetensors (SD) | 13/13 shards; SSD slot-bank works |
| **Qwen3.6-27B** | safetensors (SD) | adapter derives dims; SSM forward finite ✅ |
| **Qwen3.6-35B-A3B-UD-IQ2_M** | GGUF (SSD) | 753 tensors dissected; 12 types NaN-free |
| **DeepSeek-V4-Flash-0731 Config-I** | GGUF 3-split (SSD, downloading) | load gate PASSED (1328 tensors, 7 types, 0 mismatches) |

> Weights policy: GGUF lives on the SSD; safetensors model dirs are COLD
> STORAGE on the SD card (`/home/wubu/sdcard/models/` — mount `D:` first:
> `sudo mount -t drvfs D: /home/wubu/sdcard`). All tensor loading goes through
> the bridge + tensor store; the lazy BF16 path keeps 27B-class models under
> 13 GB RAM.

## Subsystems

- **Tensor store** (`wubu_tensor_store.c`): catalog interchange — .st ↔
  safetensors ↔ GGUF by streaming; mixed per-role export (Q8_0/Q4_0/IQ2_XXS/F32).
- **Loaders**: safetensors, GGUF (TurboQuant dequants + multi-split resolve),
  HF tokenizers — `test_gguf_load` is the load gate.
- **SSM forward** (`wubu_ssm.c`): Gated DeltaNet recurrence, conv1d, GQA, QK-norm.
- **MoE** (`wubu_moe.c`): router, shared expert, quantized experts; SSD-paged
  variant (`wubu_ssd_moe.c`) backed by the slot-bank. Hash routing:
  `wubu_hashrouter.c` (token/pos/salt hashing, TID→EID tables).
- **MLA** (`wubu_mla.c`): latent-KV multi-head attention (deepseek4).
- **mHC** (`wubu_mhc.c` + `wubu_mhc_mh.c`): manifold-constrained hyper-connections,
  multi-head form (2512.24880) verified maxdiff=0.
- **DSA** (`wubu_dsa.c`): coarse-to-fine indexer (the lightning indexer).
- **Quantization**: Q4_K/Q5_K/Q6_K/IQ2_XXS/IQ3_XXS/IQ4_XS/IQ2_S/IQ1_S/IQ1_M +
  Q8_0 + TurboQuant Q2_0/TQ3_1S/TQ4_1S, self-hosted `vec_dot` (no libggml).
- **The AGI organs**: `wubu_amoeba` + `wubu_hive` (the diagnostic hive),
  `wubu_prover2` (verifier), `wubu_agi` (the loop), `wubu_dgm` (the gate).
- **Training**: `wubu_train` + `wubu_backprop` (real backward + Muon),
  rolling checkpoints (`tools/wubu_ckpt_roll.py`).
- **Sampling**: temp/top_p, repetition + DRY penalties.

## Vault

The `THEORY/` and `vault/` trees hold the underlying math: Poincaré/hyperbolic
embeddings, GAAD (Golden Aspect Adaptive Decomposition), DFT/DCT spectral
encoders, the tailslayer hedged-read system, and the DeepSeek MoE/sparse-attention
paper set. `THEORY/math_viz/` contains runnable numerical proofs.

<!-- repodoc:BEGIN -->
## Module index (auto-generated 2026-08-04)

- **305 C modules** — full annotated table: [docs/MODULES.md](docs/MODULES.md)
- **348 test tools** (make targets `test_*`, e.g. `test_200, test_256k, test_256k_chunked, test_256k_context, test_256k_forward, test_300, test_400, test_4kv, test_512k_budget, test_adaptive_hotpath...`)
- **45 research docs** — full ledger: [research/INDEX.md](research/INDEX.md)

Regenerate with: `python3 tools/repodoc/repodoc.py . --readme`
<!-- repodoc:END -->
