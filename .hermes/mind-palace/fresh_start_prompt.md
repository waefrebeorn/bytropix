═══ WUBUTEXT AI — FRESH START PROMPT (May 15 PM v6 — COMPREHENSIVE) ═══

You are starting a session on the WuBuText AI project.

## Quick Context

**State:** All phases complete. Training at 11s/step (16× improvement). Zero NaN across all configs.

**Read first:** `.hermes/mind-palace/goal-mantra.md` (v6) — single paste for full prestige resume.
**Plan:** `.hermes/mind-palace/plan.md` — full priority queue with vault, paper findings, tailslayer.
**State:** `.hermes/mind-palace/state.md` — binary dashboard, metrics, known issues.

## Build & Run

```bash
cd /home/wubu/bytropix
make train_integrated
./train_integrated /home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf data/train_data.bin 3
```

## Key Facts

- **gguf_raw_size(IQ2_XXS) = 66** (was wrong: 72). Fixed empirically.
- **Per-expert dequant:** raw + eid × gguf_raw_size(type, per_exp_elems) → temp[ff][model] → transpose to ge[model][ff]
- **Training:** 177s→11s/step, CE 21.6→18.4, 0 NaN
- **GPU output projection:** cublasSgemm with CUBLAS_OP_T for output_weight[V,D_MODEL]^T
- **Async D→H copies:** saved_normed/attn_out skipped when !pga_enabled
- **NaN fix:** MoE weight interleaving (GGUF IQ2_XXS block layout) + raw_size bug

## Known Issues

- PGA loss jumps 21.6→69 (LR too high for PGA backward)
- ~11s/step is GPU compute bound (40 layers SSM/GQA on RTX 5050)
- CONV_DIM=8192 vs config 1536 (needs investigation)
- MRoPE 3D not implemented (position encoding degrades at >32K)
- MTP prediction head missing

## Tailslayer (NEW May 15)
Analyzed LaurieWired/tailslayer — hedged reads across DRAM channels.
Direct pattern match: N replicas → N draft tokens, first-response-wins → longest valid prefix accept.
Full findings at `.hermes/vault/tailslayer/README.md`

## Vault Status (13 entries)
- **P2 high ROI:** Sparse attention (PyTorch), Tailslayer (C++), Q-Controller/PID (JAX), Hamilton (CUDA)
- **Research:** encoders, diffusion, phase3, audio, draftPY, lean-proofs
- Full list at `.hermes/mind-palace/index.md`

## Paper Audit (May 15)
32 Qwen3.6 paper files cross-referenced. 14 config params checked.
9 ✅ match, 2 Verify, 2 ❌ missing (MRoPE, MTP), 1 ❌ discrepancy (CONV_DIM 8192 vs 1536).
Full table at `.hermes/mind-palace/state.md`

## Mind Palace

Read in this order:
1. `.hermes/mind-palace/goal-mantra.md` — Prestige paste
2. `.hermes/mind-palace/state.md` — Binary dashboard + paper audit + tailslayer
3. `.hermes/mind-palace/plan.md` — Priorities
4. `.hermes/mind-palace/entry.md` — Build commands
5. `README.md` — Full project overview
