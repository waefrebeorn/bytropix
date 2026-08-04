# wubuwizard Architecture — the inference engine that is the backbone of an AGI

> 2026-08-04. This doc is the ENGINE's spine: what the modules do, how data
> flows, and what each layer guarantees. It is the companion to
> [TOPOLOGY.md](TOPOLOGY.md) (the cross-repo master map) and
> [MODULES.md](MODULES.md) (the machine-generated module table). The Brain
> learns; the Body (`wubuos`) protects and acts. The Live Colonel (ring-0 REPL
> in wubuos) hosts this engine on metal.

## The one-sentence idea

wubuwizard is a from-scratch C11 inference + training engine (no libggml, no
Triton, no third-party BLAS) whose purpose is the WuBu-35M AGI: a colony of
specialized experts on nested spheres, watched by a diagnostic hive that grows
overworked cells, shrinks dead ones, and archives every validated mutation
(Darwin Gödel Machine: mutate → validate → archive, sandboxed).

## The data flow (what happens when you generate a token)

```
prompt ─► tokenizer (HF BPE / GPT-2 BPE) ─► token ids
  ─► embedding (Q8_0) ─► N × transformer block ─► final norm ─► head ─► logits
  ─► sampler (temp / top_p / DRY anti-repetition) ─► next token ─► KV cache
```

Per block: `norm → attention (GQA | MLA | SSM-Gated-DeltaNet) → residual →
norm → MoE FFN (router → top-k experts → shared expert) → residual`, plus the
optional agentic organs (hash router, DSA indexer, mHC hyper-connections,
KV compressor/sinks) when the architecture provides them.

## The layers

| Layer | Modules (representative) | Guarantee |
|---|---|---|
| **Loaders** | `wubu_model_safetensors_bridge.c`, `gguf_reader.c`, `safetensors_reader.c` | any checkpoint (safetensors / GGUF / .st dump) opens as a catalog of (name, offset, dtype, shape); multi-split GGUF (3-part Config-I) resolved |
| **Tensor store** | `wubu_tensor_store.c` | the catalog doctrine: a format is a catalog over the same bytes — export/import between .st, safetensors, GGUF without loading everything (live-load, streaming) |
| **Quantization** | `quantized_matmul.c`, `quantized_dot_generic.c`, `dequant_iq2_xxs.c`, Q4_0/Q8_0/Q2_0/TQ3_1S/TQ4_1S dequants | mixed per-role bit ladders (research/057-058): keep the sensitive (attn/embd Q8_0), crush the saturated (experts 2-bit) |
| **Attention** | `wubu_mla.c`, `wubu_attn_kernels.c`, `wubu_attn_gate.c`, `wubu_ring_attn.c`, `wubu_cross_attn.c`, KV: `wubu_kv_cache.c`, `wubu_paged_kv.c`, `wubu_kv_quant.c` | GQA, MLA (latent KV), SSM-Gated-DeltaNet hybrid, ring/paged KV for long context |
| **MoE** | `wubu_moe.c`, `wubu_moe2.c`, `wubu_moe_grouped.c`, `wubu_latentmoe.c`, `wubu_ssd_moe.c`, `wubu_hashrouter.c` | router → top-k routed experts + shared expert; SSD-paged experts (slot-bank LRU) for 256-expert models on small RAM |
| **Training** | `wubu_train.c`, `wubu_backprop.c`, `wubu_train_cli.c` | the standing loop: corpus → train → diagnose → mutate → validate → archive → RLHF oracle → repeat. Real backward, Muon optimizer, rolling checkpoints (3/line) |
| **The AGI organs** | `wubu_amoeba.c`, `wubu_hive.c`, `wubu_moe2.c`, `wubu_prover2.c`, `wubu_agi.c`, `wubu_dsa.c`, `wubu_mhc_mh.c` | the hive = diagnostic system (typed measurement cells, z-score anomaly, causal walker, 5+1 rollback); mHC hyper-connections; DSA indexer |
| **Sampling** | repetition/DRY penalties, temp/top_p | anti-repetition for math (deepseek4: temp 1.0 / top_p 0.95 floor) |
| **CLI/tools** | `tools/gen_text*.c`, `tools/infer_*.c`, 299 test targets | one test per module (`test_<module>.c`), `make test_all` gate |

## Model support (verified)

| Model | Form | Status |
|---|---|---|
| WuBu-35M (the seed) | safetensors + .st checkpoints | training loop runs (SFT loss 8.04→7.32 @2000 steps); tensor store round-trips maxdiff=0 |
| Agents-A1-4B (dense hybrid) | safetensors shards | real SSM forward verified finite (weights now on SD cold storage — mount D: first) |
| KAT-Coder-V2.5-Dev (MoE 256) | safetensors shards | 13/13 shards, SSD slot-bank pages experts (weights on SD) |
| Qwen3.6-27B (dense hybrid) | safetensors shards | adapter derives real dims; SSM forward finite (weights on SD) |
| Qwen3.6-35B-A3B-UD-IQ2_M | GGUF 12 GB | 753 tensors dissected (research/057); all 12 types decode NaN-free |
| **DeepSeek-V4-Flash-0731 Config-I** | GGUF 3-split, 95 GiB | **load gate PASSED** on the real headers: 43 layers, 284.3B params, 1328 tensors, 7 types, 0 mismatches (research/059). Forward wiring = next phase |

## The doctrine (the AGI backbone)

- **The standing loop** — corpus → train → diagnose → mutate → validate →
  archive → RLHF oracle → repeat. Every batch closes gaps ON this loop.
- **The amoeba** — WuBu grows/shrinks by measured cell fitness
  (`docs/wubu-amoeba-design.md`).
- **The hive is the diagnostic system** — `research/056` + the amoeba skill.
- **Mixed compression** — the greatest compression is per-role ladders,
  never uniform (`research/057-058`).
- **The model card** — AGI nature disclosed (`docs/wubu-model-card.md`).

## Build & verify

```bash
make <target>      # any target in the Makefile (C11, GCC/Clang, optional nvcc)
make test_all      # the full test gate (299 test targets — subset gates exist)
make gen_text      # CPU inference binary
make wubu_train    # the trainer
```

Fresh counts (2026-08-04): 305 C modules, 21 CUDA, 553 tools, 299 test targets.
Regenerate the module table: `python3 tools/repodoc/repodoc.py . --modules`.
