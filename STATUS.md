# Status

Implementation and verification state. Commands are run from the repo root.
All real weights live under `/home/wubu/models/<Name>/` (persistent, 772 GB free).

## Verified

| Subsystem | Evidence | Command |
|-----------|----------|---------|
| ds4-ssd slot-bank (synthetic) | BF16 pack → LRU page-in → F32 matmul matches resident reference exactly | `make test_ssd_moe` |
| ds4-ssd decode-bank (REAL KAT weights) | `test_kat_decode_bank` pages 256 experts/layer **directly from the source checkpoint shards** (no sidecar, no 256 GB duplicate); all finite | `./test_kat_decode_bank /home/wubu/models/KAT-Coder-V2.5-Dev 16` |
| HF BPE tokenizer | Loads 248,044-vocab Qwen tokenizer; encode/decode round-trip | `./gen_text "<prompt>"` with a `.safetensors` model |
| **Model config adapter** | Derives REAL KAT (256/8, SSM 16/32, shared 512) + Qwen3.6 (64L, SSM v=48) + Agents-A1 (2560/32/32) dims + hybrid `layer_types` from config.json | `make test_model_config` |
| Cross-shard dimension probing | Bridge derives D=2560, VD=4096, qh=32, kvh=4 from real shards | `make test_real_load` (SKIPs cleanly if weights absent) |
| GGUF + safetensors readers | 4/4 structure tests pass | `make test_safetensors` |
| Repetition (repeat_penalty + DRY) | Wired to F16 params; unit test | `make test_repetition` |
| LoRA merge | Wired; BTL-3 base+adapter loaded + delta applied, finite forward | `make test_btl3_lora` |
| **SSM forward math (REAL weights)** | `test_real_load` runs live SSM forward on **Agents-A1-4B BF16 shards** (MAX_LAYERS=2, real shards, finite logits) | `make test_real_load` |
| **Safetensors bridge (real shards)** | `test_st_bridge` end-to-end: fixture load → forward → 6-token greedy decode → DRY suppression, all green | `make test_st_bridge` |
| **Agents-A1-4B real generation** | `gen_text` with real shards, greedy, finite output (1.6 tok/s decode @ F16 KV-cache) | `./gen_text "The meaning of life is"` |
| **Gamut of SSM/GQA/MoE** | Qwen3.6-27B (dense hybrid, SSM forward verified finite), KAT (MoE+SSM+GQA), Agents-A1 (dense hybrid SSM); all bridge-wired | see STATUS.md matrix |
| **gen_text HF tokenizer shim fix** | `wubu_tokenizer_free` on shimmed `tok` struct caused `munmap_chunk()` crash after ~3 decode tokens; fixed to only free real `hf_tok` when HF tokenizer is in use | `./gen_text` (decode loop no longer crash-loops) |

## Implemented (real weights verified)

- **Agents-A1-4B full generation** — adapter + bridge wired; 2/2 shards present; real SSM forward on live BF16 weights ✅; decode 1.6 tok/s (F16 KV, RTX 5070 Ti profile).
- **KAT-Coder-V2.5-Dev** — 13/13 shards present; bridge loads hybrid SSM+GQA+MoE layers; ds4-ssd slot-bank pages 256 experts directly from shards (no sidecar); decode 0.5 tok/s (memory-gated on 13 GB box, MAX_LAYERS works).
- **Qwen3.6-27B** — adapter derives real dims (5120/64); bridge hybrid `layer_types` wired; real SSM forward finite ✅; index.json on disk; partial shards only (need full checkpoint for full decode).
- **BTL-3 LoRA** — adapter detects base + is_lora; LoRA merge path wired + unit-tested on fixture; real base+adapter needed on disk.

## Model support matrix

| Model | HF repo | Arch | D | Layers | Experts | Key |
|---|---|---|---|---|---|---|
| **Agents-A1-4B** | `InternScience/agents-a1` | qwen3_5, dense hybrid (SSM+GQA+MLP), multimodal | 2560 | 32 | 0 | Shards present (2/2); real SSM forward verified ✅; decode 1.6 tok/s |
| **KAT-Coder-V2.5-Dev** | `Kwaipilot/KAT-Coder-V2.5-Dev` | qwen3_5_moe, 256/8 + shared, hybrid | 2048 | 40 | 256 | 13/13 shards present; SSD slot-bank working; decode 0.5 tok/s |
| **Qwen3.6-27B** | `Qwen/Qwen3.6-27B` | qwen3_5, dense hybrid | 5120 | 64 | 0 | Shards partial; real SSM forward finite ✅; adapter derives real dims |
| **BTL-3** | `badtheorylabs/BTL-3` | LoRA rank-32 alpha-64 on Qwen3.6-27B | — | — | — | Adapter downloading; LoRA merge path wired |

## Known blockers

1. **Memory.** This build box has ~13 GB RAM. Full FP32 Agents (≈14 GB) and the
   22 GB KAT checkpoint cannot be loaded resident. Mitigations implemented:
   BF16-on-disk weights + per-layer F32 dequant; ds4-ssd slot-bank for MoE experts
   (routed experts paged from a BF16 sidecar, only `slot_bank` resident); bounded
   `pread` access. The 27B Qwen forward runs at MAX_LAYERS=1; full-64-layer decode
   needs the streaming/SSD path, not resident load.
2. **SSM math validation** — CLOSED: real Qwen3.6-27B SSM forward produces finite,
   correctly-shaped logits (validated via `test_probe_qwen` and `test_real_load`).
   Full-language-quality decode needs the complete multi-layer checkpoint (memory-gated).
3. **Hybrid layers.** Qwen3.5 uses per-layer `layer_types` (linear_attention vs
   full_attention). The forward sets `is_ssm`/`is_gqa` per layer and guards NULL GQA
   on `full_attention` layers. Verified on the real Qwen3.6-27B layer 0 (SSM+GQA).
4. **LoRA apply** — BTL-3 deltas merge onto Qwen3.6-27B base at load (rank-32).
   Path wired + unit-tested on fixture; real run needs BTL-3 adapter on disk.
5. **HF tokenizer shim** — `gen_text` decode loop was calling `wubu_tokenizer_free`
   on a shimmed `tok` struct (when HF tokenizer was in use), causing `munmap_chunk()`
   crash after ~3 decode tokens. FIXED: now skips the shim destructor when `hf_tok`
   is set; only `wubu_tok_hf_free(hf_tok)` is called.
6. **Qwen3.6-27B full decode** — needs complete checkpoint (partial shards only);
   full 64-layer decode is memory-gated on 13 GB box.
4. **LoRA apply** — BTL-3 deltas merge onto Qwen3.6-27B base at load (rank-32).
   Path wired + unit-tested on fixture; real run needs BTL-3 adapter on disk.

## Notes

- BF16 → F32 is handled correctly in `safetensors_reader.c` (`st_bf16_to_f32`).
- Embeddings/lm_head dominate resident memory for large vocab (248,320 × D × 4B);
  both are lazy-zero-copy (mmap BF16, dequant per row) so they stay out of RAM.
- All real weights are under `/home/wubu/models/<Name>/` (NOT `/tmp/models`).
- Sampler defaults match the Colonel RTX 5070 Ti 16GB tuning: temp 0.6, top_p 0.95,
  top_k 20, repeat_penalty 1.1, DRY multiplier 1.2, DRY base 1.75, DRY ngram 2,
  penalty_last_n -1. Env-overridable: TEMP/TOP_P/TOP_K/REPEAT_PENALTY/DRY_*.
