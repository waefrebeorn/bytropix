# Status

Implementation and verification state. Commands are run from the repo root.

## Verified

| Subsystem | Evidence | Command |
|-----------|----------|---------|
| ds4-ssd slot-bank (synthetic) | BF16 pack → LRU page-in → F32 matmul matches resident reference exactly | `make test_ssd_moe` |
| HF BPE tokenizer | Loads 248,044-vocab Agents tokenizer; encode/decode round-trip | `./gen_text "<prompt>"` with a `.safetensors` model |
| **Model config adapter** | Derives REAL KAT (256/8, SSM 16/32, shared 512) + Qwen3.6 (64L, SSM v=48) dims + hybrid `layer_types` from config.json | `make test_model_config` |
| Cross-shard dimension probing | Bridge derives D=2560, VD=4096, qh=32, kvh=4 from real shards | `make test_real_load` |
| GGUF + safetensors readers | 4/4 structure tests pass | `make test_safetensors` |
| Repetition (repeat-penalty + DRY) | Wired to F16 params | unit test |
| LoRA merge | Wired | unit test |

## Implemented, not yet end-to-end verified

- **`wubu_moe_forward_ssd`** — SSD-paged MoE forward. Code-complete; not yet
  exercised by a full generation (requires the MoE forward math + a packed
  real sidecar).
- **ds4-ssd wired into load path** — `wubu_model_init_safetensors_ssd` opens a
  sidecar (`wubu_ssd_moe_t`), sets `model->ssd_moe`, and the per-layer forward
  dispatches to `wubu_moe_forward_ssd` (routed experts paged from disk; router +
  shared expert stay resident). Auto-detected via `<modeldir>/sidecar` or
  `KAT_SIDECAR` env. When active, the bridge skips the 3.2 GB/layer resident
  expert blobs. `gen_text` builds with this path.
- **Real-KAT sidecar** — `tools/pack_kat_sidecar.c` builds a BF16 sidecar from
  HF MoE weights. `tools/test_ssd_moe_real.c` verifies the slot-bank against
  real KAT-256-expert weights via bounded `pread` (RAM ~MB, no full-checkpoint
  load). Blocked on a complete KAT download (all 13 shards currently PARTIAL on
  disk — only JSON headers were fetched; tensor data sections absent).
- **SSM forward math** — Gated-DeltaNet numerics not yet validated against a
  reference; output logits not confirmed correct.

## Model support matrix

| Model | Arch | D | layers | experts | State |
|-------|------|---|---------|---------|-------|
| Agents-A1-4B | qwen3_5, dense hybrid (SSM+GQA+MLP), multimodal | 2560 | 32 | 0 | Tokenizer + bridge OK. FP32 load OOMs 13GB box (9GB weights + 5GB embed/lm_head). |
| KAT-Coder-V2.5-Dev | qwen3_5_moe, 256/8 + shared, hybrid | 2048 | 40 | 256 | Adapter derives real dims (verified). Bridge loads shared expert + per-layer SSM/GQA via `layer_types`. Sidecar packer built; full load blocked on complete download + slot-bank wiring. |
| Qwen3.6-27B-base | qwen3_5, dense hybrid | 5120 | 64 | 0 | Adapter derives real dims (verified). Bridge hybrid `layer_types` wired. Full load blocked on memory (27B dense). |
| BTL-3 | LoRA on Qwen3.6-27B | — | — | — | Adapter detects base + is_lora. LoRA apply path in bridge (needs base loaded first). |

## Known blockers

1. **Memory.** This build box has ~13 GB RAM. Full FP32 Agents (≈14 GB) and the
   22 GB KAT checkpoint cannot be loaded resident. Mitigations implemented or
   pending: BF16-on-disk weights + per-layer F32 dequant; ds4-ssd slot-bank for
   MoE experts; bounded `pread` access (never `mmap`/scan the whole checkpoint).
2. **SSM math validation.** Forward produces output but logits are not verified
   correct; no reference comparison yet.
3. **Hybrid layers.** Qwen3.5 uses per-layer `layer_types` (`linear_attention`
   vs `full_attention`). The forward must set `is_ssm`/`is_gqa` per layer and
   guard NULL GQA on `full_attention` layers.
4. **LoRA apply.** BTL-3 deltas must be merged onto the Qwen3.6-27B base at load.

## Notes

- BF16 → F32 is handled correctly in `safetensors_reader.c` (`st_bf16_to_f32`).
- Embeddings/lm_head dominate resident memory for large vocab (248,320 × D × 4B).
- All real weights are under `/tmp/models/` (not in repo).
