# bytropix — Colonel on-device inference engine: status

## ds4-ssd RAM/SSD technique (NEW, this session)
Replicated Anemll/ds4-ssd's signature method ("LLM in a flash" applied to MoE):
- **Dense/shared/router tensors stay resident in RAM** (loaded by the engine).
- **Routed MoE expert weights live on SSD** in a sidecar directory `experts.<L>.bin`
  (BF16-packed), paged into a fixed **slot-bank** of resident F32 slots per layer
  on router miss, with **LRU eviction**. This lets a 256-expert model (KAT-Coder)
  run in a fraction of the RAM its full expert footprint would need.
- Module: `include/wubu_ssd_moe.h` + `src/wubu_ssd_moe.c` (self-contained, C11).
- Sidecar packer: `tools/pack_kat_sidecar.c` (extracts HF MoE experts -> BF16 sidecar).
- Verified: `make test_ssd_moe` — 12 experts paged through a 3-slot bank, BF16->F32
  dequant matmul matches a fully-resident reference exactly (PASS). LRU evictions
  and the page-in (disk pread) path both exercised.

## Model support matrix (Colonel / HF safetensors + GGUF)
| Model | Arch | D | layers | experts | status |
|-------|------|---|--------|---------|--------|
| Agents-A1-4B | qwen3_5 (dense, SSM+GQA, multimodal) | 2560 | 32 | 0 | tokenizer OK; bridge probes all shards; FP32 OOM on 13GB box |
| KAT-Coder-V2.5-Dev | qwen3_5_moe (256/8 + shared, hybrid) | 2048 | 40 | 256 | sidecar packer built; pending real pack + SSD run |
| Qwen3.6-27B-base | qwen3_5 (dense, 64 hybrid) | 5120 | 64 | 0 | config read; pending bridge hybrid + LoRA target |
| BTL-3 | LoRA on Qwen3.6-27B | — | — | — | adapter-only; pending LoRA apply path |

## Bugs fixed this session
- HF BPE tokenizer infinite loop (rewrote as bounded single-pass scanner; loads
  248,044 vocab; encode/decode round-trip verified on Agents-A1-4B).
- Safetensors bridge probed only shard 0 -> wrong dims (VD 2560 vs real 4096,
  qh 16 vs 32, kvh 2 vs 4). Now probes across all shards via `wubu_shard_dimof`.
  Added `wubu_shard_has`/`wubu_shard_dimof` public API.

## Known blockers (honest)
- Full FP32 Agents (9GB weights + 5GB embed/lm_head) OOMs the 13GB box. Needs
  BF16-on-disk + per-layer FP32 dequant to fit, OR the slot-bank concept applied
  to embed/lm_head, OR a bigger box. Real generation not yet produced.
- SSM forward math (GatedDeltaNet) not yet validated for correct logits.
- KAT real sidecar not yet built (download in progress); MoE forward must call
  the slot-bank pager instead of the in-RAM expert blob.
- Qwen3.6-27B / BTL-3 not yet loaded; hybrid `layer_types` (full_attention = GQA
  only, no SSM) must be handled per-layer in the forward.
