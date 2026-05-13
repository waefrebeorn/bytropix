# Qwen Research Papers & Technical Reports

All papers collected from QwenLM GitHub repos, arxiv, and Hugging Face model cards.

## Core Papers (PDFs)

| Paper | Source | Size |
|-------|--------|------|
| **Qwen3 Technical Report** | GitHub repo + arxiv:2505.09388 | 674K + 779K |
| **Qwen3-Coder Next** | Qwen3-Coder GitHub repo | 2.6M |
| **Qwen3 Paper (Full Text)** | Text extraction from PDF | 117K |

## Model Architecture Details (from Hugging Face config.json + model card)

### Qwen3.5-9B (Dense Model)
*We have GGUF: Qwen3.5-9B-Q4_K_M, 5.3G*

| Parameter | Value |
|-----------|-------|
| **Architecture** | `Qwen3_5ForConditionalGeneration` |
| **Text model type** | `qwen3_5_text` |
| **Hidden dim** | 4096 |
| **Layers** | 32 |
| **Attention heads (Q)** | 16 |
| **KV heads** | 4 |
| **Head dim** | 256 |
| **Hidden layout** | 8× (3×Gated DeltaNet → FFN → 1×Gated Attention → FFN) |
| **Gated DeltaNet** | Linear attention: 32 V-heads, 16 QK-heads, head_dim=128 |
| **Gated Attention** | Full attention every 4th layer (8 total), 16 Q/4 KV heads |
| **FFN intermediate** | 12288 (SwiGLU) |
| **RoPE** | theta=10M, partial_rotary_factor=0.25, MRoPE (11,11,10) |
| **Norm** | RMSNorm, eps=1e-6 |
| **Vocab size** | 248320 (padded) |
| **Context** | 262,144 native, extensible to 1M |
| **MTP** | 1 MTP head |
| **Vision encoder** | 27 layers, 1152 hidden, 16 heads, patch=16, temporal=2, out=4096 |
| **Tie embeddings** | false |

### Qwen3.6-35B-A3B (MoE Model)
*We have GGUF: Qwen3.6-35B-A3B-UD-IQ2_M, 11G. Also Ornstein3.6 finetune.*

| Parameter | Value |
|-----------|-------|
| **Architecture** | `Qwen3_5MoeForConditionalGeneration` |
| **Text model type** | `qwen3_5_moe_text` |
| **Hidden dim** | 2048 |
| **Layers** | 40 |
| **Hidden layout** | 10× (3×Gated DeltaNet → MoE → 1×Gated Attention → MoE) |
| **Gated DeltaNet** | Linear attention: 32 V-heads, 16 QK-heads, head_dim=128 |
| **Gated Attention** | Full attention every 4th layer (10 total), 16 Q/2 KV heads |
| **Head dim** | 256 |
| **MoE** | 256 experts total, 8 routed + 1 shared active, intermediate=512 |
| **RoPE** | theta=10M, partial_rotary_factor=0.25, MRoPE (11,11,10) |
| **Norm** | RMSNorm, eps=1e-6 |
| **Vocab size** | 248320 (padded) |
| **Context** | 262,144 native, extensible to 1M |
| **MTP** | 1 MTP head |
| **Vision encoder** | 27 layers, 1152 hidden, 16 heads, patch=16, temporal=2, out=2048 |
| **Tie embeddings** | false |
| **Router aux loss** | 0.001 |

### Qwen3.6-27B (Dense Model)
*No GGUF for this one locally*

| Parameter | Value |
|-----------|-------|
| **Architecture** | `Qwen3_5ForConditionalGeneration` |
| **Hidden dim** | 5120 |
| **Layers** | 64 |
| **Hidden layout** | 16× (3×Gated DeltaNet → FFN → 1×Gated Attention → FFN) |
| **Gated DeltaNet** | Linear: 48 V-heads, 16 QK-heads, head_dim=128 |
| **Gated Attention** | Full attention every 4th (16 total), 24 Q/4 KV heads |
| **FFN intermediate** | 17408 (SwiGLU) |
| **Output gate** | swish |
| **Vocab** | 248320 |
| **Context** | 262K native, 1M extensible |

### Qwen3 Dense (for reference — paper Table 1)
| Model | Layers | Q/KV Heads | Tie Emb | Context |
|-------|--------|-----------|---------|---------|
| Qwen3-0.6B | 28 | 16/8 | Yes | 32K |
| Qwen3-1.7B | 28 | 16/8 | Yes | 32K |
| Qwen3-4B | 36 | 32/8 | Yes | 128K |
| Qwen3-8B | 36 | 32/8 | No | 128K |
| Qwen3-14B | 40 | 40/8 | No | 128K |
| Qwen3-32B | 64 | 64/8 | No | 128K |

### Qwen3 MoE (paper Table 2)
| Model | Layers | Q/KV Heads | Experts (T/A) | Context |
|-------|--------|-----------|---------------|---------|
| Qwen3-30B-A3B | 48 | 32/4 | 128/8 | 128K |
| Qwen3-235B-A22B | 94 | 64/4 | 128/8 | 128K |

## Key Differences: Qwen3 → Qwen3.5/3.6

| Aspect | Qwen3 | Qwen3.5+ |
|--------|-------|----------|
| **Attention** | GQA (full softmax) only | Hybrid: 75% Gated DeltaNet (linear) + 25% GQA (full) |
| **Output gate** | None | `attn_output_gate=true`, `output_gate_type=swish` (27B) |
| **Layer pattern** | All same attention | 3 linear → 1 full (repeating) |
| **Vocab** | 151,669 | 248,320 (padded) |
| **Context** | 32K-128K | 262K native, 1M extensible |
| **MoE experts** | 128/8, no shared | 256/8+1 shared, intermediate=512 |
| **Tokenizer** | BBPE 151669 | BBPE 248320 |

## The Novelties for WuBu

1. **Gated DeltaNet** — linear attention replaces 75% of softmax layers. Key for wubu hyperbolic because linear attention naturally works with gyration. Head_dim=128 for linear, 256 for full.

2. **Hybrid layout** — `3×linear → 1×full` repeating block. This is the efficient hybrid architecture mentioned in Qwen3.5. Every 4th layer has full softmax attention; the rest use Gated DeltaNet (linear attention with gating).

3. **MRoPE** — multi-resolution RoPE with `(11, 11, 10)` section for 3D positional encoding. 32 total dims. This is for their vision-language model (3D spatiotemporal).

4. **Huge vocab** — 248,320 tokens (padded). This is how they handle 201 languages. Our tokenizer will need to be different but the embedding extraction is from this space.

## READMEs (Model descriptions)

| Model | File | Size |
|-------|------|------|
| Qwen3.6 | Qwen3.6_README.md | 12K |
| Qwen3 | Qwen3_README.md | 25K |
| Qwen3-VL | Qwen3_VL_README.md | 53K |
| Qwen3-Coder | Qwen3_Coder_README.md | 22K |
| Qwen3-Omni | Qwen3_Omni_README.md | 101K |
| Qwen2.5-Omni | Qwen2.5_Omni_README.md | 60K |
| Qwen-Image | Qwen_Image_README.md | 42K |
| Qwen2-Audio | Qwen2_Audio_README.md | 21K |
| Qwen2.5-Math | Qwen2.5_Math_README.md | 14K |
| Qwen-Agent | Qwen_Agent_README.md | 15K |
| Qwen3.5-9B model card | Qwen3.5-9B_modelcard.md | 78K |
| Qwen3.6-35B model card | Qwen3.6-35B_modelcard.md | 65K |

## Config files (exact arch specs)

| Model | File |
|-------|------|
| Qwen3.5-9B | Qwen3.5-9B_config.json |
| Qwen3.6-27B | (from HF API) |
| Qwen3.6-35B-A3B | Qwen3.6-35B_config.json |

## Related Repos at QwenLM (42 total)
Key repos: Qwen, Qwen3, Qwen3.6, Qwen3-VL, Qwen3-Coder, Qwen3-Omni, Qwen2.5-Omni, Qwen-Image, Qwen2-Audio, Qwen2.5-Math, Qwen-Agent, QwQ, Qwen3-Embedding, WebWorld, FlashQLA, ParScale, ProcessBench, qwen-code, Qwen3-TTS, Qwen3-ASR
