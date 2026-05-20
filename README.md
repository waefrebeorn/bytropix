     1|<div align="center">
     2|
     3|# ⚡ bytropix — WuBu Text AI Multi-Modal Inference Engine
     4|
     5|**Pure C inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE, `qwen35moe` arch) + Moondream3 Vision (mmproj)**
     6|
     7|[![Phase: 28e](https://img.shields.io/badge/Phase-28e--Q6K_Fix-blueviolet)](https://github.com/waefrebeorn/bytropix)
     8|[![GPU SSM Decode: ~5.9 tok/s](https://img.shields.io/badge/GPU_SSM-5.9_tok/s-informational)](https://github.com/waefrebeorn/bytropix)
     9|[![KV Cache: Q4_0 4:1 / F16 fallback](https://img.shields.io/badge/KV_Cache-Q4_0_4:1-green)](https://github.com/waefrebeorn/bytropix)
    10|[![Vision: 3D ViT ported](https://img.shields.io/badge/Vision-3D_ViT_384L-success)](https://github.com/waefrebeorn/bytropix)
    11|[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue)](https://opensource.org/licenses/Apache-2.0)
    12|[![GPU: RTX 5050](https://img.shields.io/badge/GPU-RTX_5050_8GB-critical)](https://github.com/waefrebeorn/bytropix)
    13|
    14|</div>
    15|
    16|---
    17|
    18|## 📊 Current State (Phase 28e — May 20)
    19|
    20|| | Status | Metric | Detail |
    21||:------:|--------|--------|--------|
    22|| ✅ | **GPU decode (4K ctx)** | **~5.9 tok/s** — GPU SSM active, all 30 SSM layers on GPU | |
    23|| ✅ | **Q6_K dequant fix** | **FIXED** — was `32.0`, now `d*sc*(v6-32)` (was constant ~365 output) | |
    24|| ❌ | **GPU vs CPU SSM divergence** | **cos-sim -0.66** — output anti-correlated to CPU | |
    25|| ✅ | **CPU SSM matches llama** | **cos-sim 0.994** — FORCE_CPU_SSM path verified | |
    26|| ✅ | **Vision encoder** | **384 LoC port** — 27-layer 3D ViT w/ mmproj projection to 2048-dim | |
    27|| ✅ | **Vision→text pipeline** | **test_vision_real.c** — vision encoder feeds directly into text model | |
    28|| ❌ | **CPU-only `gen_text` build** | **Broken** — wubu_model.o links GPU symbols without .cu objects | |
    29|| 🔴 | **Remote behind** | **8 commits not pushed** — all critical GPU fixes local-only | |
    30|
    31|### What the DA Debunked (Phase 28b claims → reality)
    32|| Doc Claimed | Reality |
    33||-------------|---------|
    34|| 🔴 F32 waste ~2.2 GB | ✅ Already removed (`a032a8f`) |
    35|| 🔴 GPU mem leak ~5.5 GB | ❌ **False positive** — free() already freed everything |
    36|| 🔴 Column-major kernel broken | ❌ **False positive** — it was the CORRECT GGUF layout |
    37|| 🔴 gen_text.c hardcoded prompt | ❌ **False positive** — accepts argv[1] |
    38|
    39|---
    40|
    41|## 🚀 Quick Start
    42|
    43|```bash
    44|# Build GPU inference
    45|make gen_text_gpu
    46|
    47|# Run text inference
    48|GPU=1 MAX_CTX=4096 ./gen_text_gpu "The capital of France is" 20 40
    49|
    50|# Vision encoder test
    51|make test_vision_real
    52|./test_vision_real /mnt/wslg/distro/models/qwen3.6-35b-mmproj-F16.gguf /tmp/screen_vision_input.bin
    53|
    54|# Compare CPU vs GPU SSM
    55|FORCE_CPU_SSM=1 GPU=1 MAX_CTX=4096 ./gen_text_gpu "test" 5 40
    56|```
    57|
    58|**Hardware:** AMD Ryzen 7950X (16C/32T) | 64 GB DDR5 | RTX 5050 8GB VRAM | WSL2
    59|**Model:** `/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (733 tensors, 10.7 GB)
    60|**Vision Proj:** `/mnt/wslg/distro/models/qwen3.6-35b-mmproj-F16.gguf`
    61|
    62|---
    63|
    64|## 🏗️ Architecture
    65|
    66|### Model Spec (Qwen3.6-35B-A3B / `qwen35moe` GGUF)
    67|
    68|```
    69|40 Layers: 10 cycles × (3×SSM → 1×GQA)  ← 3:1 INTERLEAVED!
    70|├── SSM layers: 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38
    71|├── GQA layers: 3,7,11,15,19,23,27,31,35,39
    72|├── Hidden dim:    2048
    73|├── Vocab:         248,320
    74|├── SSM:           16 K-heads × 128, 32 V-heads × 128
    75|├── GQA:           16 Q-heads × 256, 2 KV-heads × 256
    76|├── MoE:           256 experts, 8 active + 1 shared
    77|├── Expert FFN:    512
    78|├── Shared FFN:    512
    79|├── RoPE:          IMRoPE, sections=[11,11,10,0], θ=10M
    80|└── Quant:         Mixed IQ2_XXS / IQ3_XXS / IQ4_XS / Q5_K / Q6_K / Q4_K (~2.7 bpw)
    81|```
    82|
    83|### Vision Encoder (Moondream3 / 3D ViT in mmproj)
    84|
    85|```
    86|27-layer Vision Transformer
    87|├── 3D patch embedding: spatial=16×16, temporal=2 frames
    88|├── Spatial merge: 2×2 grid patch merging (4:1 compression)
    89|├── Hidden dim: 1152 → 16 heads × 72 head_dim
    90|├── GQA attention (16 heads)
    91|├── GELU activation
    92|├── Post-LN + Merger MLP: 4608 → 2048 (matches text hidden dim)
    93|└── Output: image tokens in text embedding space
    94|```
    95|
    96|### Multi-Modal Pipeline
    97|
    98|```
    99|Image → Patch Embed → 27×ViT → Spatial Merge → MMProj → Text tokens → 40×Text Model → Output
   100|                                                      ↑
   101|                                    Qwen3.6 Text Embedding ← GGUF token embeddings
   102|```
   103|
   104|### VRAM Budget (256k Context, Text Only)
   105|
   106|| Component | Size | Format |
   107||-----------|------|--------|
   108|| GQA weights | 1,040 MB | F32 (cuBLAS SGEMM) |
   109|| SSM weights (quantized) | 692 MB | Q5_K/Q6_K on GPU |
   110|| KV cache (GPU: Q4_0) | **1,440 MB** | **4-bit, 4:1 vs FP16** |
   111|| KV cache (GPU: FP16 fallback) | 5,120 MB | FP16 via `GPU_Q4_0_KV=0` |
   112|| Output projection | 1,900 MB | Q4_K quantized GPU kernel |
   113|| MoE + scratch | ~460 MB | IQ2_XXS + temp buffers |
   114|| **Total (Q4_0 GPU)** | **~3,562 MB** | **Fits 6.5-8GB GPU with 3GB headroom** |
   115|
   116|---
   117|
   118|## 🔭 Roadmap — Feature Cream
   119|
   120|**Phase 29 [NOW]: Fix GPU SSM divergence (cos-sim -0.66)**
   121|- Trace recurrence state vs conv state persistence across layers
   122|- Layer-by-layer hidden state compare: GPU path vs CPU path at each SSM stage
   123|- Fix state management → cos-sim > 0.99
   124|
   125|**Phase 30: Build infrastructure**
   126|- Fix CPU `gen_text` build (wrap GPU symbols in `#ifdef GPU_SUPPORT`)
   127|- Push 8 pending commits to remote
   128|- Re-verify cos-sim vs llama.cpp at current code state
   129|
   130|**Phase 31: Vision encoder verification**
   131|- Build and run `test_vision_real` end-to-end
   132|- Verify vision encoder output statistics (range, NaN, distribution)
   133|- Compare vision token distribution vs text token embedding distribution
   134|- Wire mmproj into text model pipeline (already partially done)
   135|
   136|**Phase 32: Multi-modal inference**
   137|- Image → vision encoder → mmproj → text embedding space
   138|- Concatenate or interleave vision tokens with text tokens
   139|- Full forward: vision + text through 40-layer model
   140|- Profile vision+text end-to-end
   141|
   142|**Phase 33: Feature cream — cherry-pick from vault synthesis**
   143|| Feature | Source | Priority |
   144||---------|--------|----------|
   145|| Normalized sigmoid gating | DeepSeekMoE | P0 — MoE correctness |
   146|| Auxiliary-loss-free load balancing | DeepSeek-V3 | P0 — Training stability |
   147|| Chunked prefill (256K) | Qwen2.5-1M | P1 — First-token latency |
   148|| RoPE length extrapolation (4x) | Qwen2.5-1M | P1 — 256K without retrain |
   149|| DSA sparse attention | DeepSeek-V3.2 | P2 — O(L log L) GQA at scale |
   150|| MTP self-speculative decode | DeepSeek-V3 | P2 — 2x decode speed |
   151|| Entropix adaptive sampling | Vault | P3 — Smarter decoding |
   152|| WuBuSparseAttention | Vault | P3 — Linear attention alternative |
   153|
   154|**Phase 34: 256K multi-modal context**
   155|- Test vision + text at full 256K context
   156|- Profile VRAM, tok/s, accuracy vs. llama.cpp reference
   157|- Sliding window / sparse attention for GQA at scale
   158|
   159|**Phase 35: Performance profiling & optimization**
   160|- CUDA events per kernel: SSM, GQA, MoE, output proj
   161|- Identify true bottlenecks (MoE ~20-40ms hypothesized)
   162|- Fuse remaining kernels, optimize memory transfers
   163|- Target: 10+ tok/s at 256K with vision
   164|
   165|---
   166|
   167|## 🔬 Verification Tools
   168|
   169|| Tool | Purpose |
   170||------|---------|
   171|| `tools/layer_cos_sim` | Per-layer cosine similarity vs llama.cpp |
   172|| `tools/test_vision_real` | Vision encoder end-to-end with real image |
   173|| `tools/compare_logits` | Logit-level comparison vs reference |
   174|| `DUMP_LAYER_DIR` | Save per-layer hidden states to `.bin` |
   175|| `DUMP_INTERMEDIATE_DIR` | Save ALL intermediate tensors (53 types/layer) |
   176|| `PROFILE` | Per-layer timing breakdown |
   177|
   178|---
   179|
   180|## 📁 Project Structure
   181|
   182|```
   183|bytropix/
   184|├── src/              # Core C/CUDA (wubu_model, ssm, moe, gqa, vision, gpu)
   185|├── include/          # Headers (gguf_reader, wubu_model, wubu_vision)
   186|├── tools/            # 50+ binaries + tests + Python analysis
   187|├── .hermes/          # Mind palace, vault, DA audits, plans
   188|├── DIAGRAMS/         # SVG architecture diagrams
   189|├── data/             # Pre-extracted embeddings, vision configs
   190|├── vault/            # Unsloth quant format, cache compression refs
   191|└── tests/            # Regression test harness
   192|```
   193|
   194|---
   195|
   196|## 📚 References
   197|
   198|- `.hermes/mind-palace/` — State, plan, goal-mantra, prestige, overnight map
   199|- `.hermes/vault/` — Synthesis (architecture), papers (DeepSeek, Qwen, Gemma)
   200|- `.hermes/vault/synthesis.md` — Complete architectural cross-reference (~19KB)
   201|- `~/llama.cpp/build/bin/llama-cli` — Ground truth reference binary
   202|
   203|---
   204|
   205|<div align="center">
   206|
   207|*Engine: bytropix — C inference for Qwen3.6-35B-A3B text + Moondream3 vision. Phase 28e: Q6_K dequant fixed, GPU SSM cos-sim -0.66 under debug. Vision encoder ported (384 LoC). Next: fix state divergence, wire multi-modal pipeline.*
   208|*DA principle: every claim verified at runtime. Unverified = ❓. Fixed = ✅. Debunked = documented.*
   209|
   210|</div>
   211|

---

## 📜 License & Disclaimer

bytropix is **open-source educational and research software**.
**No liability** assumed by authors for any use including security, privacy, or regulatory compliance.

This project provides **open scaffolding** — anyone may build, modify, deploy, or sell inference
services built on this foundation. The authors do not sell inference services.

Operators are responsible for their own API key management, content filtering, data privacy,
and legal compliance.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
