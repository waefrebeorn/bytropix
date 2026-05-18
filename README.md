# bytropix — WuBu Text AI

**Pure C + CUDA inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE).**
*May 18 — Phase 2 complete: 0.6 tok/s decode, cos-sim 0.9968 vs llama.cpp.*

---

## Current State

| Metric | Value |
|--------|-------|
| Layers | 40 (30 SSM + 10 GQA) |
| Dequant types | 7 verified (all match llama.cpp) |
| **Full model cos-sim vs ref** | ✅ **0.9968** (quantization noise, no arch bugs) |
| **Per-layer cos-sim** | ✅ All 40 layers > 0.995 (smooth decay 0.9985→0.9952) |
| **GQA Q/gate interleave** | ✅ **FIXED** (cos-sim -0.51 → 0.9968) |
| **IMRoPE** | ✅ Implemented ([11,11,10,0], θ=10M) |
| **gen_text** | ✅ Working — coherent 32-token generation |
| **Decode speed** | ⚠️ 0.6 tok/s (CPU, 16 threads) |
| **MoE OpenMP** | ✅ 3× speedup (44ms→15ms/layer) |

### Known Bugs

No remaining DA-v10 gaps. All 10 closed.

| Issue | Status | Notes |
|-------|--------|-------|
| **Chat template** | ✅ Fixed (CHAT=1 env var) | Generates structured reasoning |
| **SSM L2 eps** | ⚠️ 1e-12 vs 1e-6 | Not blocking (cos-sim 0.9968) |
| **No SIMD vec_dot** | ⚠️ Generic C | ~0.003 cos-sim gap |

---

## Quick Start

```bash
# Build
make gen_text

# Run inference (CPU, 16 threads)
./gen_text "The capital of France is" 32

# Run full 40-layer cos-sim verification
make test_full_moe && ./test_full_moe

# Per-layer profile
PROFILE=1 ./test_full_moe

# Reference comparison (requires ref_dumper)
make ref_dumper && ./ref_dumper /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf 248044
```

**CUDA:** `/usr/local/cuda-13.1/bin/nvcc -arch=sm_120` | **GPU:** RTX 5050 6.4GB | **Model:** Qwen3.6-35B-A3B-UD-IQ2_M.gguf

---

## Architecture

### Model Spec (Qwen3.6-35B-A3B)

| Component | Value |
|-----------|-------|
| Layers | 40 (30 SSM Gated DeltaNet + 10 GQA full attention) |
| Hidden dim | 2048 |
| Context | 262K native |
| Vocab | 248,320 |
| MoE | 256 experts, 8 active + 1 shared per token |
| SSM heads | 16 K-heads × 128, 32 V-heads × 128 |
| GQA heads | 16 Q × 256, 2 KV × 256 |
| Expert FFN dim | 512 |
| RoPE | θ=10,000,000, MRoPE 3D sections=[11,11,10,0] |
| Quant | Mixed: IQ2_XXS / IQ3_XXS / IQ4_XS / Q5_K / Q6_K / Q4_K |

### Layer Structure

Each of the 40 layers:
1. **RMS Norm** — `rms_norm(x, weight, eps=1e-6)`
2. **SSM (30 layers) or GQA (10 layers)** — every 4th is GQA
3. **MoE** — router → top-8 experts + shared expert → SiLU-gated sum
4. **Residual** — `x += moe_output`

### Tensor Layout (GGUF)

```
ffn_gate_exps.weight:  [2048, 512, 256]  ← dims[0]=D_MODEL (innermost), dims[2]=N_EXPERTS (outermost, contiguous)
ffn_up_exps.weight:    [2048, 512, 256]  ← same layout, expert eid = e * raw_per_exp
ffn_down_exps.weight:  [512, 2048, 256]  ← same, expert eid = e * raw_per_exp
output.weight:         [2048, 248320]    ← D_MODEL × vocab
```

**Critical:** MoE expert tensors are 3D with expert index varying SLOWEST (outermost dim).  
This means for each expert, the (d_model, d_ff) matrix is stored contiguously.  
Extracting expert e requires reading `e * raw_per_exp` bytes from the raw data blob — exactly what `dequant_multi_expert_contiguous` does.  
The earlier "interleaved" hypothesis was WRONG — confirmed by cos-sim 1.0 vs ggml matmul.

---

## Diagrams

![Bug Status](DIAGRAMS/bug-status.svg)
![Phase Roadmap](DIAGRAMS/phase-roadmap.svg)
![Inference Pipeline](DIAGRAMS/inference-pipeline.svg)
![Quant Layer Map](DIAGRAMS/quant-layer-map.svg)

---

## Project Structure

```
bytropix/
├── src/               # Core C implementation
│   ├── wubu_ssm.c            SSM Gated Delta Net
│   ├── wubu_mobius.c         Hyperbolic operations
│   ├── wubu_moe.c            MoE forward/backward
│   ├── wubu_poincare_gqa.c   Poincaré attention
│   ├── wubu_nested_ssm.c     Nested hyperbolic SSM
│   ├── gguf_reader.c         GGUF format + dequant
│   ├── cuda_kernels.cu       GPU kernels
│   └── wubu_vision.c         Vision transformer
├── include/           # Headers
├── tools/             # Binaries
│   ├── infer_text.c          CPU inference (debug target)
│   ├── infer_text_gpu.cu     GPU inference
│   └── dump_llama_logits.c   Reference extraction
├── DIAGRAMS/          # Architecture diagrams (SVG)
├── .hermes/           # Mind palace + research vault
└── THEORY/            # Papers and proofs
```

## Key Binaries

| Binary | What It Does | Status |
|--------|-------------|--------|
| `gen_text` | CPU text generation | 🟢 Works (0.6 tok/s) |
| `ref_dumper` | Reference extraction (libllama.so) | 🟢 Works |
| `test_full_moe` | Cos-sim vs reference (40 layers) | 🟢 0.9968 verified |
| `dump_tensor_our` | Tensor dims + dequant dump | 🟢 Works |
| `llama-cli` | Reference (external) | 🟢 Ground truth |

## References

- `.hermes/mind-palace/` — State, goal, plan, prestige, overnight map
- `~/.hermes/skills/mlops-inference/bytropix-moe-expert-layout` — Bug skill
- `~/llama.cpp/build/bin/llama-cli` — Ground truth reference
- `~/llama.cpp/src/models/qwen35moe.cpp` — Reference implementation
- `~/llama.cpp/src/models/qwen3next.cpp` — Newer arch reference
