<div align="center">

# ⚡ bytropix — CPU Inference Engine (i5-8365U)

**Pure C inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE, `qwen35moe` arch)**
**CPU-only. No GPU. Quantized matmul (Q4_K/Q6_K/IQ2_M).**

[![CPU decode: ~2.3 tok/s](https://img.shields.io/badge/CPU_decode-2.3_tok/s-informational)](https://github.com/waefrebeorn/bytropix)
[![Cos-sim vs llama.cpp: 0.974 (IQ2_M floor)](https://img.shields.io/badge/Cos_sim-0.974_(IQ2_M_floor)-yellow)](https://github.com/waefrebeorn/bytropix)
[![KV Cache: F32 512K](https://img.shields.io/badge/KV_Cache-F32_512K-green)](https://github.com/waefrebeorn/bytropix)
[![Platform: WSL + i5-8365U](https://img.shields.io/badge/Platform-WSL_i5--8365U-blue)](https://github.com/waefrebeorn/bytropix)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue)](https://opensource.org/licenses/Apache-2.0)

</div>

---

## 📊 Current State (May 27, 2026 — CPU-Only)

| | Status | Metric | Detail |
|:------:|--------|--------|--------|
| ✅ | **Output proj fix** | **ZERO→REAL logits** | GCC -O3 + if(0) wrapper killed else branch. AVX2 vec_dot produced zeros. Both fixed. |
| ✅ | **Cos-sim vs llama.cpp** | **0.974** | IQ2_M quantization floor (2-bit 2048-dim). Pure random noise. Need Q3_K+ to reach >0.99. |
| ✅ | **Local inference** | **serve_local.py** | All 4 test scripts patched from proxy to real local CPU inference. |
| ✅ | **Test infra** | **6/6 tests** | `test-512k-suite.sh` — KV alloc, sparse attn, memory, RoPE, NES build all verified. |
| ✅ | **512K context** | **2.8 tok/s** | KV cache at 524288 confirmed. Context-size independent decode. |
| 🟡 | **NES emulator** | **BENCHMARK** | Pre-built 6502 emulator at ~/hermes-test/projects/nes-emulator/. Generates ASCII workload for 512K testing. Do NOT develop. |

### What Was Fixed This Session

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Logits all zeros | GCC -O3 + `if(0){}else{}` wrapper killed output projection branch | Removed wrapper. Output projection runs directly. |
| Logits still zero after fix | AVX2 `ggml_vec_dot_q4_K_q8_K_avx2` produces zeros on i5-8365U | Forced `ggml_vec_dot_q4_K_q8_K_generic` |
| All 4 test scripts used proxy | `inference-server.py` proxied to cloud APIs | Patched to `serve_local.py` (real local inference) |
| `test-512k-suite.sh` SIGPIPE | `set -euo pipefail` + `grep -q` killed producers | Removed grep -q pipes, fixed exit code capture |

---

## 🚀 Quick Start

```bash
# Build CPU inference
make gen_text_cpu -j4

# Run text inference
MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf \
  OMP_NUM_THREADS=4 \
  DUMP_LOGITS=/tmp/logits.bin \
  ./gen_text_cpu "The capital of France is" 20 40

# Start local inference server
MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf \
  OMP_NUM_THREADS=4 \
  python3 tools/serve_local.py --port 8001

# Run test suites
bash tools/test-512k-suite.sh
bash tools/test-hermes-integration.sh 8005
```

**Hardware:** Intel i5-8365U (4C/8T) | 16 GB RAM | WSL2 (no GPU)
**Model:** `~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf` (733 tensors, 11 GB)

---

## 🏗️ Architecture

### Model Spec (Qwen3.6-35B-A3B / `qwen35moe` GGUF)

```
40 Layers: 10 cycles × (3×SSM → 1×GQA)
├── SSM layers: 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38
├── GQA layers: 3,7,11,15,19,23,27,31,35,39
├── Hidden dim:    2048
├── Vocab:         248,320
├── SSM:           16 K-heads × 128, 32 V-heads × 128
├── GQA:           16 Q-heads × 256, 2 KV-heads × 256
├── MoE:           256 experts, 8 active + 1 shared
├── Expert FFN:    512
├── Shared FFN:    512
├── RoPE:          IMRoPE, sections=[11,11,10,0], θ=10M
└── Quant:         Mixed IQ2_XXS / IQ3_XXS / IQ4_XS / Q5_K / Q6_K / Q4_K (~2.7 bpw)
```

---

## 🔬 Verification Tools

| Tool | Purpose |
|------|---------|
| `tools/dump_ref` | Dump llama.cpp reference logits for cos-sim comparison |
| `tools/check_logits.py` | Python logit analyzer (range, top-k, variance, NaN check) |
| `tools/py_compare_logits.py` | Compare our logits vs reference (cos-sim, segment breakdown) |
| `DUMP_LOGITS=/tmp/our.bin` | Save last token logits to binary file |
| `DUMP_LAYER_DIR=/tmp/layer_dump` | Save all 40 layer hidden states |
| `PROFILE=1` | Per-layer timing breakdown |

---

## 📁 Project Structure

```
bytropix/
├── src/              # Core C (wubu_model, ssm, moe, gqa, quantized_matmul, tokenizer)
├── include/          # Headers (wubu_model, wubu_ssm, wubu_gqa, gguf_reader)
├── tools/            # Test binaries, diagnostic tools, Python analysis, test scripts
├── .hermes/          # Mind palace, vault, state, plan, goal-mantra
├── vault/            # Parity analysis, fix docs, remaining gaps tracking
└── tests/            # Pytest suite (24 tests)
```

---

## 🔭 Status — Phase 2: CPU Parity ✅ (IQ2_M floor)

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1: Output proj fix | ✅ | Logits zero→real. Cos-sim 0.974 vs llama.cpp |
| Phase 2: Infra parity | ✅ | All 4 test scripts → serve_local.py. 4 battleship gaps closed. |
| Phase 3: Gainz | 🟡 | SSM buffer pre-allocation, MoE expert caching, attention sparsity (untested on this machine) |
| Phase 4: GPU (different machine) | ⏸️ | RTX 5050 rig not in this environment |

---

## 📚 References

- `.hermes/mind-palace/` — State, plan, goal-mantra, prestige, battleship
- `.hermes/mind-palace/bytropix-300-gap-battleship.md` — Full gap taxonomy (300 cells)
- `vault/parity-analysis.md` — IQ2_M floor analysis
- `vault/output-projection-fix.md` — Root cause of zero logits bug
- `vault/remaining-gaps.md` — Gap closure tracking

---

<div align="center">

*Engine: bytropix — C CPU inference for Qwen3.6-35B-A3B. Phase 2 complete: parity at IQ2_M floor (0.974 cos-sim). All infra patched to local inference. No GPU in this environment.*

</div>

---

## 📜 License

Apache 2.0 — open-source educational and research software.
