# WuBuText AI — Entry Point

## Purpose
Build commands, hardware spec, and quick-start guide.

---

## Hardware

- **GPU:** NVIDIA RTX 5050, 6.4GB VRAM, compute 8.9 (Ada Lovelace)
- **Arch:** sm=120
- **CPU:** AMD, 16+ cores, ~0.2 tok/s for 40-layer forward

## CUDA Setup

```
nvcc at /usr/local/cuda/bin/nvcc — NOT on PATH by default
USE: PATH="/usr/local/cuda/bin:$PATH" make <target>
CUDA libs: -lcublas -lcudart -L/usr/local/cuda/lib64
```

## Build Commands

```
# Core binaries:
make train_real                     # CPU training pipeline (works ✅)
make test_fused_vs_old             # GPU fused SSM test (works ✅)
make test_tokenizer                # Tokenizer test (works ✅)
make test_moe                      # MoE forward test (works ✅)
make dump_mmproj                   # Vision projector dump (works ✅)

# Broken binaries (GPU weight loading bug):
make bench_e2e                     # ⛔ ALL ZEROS output
make train_gpu                     # ⛔ CE loss 69 vs expected 12.4
make train_backprop                # ⛔ HANGS during model init

# Debug/inspection tools:
make check_forward                 # Full model forward check
make verify_dequant                # IQ2 dequant verification
./verify_iq2s                      # IQ2_S block verification
```

## Quick Model Load

```
# train_real is the one correct forward path:
./train_real /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf data/train_data.bin
# Expect: CE loss ~12.66, logits +2.2 to +3.3, 0.2 tok/s CPU
```

## File Layout

```
src/               — Core C sources (ssm, moe, model, tokenizer, gguf reader, etc.)
include/            — Headers
tools/              — Tool binaries (train_real, bench_e2e, test_*)
data/               — Training data, embeddings
.hermes/            — Mind palace, references, presentation
/models/            — GGUF model files (symlink/wslg mount)
```
