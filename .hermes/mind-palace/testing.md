# WuBuText AI — Testing Protocol (May 21, 2026)

## IMPORTANT STATE UPDATE
**CPU inference is FIXED.** `./gen_text` produces coherent text (verified: "Paris is the capital of France..."). CPU-only cos-sim vs llama.cpp reference: 0.9968 overall. All known bugs fixed.

**GPU text inference is NET-NEGATIVE** — not broken, but slower than CPU. GPU MoE 0.9888 cos-sim is a FUNDAMENTAL code-path difference (DA v13), not a fixable bug.

**GPU vision encoder WORKS** — 0.52s GPU vs 63.7s CPU (122x). Full vision→text pipeline verified at 15.7s.

---

## Quick Start
```bash
# CPU inference (works correctly)
./gen_text "The capital of France is" 20

# Compare with reference (CPU)
./test_full_model

# Vision test
./test_vision_real <mmproj.gguf> <pixels.bin>

# Per-layer cos-sim verification
DUMP_LAYER_DIR=/tmp/ref ./ref_dumper model.gguf "prompt" 0
DUMP_LAYER_DIR=/tmp/our ./gen_text "prompt" 0
./layer_cos_sim /tmp/ref /tmp/our 40
```

## Current Test Categories

### Correctness Tests (All CPU path)
| Test | Command | What It Checks | Status |
|------|---------|----------------|--------|
| Full model cs | `./test_full_model` | Cos-sim vs llama.cpp reference | ✅ 0.9968 |
| Layer-by-layer | `./layer_cos_sim /ref /our 40` | Each layer's hidden state vs ref | ✅ 0.9968+ |
| Per-expert MoE | `./compare_moe_expert` | Single expert cos-sim vs CPU | ✅ ~0.9888 (fundamental) |
| Output projection | `./compare_outw` | Final logit comparison | ✅ ~0.999 |
| Text generation | `./gen_text "prompt" 20` | Coherent text output | ✅ Verified |

### GPU Tests
| Test | Command | What It Checks | Status |
|------|---------|----------------|--------|
| GPU vision | `./test_vision_real` with GPU_SUPPORT | ViT layer cos-sim vs CPU | ✅ 122x faster, NaN=0 |
| GPU hybrid | `GPU=1 FORCE_CPU_MOE=1 ./gen_text_gpu` | Full hybrid pipeline | ✅ Coherent text at 5.5 tok/s |
| GPU GQA prefill | internal test | Batched C=N prefill | ✅ Fixed |
| GPU quant matmul | internal test | Q5_K/Q6_K batched | ✅ Fixed |

### MTP Tests
| Test | Command | What It Checks | Status |
|------|---------|----------------|--------|
| MTP load | `./test_mtp_load` | MTP model tensor loading | ✅ blk.40 + nextn.* |
| MTP draft | `./test_mtp_draft` | Draft token generation | ✅ Working |
| MTP decode | `MTP=1 ./gen_text_mtp "prompt" 10` | Full speculative decode | ✅ 8.5 tok/s |

### Performance Tests
| Test | Command | What It Checks | Status |
|------|---------|----------------|--------|
| Decode speed | `PROFILE=1 ./gen_text "hello" 1` | Per-layer timing | ✅ CPU: 8.9 tok/s |
| Prefill speed | `./gen_text "long prompt" 0` | Prefill throughput | ✅ 17.8 tok/s |
| GPU vision speed | GPU vision pipeline | End-to-end timing | ✅ 15.7s total |

## Reference Data Protocol
```bash
# Generate reference data via libllama.so (NOT llama-cli)
./ref_dumper /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "prompt" 0

# Environment variables:
DUMP_LAYER_DIR=/tmp/ref_layers       # Per-layer hidden state dumps
DUMP_INTERMEDIATE_DIR=/tmp/ref_interm  # Per-layer Q/K/V/attention dumps
REF_LOGITS_PATH=/tmp/ref_logits.bin   # Final logits
REF_HIDDEN_PATH=/tmp/ref_hidden.bin    # Final hidden state

# Reference MTP
./ref_dumper_mtp /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf "prompt"
```

## Known Testing Limitations

- **GPU MoE cos-sim 0.9888** — NOT a bug, fundamental code-path diff (DA v13)
- **No 256K stress test** — max tested: ~65K tokens
- **No GPU memory leak detection** — tool exists but not integrated
- **No automated CI** — manual test runs
- **MTP acceptance 4%** — low due to quantized IQ2_M MTP head weights
- **GPU thermal throttling** — GPU init heats CPU, skews CPU benchmark timing

## What Tests DON'T Catch (Future Work)

- **Precision impact on real tasks**: Does 0.9968 cos-sim affect MMLU/GSM8K scores?
- **KV cache garbage at 256K**: SSD offload path not tested
- **Tokenizer correctness**: Not compared against GGUF-native tokenizer
- **GPU performance under sustained load**: Thermal throttling after 5+ minutes
