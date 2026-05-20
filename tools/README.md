# `tools/` — Binaries, Tests, and Analysis Scripts

**~50 tools: generation frontends, verification harnesses, analysis scripts.**

## Core Generation

| Binary | Source | Purpose | Build |
|--------|--------|---------|-------|
| `gen_text` | `gen_text.c` | CPU-only text generation (main entry point) | `make gen_text` |
| `gen_text_gpu` | `gen_text.c` + CUDA | GPU inference (⚠️ pre-existing hang) | `make gen_text_gpu` |
| `gen_text_mtp` | `gen_text_mtp.c` | MTP speculative decode | `make gen_text_mtp` |

## Reference & Verification

| Binary | Source | Purpose |
|--------|--------|---------|
| `ref_dumper` | `ref_dumper.cpp` | Links libllama.so: per-layer + intermediate tensor dumps |
| `ref_dumper_mtp` | `ref_dumper_mtp.cpp` | MTP cross-reference (libllama.so) |
| `layer_cos_sim` | `layer_cos_sim.c` | Per-layer cosine similarity comparison |
| `compare_ggml_matmul.cpp` | `compare_ggml_matmul.cpp` | Quantized matmul vs ggml SGEMM |

## Component Tests

| Binary | Tests |
|--------|-------|
| `test_ssm` | SSM unit test vs golden vectors |
| `test_full_moe` | Full MoE forward verification |
| `test_moe_*` | MoE router, expert weights, quantization |
| `test_kv_cache` | KV cache match vs full recompute |
| `compare_*` | Quant types vs F32 SGEMM (Q4_K, Q5_K, Q6_K, IQ2_XXS, etc.) |

## Python Analysis (Phase 22)

| Script | Purpose |
|--------|---------|
| `classify_layers.py` | Classify SSM/GQA from GGUF tensor names |
| `analyze_intermediates.py` | Browse DUMP_INTERMEDIATE_DIR output |
| `analyze_l31.py` | Deep-dive into L31 GQA attention |
| `inspect_ref_intermediates.py` | Reference intermediate tensor browser |
| `unified_ssm_plan.md` | Fusion kernel design document |
| `example_rotorquant.py` | RotorQuant Givens rotation + Q4_0 demo |
| `example_turboquant.py` | TurboQuant WHT + Q4_0 demo |
| `example_hamilton_encoder.py` | Hamilton quaternion manifold demo |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `DUMP_LAYER_DIR` | Save per-layer hidden states as `.bin` files |
| `DUMP_INTERMEDIATE_DIR` | Save ALL intermediate tensors (53 types/layer) |
| `PROFILE` | Per-layer timing breakdown |
| `GQA_WINDOW` | Sliding window size for GQA attention |
| `OMP_NUM_THREADS` | OpenMP thread count |
| `REF_LOGITS_PATH` | Reference logits output path (used by ref_dumper) |

## Make Targets

```bash
make gen_text           # CPU inference binary
make gen_text_gpu       # GPU inference (with CUDA)
make gen_text_mtp       # MTP speculative decode
make ref_dumper         # Reference comparison tool
make test_ssm           # SSM unit test
make layer_cos_sim      # Cos-sim comparison tool
make all                # Build all targets (slow)
```
