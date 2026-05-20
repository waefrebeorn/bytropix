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
| `api_server` | `api_server.c` | OpenAI-compatible API server | `make api_server` |

## API Server

`api_server` wraps `infer_text_gpu` as an OpenAI-compatible HTTP API for educational and research purposes.

```bash
# Build
make api_server

# Run with sandbox mode (no GPU needed)
./api_server --sandbox --port 8080

# Test
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":50}'

# Production mode (with real inference binary)
INFER_BIN=./infer_text_gpu MODEL_PATH=/path/to/model.gguf ./api_server --port 8080 --auth sk-your-key --tls cert.pem key.pem
```

### Security & Legal Notice

This server is provided as **open-source educational scaffolding**. It is not a
production-ready deployment — operators are responsible for:

- **API key management** — use `--auth` or `API_AUTH_KEY` env var
- **TLS encryption** — use `--tls cert.pem key.pem` for HTTPS
- **Rate limiting** — built-in limiter (60 req/min per IP)
- **Content filtering** — no built-in content moderation
- **Regulatory compliance** — GDPR, CCPA, EU AI Act compliance is operator's responsibility

THE AUTHORS ASSUME NO LIABILITY. This software is for educational and research
use only. Do not expose to the internet without proper security measures.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions |
| POST | `/v1/completions` | Text completions |


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
