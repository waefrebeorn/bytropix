# bytropix State — May 27, 2026

## Critical Fixes Applied

| Problem | Root Cause | Fix | Status |
|---------|-----------|-----|--------|
| **Garbage output** | `inference-server.py` on :8001 was a **proxy** to DeepSeek/Nous — never called local model | `tools/serve_local.py` wraps `gen_text_cpu` directly | ✅ |
| **No real tests** | Only bash scripts existed | `tests/test_inference.py` — 24 pytest tests, 1.16s | ✅ |
| **No Hermes provider** | No way for Hermes to use bytropix | `custom_providers.bytropix` in `~/.hermes/config.yaml` | ✅ |
| **Output proj zeros** | `if(0){/*cache*/}else{/*proj*/}` wrapper caused GCC `-O3` to dead-code-eliminate entire output projection. `#pragma omp parallel for` inside dead `if(0)` block confused control flow analysis. All logits were zero, model appeared to work because argmax of zeros = token 0 = "!" | Removed `if(0)` wrapper entirely, output proj runs directly. Also switched Q4_K vec_dot from AVX2 to generic (AVX2 path had bug producing zeros) | ✅ |

## What Works

- **Local inference** via `gen_text_cpu` — calls model directly (no proxy)
- **serve_local.py** — OpenAI-compatible HTTP API on port 8001
  - POST /v1/chat/completions
  - POST /v1/completions
  - GET /v1/models
  - GET /health
- **Pytest test suite** — `tests/test_inference.py`: 24 tests, tests server startup, API endpoints, thread safety, edge cases
- **Hermes integration test** — `tools/test-hermes-integration.sh`: 9 tests, verifies full server → Hermes pipeline
- **Server startup** — `tools/start-bytropix-server.sh`
- **Hermes provider** — `custom_providers.bytropix` in `~/.hermes/config.yaml`

## Limitations

- **IQ2_M model quality** — 2-bit quant at 35B produces garbled output (known quantization trade-off)
- **No model persistence** — server loads model fresh per request (subprocess overhead)
- **No streaming** — current implementation sends full response (not token-by-token)

## Active Cells

| Cell | Status | Description |
|------|--------|-------------|
| 175 | ✅ | Pytest suite: 24 tests |
| 176 | ✅ | Hermes integration: 9 tests |
| 177 | ✅ | Local model calls (not proxy) |
| 178 | ✅ | Hermes custom_providers config |
| 001-030 | 🔴 | Poincaré backward identity |
| 171 | 🟡 | test_regression.c |
| 172 | 🟡 | cos-sim validation |
| 174 | 🟡 | CI pipeline |
| 179-200 | 🟢 | Kernel tests |

## Memory

- **vault discovery:** `inference-server.py` was a proxy — never called local model. Fixed with `serve_local.py` that wraps `gen_text_cpu`.
- **vault discovery:** `serve.py` calls `./infer_text_gpu` (GPU binary) via subprocess. Not usable on CPU-only systems.
- **vault discovery:** IQ2_M (2-bit) quantization at 35B produces incoherent output — model quality limitation, not code bug.
