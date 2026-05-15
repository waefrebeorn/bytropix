# WuBuText AI — Plan (May 16 AM v11)

## Purpose
Serve Qwen3.6-35B-A3B-UD-IQ2_M as a real API. Fix remaining inference bugs, add HTTP server, sandbox for security testing, add proper sampling params. Compare against llama.cpp as reference.

---

## Critical Bugs Found & Fixed

| Bug | Status | Impact |
|-----|--------|--------|
| SGEMM ldC=vocab_size instead of ldC=N | ✅ Fixed | All-zero logits |
| Q5_K dequant high-bit byte indexing | ✅ Fixed (in source) | Block-level constant values |
| BOS not prepended | ✅ Fixed | Wrong first-token predictions |
| RoPE missing from CPU GQA path | ✅ Fixed | CPU/GPU divergence |
| No temperature/top-k/top-p sampling | ✅ Fixed | Only greedy available |
| CPU GQA gate was applied ✓ | ✅ Verified | Not a bug after all |
| `MOE=0` = NO FFN (model is MoE-only) | 🔴 Unfixed | **Root cause of garbage output with MOE=0** |

## Remaining Issues

### P0 — Model Produces Garbage (Even with MOE=1)
With MoE on (`MOE=1`), output is still wrong: "The capital of France isiscInset了下去idesiby客的我们都会论usher..."
Our output vs llama.cpp reference: "Here's a thinking process:"
Causes could be:
- **MoE expert dequantization bug** (IQ2_XXS, IQ3_XXS tensors use separate dequant functions)
- **SSM forward pass bug** (Gated DeltaNet implementation doesn't match reference)
- **Tokenizer mismatch** (custom tokenizer vs llama.cpp's GGUF-native tokenizer)
- **Shared expert not routed correctly** (has `ffn_gate_inp_shexp.weight` bias)
- **Q5_K dequant still wrong** (need to compare float output against llama.cpp)

### P0 — llama.cpp Reference Comparison
Build `~/llama.cpp/build/bin/llama-cli` ✅ Done. Uses GGUF-native tokenizer.
Need to extract hidden states at each layer from llama.cpp and compare to ours.

### P0 — JSON API Server
- POST /completions endpoint
- POST /chat/completions with OpenAI-compatible format
- SSE streaming
- Chat template: `<|im_start|>system...<|im_end|><|im_start|>user...<|im_end|><|im_start|>assistant`
- Rate limiting, error handling, model loading
- Bind to configurable port (default 8080)

### P1 — Sandboxed Testing Environment
- Dedicated test directory: `tests/sandbox/`
- Fake network interface for security testing
- Mock user sessions with fake API keys
- Rate limit testing, malformed request fuzzing
- Load testing with concurrent requests
- Memory leak detection per request

### P1 — CPU GQA Decode Path RoPE
The per-token decode path in infer_text.c also needs RoPE (was only added to prefill path).

### P1 — Update Test Suite
- New golden outputs based on working model
- CPU vs GPU output parity test
- Sampling reproducibility test (same seed = same output)
- Chat template integration test
- API server integration test (curl sanity check)

### P2 — GPU MoE On-Device Dequant
Current PCIe upload bottleneck: 67MB/expert × 8 × 40 layers = 21GB/token.
Need GPU-side dequant (IQ2_XXS → BF16 on GPU).

### P2 — MTP Speculative Decode
1 MTP head available in model weights. Use for speculative decoding speedup.

---

## 256K Context Roadmap

| Step | What | Status |
|------|------|--------|
| 1 | GQA KV cache (append-only) | ✅ Done |
| 2 | SSM state carry | ✅ Done |
| 3 | Lazy MoE cache | ✅ Done |
| 4 | GPU forward for GQA/SSM decode | ✅ Kernels exist |
| 5 | Verify 256K forward pass | ⬜ Not yet |
| 6 | Single-token generation at 256K | ⬜ Not yet |
| 7 | Tailslayer spec decode | ⬜ Not yet |
| 8 | Coherence test with MoE @ 256K | ⬜ Not yet |

## Files Referenced

| File | Purpose |
|------|---------|
| `src/gguf_reader.c` | Q5_K, IQ2_XXS, IQ3_XXS, Q6_K dequantization |
| `tools/infer_text.c` | CPU inference (prefill + decode) |
| `tools/infer_text_gpu.c` | GPU-accelerated inference v5 |
|| `tools/serve.py` | HTTP API server (sandbox + production) | ✅ Built |
|| `tests/test_api.sh` | API sandbox test suite (14 tests) | ✅ Built |
| `src/wubu_ssm.c` | SSM/Gated DeltaNet forward/backward |
| `src/wubu_moe.c` | MoE router + expert computation |
| `src/wubu_tokenizer.c` | BPE tokenizer |
| `tests/run.sh` | Test harness |
| `~/llama.cpp/` | Reference implementation (for comparison) |
| `vault/unsloth-quantization-format.md` | UD GGUF format documentation |
