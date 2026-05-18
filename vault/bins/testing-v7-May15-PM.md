# bytropix — Testing Protocol (May 18 — Phase 2 Complete)

## Purpose
Automated accuracy and performance verification before any commit.

## Quick Tests
```bash
# 1. Cos-sim vs reference (the ONE number that matters)
make test_full_moe && ./test_full_moe
# Expected: cos-sim=0.9968, all 40 layers > 0.995

# 2. Generation sanity
make gen_text && ./gen_text "The capital of France is" 16
# Expected: coherent English (e.g., "the city of Paris")

# 3. Performance profile
PROFILE=1 ./test_full_moe
# Expected: MoE ~15ms, SSM ~13ms, output ~12ms
```

## Core Metrics

| Test | What it checks | How | Exact Expectation |
|------|---------------|-----|-------------------|
| **Cos-sim** | Full 40L forward vs llama.cpp | test_full_moe | > 0.99 (actual: 0.9968) |
| **Per-layer decay** | Smooth quantization noise | test_full_moe internal | 0.9985→0.9952 monotonic |
| **Generation** | Text coherence | gen_text | English words, no garbage |
| **Decode speed** | Per-token wall clock | gen_text --Stats | > 0.5 tok/s |

## Golden Output Files

```bash
./gen_text "The capital of France is" 32 2>/dev/null
# Expected output (approximate):
#  the city of Paris. It is the capital of France.
#  <think></think>Paris is the capital of France.
#  **Note:** The above statement is
```

## Performance Baselines (CPU, 16 threads)

| Measurement | Expected | Notes |
|-------------|----------|-------|
| Decode speed | 0.5-0.7 tok/s | 40 layers, all quantized |
| Prefill speed | 1.0-1.5 tok/s | depends on prompt length |
| MoE layer | 15-20ms | 8 experts + shared |
| SSM layer | 13-45ms | L0 warmup higher |
| Output proj | 10-15ms | Q4_K matmul 2048×248320 |

## Known Testing Gaps

- T>1 cos-sim NOT verified (only T=1)
- Chat template NOT tested
- No GPU test (decode path not wired)
- No KV cache test (not implemented)
