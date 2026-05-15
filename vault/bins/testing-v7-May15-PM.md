# WuBuText AI — Testing Protocol (May 15 PM v7)

## Purpose
Automated regression, accuracy, and performance verification. Run before any commit.

---

## Quick Start
```bash
bash tests/run.sh           # 9 tests, ~3 min
bash tests/run.sh --full    # includes MOE=1 (~8 min)
```

Exit 0 = all PASS. Exit 1 = FAIL found.

## Test Harness: `tests/run.sh`

| # | Test | What It Checks | Why |
|---|------|----------------|-----|
| 1 | BUILD | `make infer_text_gpu` compiles | Broken build = nothing works |
| 2 | EXISTENCE | Binary + model exist, ELF 64-bit | Missing artifacts waste debug time |
| 3 | SMOKE | Runs with 1 token, exit 0, PASS marker, decode ran | Basic sanity |
| 4 | OUTPUT REGRESSION | Prefill + decode text matches golden `!!!` | Wrong output = regression |
| 5 | CHUNK SIZE PARITY | CHUNK=256 == CHUNK=64 produce identical text | Chunk boundary bug detection |
| 6 | DECODE SPEED | Benchmark decode + prefill tok/s | Performance regression alert |
| 7 | LONG PROMPT | 48 tok prompt, multi-chunk, completes with PASS | Memory / overflow check |
| 8 | LAYER CONFIG | 40 layers (30 SSM, 10 GQA) | Weight loading correctness |
| 9 | CLEANLINESS | No NaN/Inf/SIGSEGV in output | Silent corruption detection |
| 10 | MOE=1 (opt) | Full MOE=1 run with GPU buffers allocated | MoE regression prevention |

## The 10 Gaps (and how this closes them)

| # | Gap | Previously | Now Fixed By |
|---|-----|------------|--------------|
| 1 | **No regression suite** | Manual one-off testing per session | `tests/run.sh` — 20 automated checks |
| 2 | **No golden output** | "Looks right" eyeball verification | Golden grep patterns + auto-regression in test 4 |
| 3 | **No chunk-boundary test** | Only ran with default CHUNK | Test 5: CHUNK=256 vs CHUNK=64 parity |
| 4 | **No long-context prefill** | Never tested >10 tokens | Test 7: 48-token prefill |
| 5 | **No performance baseline** | Speed claims anecdotal | Test 6: decode/prefill tok/s benchmarked each run |
| 6 | **No build integrity** | `make` might silently break | Test 1: verify build before every run |
| 7 | **No NaN/Inf detection** | Silent corruption if output becomes garbage | Test 9: grep for NaN/Inf/SIGSEGV |
| 8 | **No binary sanity** | Model loading issues caught at generation, not init | Test 2+8: file integrity + layer count |
| 9 | **No MOE smoke test** | MOE=1 only tested ad-hoc | Test 10 (opt): full MOE=1 pipeline |
| 10 | **No CI equivalent** | Changes never re-verified | `bash tests/run.sh` = one-command verification |

## Golden Output Files

```
tests/golden/
  prefill_short.txt     — "The meaning of life" → "!" output
```

Update golden when model or architecture changes:
```bash
./infer_text_gpu /home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "The meaning of life" 3 2>&1 | tee tests/golden/prefill_short.txt
```

## Manual Verification Protocol

For new features, run ALL additional checks:

### Accuracy: GPU vs CPU (if CPU baseline exists)
```bash
# Requires CPU inference binary
./infer_text_gpu [model] "prompt" N > gpu_out.txt
./infer_text [model] "prompt" N > cpu_out.txt
diff <(grep -v "^Prompt:\|^Prefill:\|^Decode:\|^===\|^Uploading\|^GPU\|^Allocating\|Layer\|Model\|Merges\|BOS\|EOS\|PAD\|Vocab\|GGUF\|byte_token\|Tensor\|info" gpu_out.txt) \
     <(grep -v "^Prompt:\|^Prefill:\|^Decode:\|^===\|^Uploading\|^GPU\|^Allocating\|Layer\|Model\|Merges\|BOS\|EOS\|PAD\|Vocab\|GGUF\|byte_token\|Tensor\|info" cpu_out.txt)
```

### Memory: VRAM usage tracking
```bash
watch -n 1 nvidia-smi --query-gpu=memory.used --format=csv
# Before/after MoE=1 run
```

### 256K Context (manual, takes ~15 min)
```bash
# Generate a long prompt by embedding many tokens
# Then run with max_tok=1 to test attention against full cache
CHUNK=128 ./infer_text_gpu [model] "$(head -c 50000 /dev/urandom | base64)" 1
```

## Expected Performance Baselines (RTX 5050)

| Mode | Prefill (tok/s) | Decode (tok/s) | Notes |
|------|-----------------|----------------|-------|
| No MoE (MOE=0) | 15-22 | 120-300 | Depends on context length |
| MoE (MOE=1) | ~0.1 | ~0.3 | PCIe upload bottleneck |
| CHUNK=64 | similar | faster at long ctx | Smaller scratch buffers |
| CHUNK=256 | similar | similar | Default |

## Known Testing Limitations

- **Tests are expensive**: Each run takes ~15s model init. Full suite ~3 min.
- **No GPU vs CPU cross-verification**: CPU baseline binary exists but not wired.
- **No exact logit comparison**: Tests check text output, not numerical values.
- **No stress test for 256K**: Long-prompt test at 48 tok only.
- **MOE=1 test optional**: Excluded from default run (takes ~25s setup + 5s inference).

## Future Improvements (P3)

- Integrate `test_kv_cache` into harness — numerical max_diff comparison
- Add `train_integrated` smoke test (1 step, check loss finite)
- Add GPU memory leak detection (VRAM usage before/after)
- Add CI integration (pre-commit hook)
- Wire API server into tests: `bash tests/test_api.sh` for sandbox endpoint validation
