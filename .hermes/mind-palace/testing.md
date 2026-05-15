# WuBuText AI — Testing Protocol (May 16 v8 — HONEST)

## IMPORTANT
**INFERENCE IS BROKEN.** Current tests check compilation + non-crash, NOT correctness.
Reference output (llama.cpp): "Here's a thinking process:" — Our output: garbage.
All test results should be interpreted as "compiles and doesn't crash" unless verified against llama.cpp.

---

## Quick Start
```bash
bash tests/run.sh           # 9 tests, ~3 min
bash tests/run.sh --full    # includes MOE=1 (~8 min)
```
Exit 0 = all PASS (compilation + non-crash). Exit 1 = FAIL found.

## Test Harness: `tests/run.sh`

| # | Test | What It Checks | Limitation |
|---|------|----------------|------------|
| 1 | BUILD | `make infer_text_gpu` compiles | Doesn't verify correctness |
| 2 | EXISTENCE | Binary + model exist, ELF 64-bit | Basic sanity only |
| 3 | SMOKE | Runs with 1 token, exit 0, PASS marker | Doesn't check output content |
| 4 | OUTPUT REGRESSION | Prefill + decode text matches golden `!!!` | Golden is from pre-fix version |
| 5 | CHUNK SIZE PARITY | CHUNK=256 == CHUNK=64 produce identical text | Both produce garbage |
| 6 | DECODE SPEED | Benchmark decode + prefill tok/s | Speed real, output wrong |
| 7 | LONG PROMPT | 48 tok prompt, multi-chunk, completes with PASS | Memory check only |
| 8 | LAYER CONFIG | 40 layers (30 SSM, 10 GQA) | Weight loading check |
| 9 | CLEANLINESS | No NaN/Inf/SIGSEGV in output | No garbage detection |
| 10 | MOE=1 (opt) | Full MOE=1 run with GPU buffers allocated | Output not verified |

## What Tests DON'T Catch (Critical Gaps)

- **No reference comparison**: Never compares output against llama.cpp
- **No hidden state comparison**: Layer-by-layer numerical comparison missing
- **No tokenizer verification**: Custom tokenizer never checked against GGUF-native
- **No logit validation**: Never checks if output logits match reference
- **GPU vs CPU mismatch**: Different architectures (SSM, RoPE, gates) — no cross-verification

## Golden Output Files — DEPRECATED
```
tests/golden/prefill_short.txt     — "The meaning of life" → "!" output
```
This golden is from a broken inference. Do NOT use as correctness reference.
Use llama.cpp output as ground truth instead.

## Verification Protocol (Manual — until P0 fixed)

### Compare vs llama.cpp
```bash
~/llama.cpp/build/bin/llama-cli -m /home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf -p "The capital of France is" -n 20 --temp 0.0
./infer_text_gpu /home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "The capital of France is" 20 2>&1
# Compare outputs — they must MATCH for correctness
```

### Layer-by-layer hidden state comparison (TODO)
This is the gold standard fix for P0 — compare SSM hidden states, GQA logits, MoE output against reference.

## What Actually Works (Verified)
- **test_kv_cache**: `make test_kv_cache` — max_diff=0.00 vs full recompute ✅
- **test_256k**: MoE router O(T) scaling to 65K ✅
- **API server**: `bash tests/test_api.sh` — 14 sandbox tests pass ✅
- **Compilation**: All binaries build cleanly ✅

## Known Testing Limitations

- Tests check compilation + non-crash, NOT correctness
- No GPU vs CPU cross-verification
- No exact logit comparison (text output only)
- No 256K stress test (48 tok max)
- MOE=1 test optional (not default)

## Future Improvements

- Integrate `test_kv_cache` into harness (numerical max_diff)
- Add llama.cpp reference comparison
- Add hidden state layer-by-layer comparison
- Add `train_integrated` CE reference baseline check
- Add GPU memory leak detection
