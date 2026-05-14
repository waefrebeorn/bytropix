# Plan — Fix GPU Forward + Training Pipeline

## Purpose
The GPU forward and training paths are broken (zeros / wrong loss / hangs). Fix P0-P2 in order.

---

## P0 — Fix GPU Weight Loading (bench.c → gpu_load_ssm_layer produces zeros)

**Symptom:** `bench_e2e` produces ALL ZEROS for both CPU and GPU forward paths. Speedup 29x on zero is meaningless.

**Root cause hypothesis:** `bench.c` functions `gpu_load_ssm_layer()` / `gpu_load_gqa_layer()` read weights from GGUF using per-layer reopen + `gguf_find_tensor` + `gguf_read_tensor_f32`. The dequantized float values may be wrong or the tensors are at wrong GGUF offsets.

**Fix actions:**
1. Add debug print in `gpu_load_ssm_layer`: print first 8 float values of each loaded tensor
2. Also print the dimension for validation
3. Compare versus train_real's `wubu_model_init` loading path — same source?
4. Try replacing bench_e2e's forward with `wubu_model_forward_from_embd` and measure speed

**Verification:**
- `bench_e2e` outputs non-zero values for CPU and GPU
- CPU/GPU diff < 0.05 at final hidden state

---

## P1 — Fix train_backprop hang

**Symptom:** Times out at 180s during model init. Train_real loads same model in 7.7s.

**Root cause hypothesis:** Same object files, same includes. Issue may be in `wubu_tokenizer_init` not flushing stdout after finish. Or a very slow memory allocation path.

**Fix actions:**
1. Add `fflush(stdout)` after each major step
2. If still hangs: add malloc guards
3. If still hangs: build with `-O0 -g` and use GDB backtrace

**Verification:**
- `train_backprop` prints "Step 1: loss=..." within 30s

---

## P2 — Fix train_gpu GPU forward (wrong loss)

**Symptom:** CE loss 69 instead of ~12.4. GPU weight loading is broken.

**Root cause hypothesis:** `train_gpu.c` re-opens GGUF and loads weights per-layer-per-step. The `gpu_load_ssm_layer` path is broken (same as bench_e2e).

**Fix actions:**
1. Fix P0 first (GPU weight loading)
2. Rebuild train_gpu with fixed bench.c
3. Compare CE loss to train_real's 12.66

**Verification:**
- train_gpu CE loss ≈ 12.4-12.7 (matches train_real)

---

## P3 — Verify GPU backward pass (after P0-P2)

**Symptom:** Not started yet.

**After P0-P2:** GPU forward gives correct CE. Add GPU gradient computation. Compare to CPU gradient.

---

## Dependency Graph

```
P0 (GPU weight loading → zeros)
├── unlocks P2 (train_gpu wrong loss)
├── unlocks bench_e2e (was giving false PASS)
P1 (train_backprop hang) — independent of P0
```

P0 blocks the most. P1 is independent. Both can run in parallel.
