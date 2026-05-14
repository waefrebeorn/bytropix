# WuBuText AI — Testing Protocol

## Purpose
How to run and verify each binary. PASS criteria for every claim.

---

## train_real — CPU Training Pipeline ✅ (Verified)

```
./train_real /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf data/train_data.bin
Timeout: 120s
```
**PASS criteria:**
- Prints "CE loss: ~12.4-12.7"
- Logits[0:8] non-zero (shows +X.XXXX values)
- 0.2 tok/s CPU throughput
- No NaN, no Inf

**FAIL signals:**
- Hangs (needs >120s)
- Loss >> 100 (broken dequant)
- Loss << 11 (too good — means data leak)

---

## test_fused_vs_old — GPU Fused SSM ✅

```
PATH="/usr/local/cuda/bin:$PATH" ./test_fused_vs_old
```
**PASS:** Output max diff < 0.04, State max diff < 0.04. cuBLAS FP artifact accepted.

---

## test_tokenizer — CJK Tokenizer ✅

```
./test_tokenizer /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "你好"
```
**PASS:** `"你好" → 1 token (109266) → "你好"` round-trip.

---

## test_moe — MoE Forward ✅

```
./test_moe
```
**PASS:** Output range < |1.0|, NaN=0, router probs sum ≈ 1.0, > 0 tok/s.

---

## dump_mmproj — Vision Projector ✅

```
./dump_mmproj /models/qwen3.6-35b-mmproj-F16.gguf
```
**PASS:** 334 tensors, 27 ViT blocks, mm.0[4608,4608] ✓, mm.2[4608,2048] ✓, no mm.1 ✓.

---

## bench_e2e — GPU Benchmark ⛔ (BROKEN — all zeros)

```
PATH="/usr/local/cuda/bin:$PATH" ./bench_e2e
```
**Current FAIL:** Both CPU and GPU output all zeros. Max val = 0.000000.
**Fix needed:** GPU weight loading in bench.c/bench.h broken.

---

## train_gpu — GPU Training ⛔ (BROKEN — wrong loss)

```
PATH="/usr/local/cuda/bin:$PATH" ./train_gpu
```
**Current FAIL:** CE loss 69 (expected ~12.4).
**Fix needed:** Same root cause as bench_e2e — GPU weight loading.

---

## train_backprop — Gradient Training ⛔ (BROKEN — hangs)

```
./train_backprop /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf data/train_data.bin
```
**Current FAIL:** Hangs at 180s during model init.
**Fix needed:** Unknown. Same source as train_real but binary behaves differently.
