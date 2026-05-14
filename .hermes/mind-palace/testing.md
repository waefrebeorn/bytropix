# WuBuText AI — Testing Protocol (May 14 PM)

## Infer_moe_lazy — Lazy MoE Dequant ✅
```
./infer_moe_lazy [model.gguf] [layer] [T]
```
**PASS:** Dequant time < 1s (vs 3.1s full), output match max_diff=0.00 vs full dequant.

## Infer_unified — 40-Layer Forward ✅
```
./infer_unified [model.gguf] [T]
```
**PASS:** Prints per-layer timing, no segfault. NaN expected from random inputs on GQA layers (pre-existing).

## Test_kv_cache — KV Cache Test ✅
```
./test_kv_cache [model.gguf]
```
**PASS:** max_diff=0.00 at all decode steps. Cache time < refull time.

## Infer_vision_gpu — GPU Vision ✅
```
./infer_vision_gpu [model.gguf] [image.png]
```
**PASS:** < 1s for 256×256. Output has non-zero values.

## Infer_poincare — GPU Poincaré SSM ✅
```
./infer_poincare
```
**PASS:** > 2000 tok/s. Non-zero output.

## Train_real — CPU Training ✅
```
./train_real [model.gguf] [corpus.bin]
```
**PASS:** CE loss ~12.4-12.7, 0.2 tok/s, no NaN.

## Test_moe — MoE Forward ✅
```
./test_moe
```
**PASS:** Output range < |1.0|, NaN=0.

## Bench_e2e — GPU Benchmark ⛔
```
PATH="/usr/local/cuda/bin:$PATH" ./bench_e2e
```
**FAIL:** All zeros output. GPU weight loading broken.

## Train_gpu — GPU Training ⛔
**FAIL:** CE loss 69 vs expected 12.66.

## Train_backprop — Gradient Training ⛔
**FAIL:** Hangs at 180s model init.
