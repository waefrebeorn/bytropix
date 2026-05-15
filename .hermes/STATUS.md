# WuBuText AI — State (May 16 PM v4 — HONEST)

## Ground Truth
**Inference is BROKEN.** All inference binaries produce garbage output.
llama.cpp reference: "Here's a thinking process:" — Us: garbage.

Only 2/8 binaries verified: test_kv_cache (cache match) and test_256k (MoE router only).

## What Works
- test_kv_cache: KV cache matches full recompute (max_diff=0.00) ✅
- test_256k: MoE router O(T) scaling to 65K ✅
- API server sandbox: 14 tests pass ✅
- llama.cpp reference: BUILT at ~/llama.cpp/build/bin/llama-cli ✅
- Individual component kernels: compile and run ✅
- NaN root cause: FIXED (MoE weight interleaving) ✅
- Training: 11s/step (16× improvement), 0 NaN ✅

## What's Broken (P0)
- ALL inference binaries: infer_text_gpu (245 tok/s but garbage)
- MOE=1 also garbage (not just MOE=0 with no FFN)
- train_integrated CE=12.42: no reference baseline
- 6/15 math components forward-only (no gradient flow)
- Q5_K dequant fix impact unverified

## Reference
```bash
~/llama.cpp/build/bin/llama-cli -m /home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf -p "The capital of France is" -n 20 --temp 0.0
```

## Priorities
P0 — Fix inference (compare vs llama.cpp layer by layer)
P1 — Verify components against reference
P2 — Hyperbolic backward passes
P3 — GPU acceleration, tailslayer, 256K
