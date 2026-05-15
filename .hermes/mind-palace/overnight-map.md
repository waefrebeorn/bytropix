# WuBuText AI — Overnight Navigation Map (May 16 v7 — HONEST)

## Where We Are

**HARD TRUTH: Inference is BROKEN.** All binaries produce garbage.
Only 2/8 binaries verified: `test_kv_cache` (cache match) and `test_256k` (MoE router only).
6/15 math components forward-only — no gradient flow for backward passes.

**DA v5 completed (May 16):** Full meta audit of all 8 binaries, 15 math components, 4 models. No survivorship bias.

### Verified Claims
- ✅ test_kv_cache: KV cache matches full recompute (max_diff=0.00)
- ✅ test_256k: MoE router O(T) scaling to 65K
- ✅ llvm.cpp reference: BUILT at ~/llama.cpp/build/bin/llama-cli
- ✅ NaN root cause FIXED: MoE weight interleaving (IQ2_XXS block layout)
- ✅ Per-expert dequant: 3.9ms/expert, 177s→11s/step
- ✅ API server: 14 sandbox tests pass
- ✅ SGEMM ldC bug FIXED (was all-zero logits)
- ✅ All 6 flag combos verified individually + combined, 0 NaN
- ✅ Tailslayer: 5 files analyzed, pattern map complete
- ✅ Paper audit: 14 discrepancies catalogued

### Known Issues

| Issue | Severity | Status |
|-------|----------|--------|
| ALL INFERENCE BINARIES PRODUCE GARBAGE | **CRITICAL P0** | Root cause unknown |
| ~11s/step GPU compute (40 layers) | Performance | GPU forward, not dequant bottleneck |
| PGA loss jumps 21.6→69 | Numeric | Pre-existing LR issue |
| CPU output projection ~2s/token | Performance | O(N·V·D) for V=248320 |
| CONV_DIM=8192 vs config 1536 | Possible bug | Needs code audit |
| MRoPE 3D not implemented | Correctness | P2 |
| MTP head missing | Feature | P3 |
| Q5_K dequant fix impact unverified | Correctness | Fix may be wrong |

### Build Command
```bash
cd /home/wubu/bytropix && make infer_text_gpu
```

### Reference
```bash
cd ~/llama.cpp/build/bin && ./llama-cli -m /home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf -p "The capital of France is" -n 20 --temp 0.0
```

### Priority for Next Session
1. **P0: Fix inference** — Compare hidden states layer-by-layer vs llama.cpp
2. P1: Verify train_integrated CE reference baseline
3. P2: Tailslayer spec-decode, MRoPE, sparse attention

Archived: `overnight-map-v6-May15-PM.md` in `vault/bins/`.
