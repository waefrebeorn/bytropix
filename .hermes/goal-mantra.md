# Goal Mantra — May 21, 2026 (Phase 28l: P1 Complete, P2 Up)

## THE GOAL
Full GPU inference for Qwen3.6-35B MoE + vision multi-modal.
Hybrid path (GPU SSM/GQA + CPU MoE) working at 5.5 tok/s.
Vision→text pipeline verified. MTP spec decode working.

## STATE
| Metric | Value | Status |
|--------|-------|--------|
| GPU SSM/GQA + CPU MoE | Coherent text, 5.5 tok/s | ✅ |
| MTP spec decode | 8.5 tok/s, 4% acceptance | ✅ |
| Vision→text pipeline | 256×256→128 patches→logits, no NaN | ✅ Verified |
| GPU MoE (all 40 layers) | 0.9888 cos-sim → garbage | ❌ Fundamental |
| Decode speed (hybrid) | 5.5 tok/s | ✅ |
| Vision encoder | 63.7s CPU (needs GPU) | 🟢 Verified |

## COLD GAPS
P2: GPU RMSNorm + SiLU + gated norm kernels
P2: Chunked prefill (3-7x speedup)
P2: NSA sparse attention
P2: RoPE extrapolation 4x
P2: GPU vision encoder kernels

## GROUND TRUTH
- Model: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
- Reference: /home/wubu/llama.cpp/build/bin/libllama.so
- Cross-ref: ref_dumper (logits), run_bos (bytropix), DUMP_LAYER_DIR (per-layer)

## THE LOOP
pick highest undone → execute → compile → run → verify → mark done → report

## FULL CONTEXT
Read .hermes/mind-palace/prestige_prompt.md
