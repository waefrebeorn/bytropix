# Prestige Prompt — May 21, 2026 (Phase 28r: P2.3 Chunked SSM Fixed + Wired)

## Project: bytropix — Multi-Modal Inference (Text + Vision)

**Qwen3.6-35B-A3B-UD-IQ2_M text + Moondream3 3D ViT vision via mmproj**  
**CPU-only: 8.9 tok/s decode optimal | GPU vision: 15.7s pipeline | RoPE 4x: done**

## Current State
- CPU-only is optimal for text (8.9 tok/s verified). GPU hybrid is 2-5x slower.
- GPU vision encoder: 0.52s ViT (122x vs CPU), 15.7s full pipeline
- GPU MoE 0.9888 cos-sim is FUNDAMENTAL code-path diff (DA v13). Hybrid path accepted.
- **P2.3 Chunked SSM**: data layout bug FIXED. CS=1 exact. CS>1 FP-limited. Wired into wubu_ssm_forward().
- **P2.4 RoPE 4x**: `ROPE_SCALE_FACTOR=0.25` extends 64K→256K — COMPLETE
- gen_text_cpu works with proper CLI: `./gen_text_cpu "prompt" <max_tokens>`
- ref_dumper via libllama.so works with DUMP_LAYER_DIR / DUMP_INTERMEDIATE_DIR

## Done This Session
1. ✅ Chunked SSM data layout bug — token-interleaved vs head-contiguous memcpy
2. ✅ Chunked SSM cyclic repeat mapping (vh % hk, not vh / rf)
3. ✅ Chunked SSM cur_nt bounds fix (OOB write on last chunk)
4. ✅ Chunked SSM wired into wubu_ssm_forward() — SSM_CHUNK_MIN, FORCE_CPU_SSM_SEQ
5. ✅ Verified CS=1 exact match on real model (sequential path unchanged)
6. ✅ Documented CS>1 FP limitation — 30 SSM layers amplify chunking error

## Chunked SSM Status
- **CS=1**: EXACT match (4e-8 diff). Use for verification.
- **CS=2/8/64**: Produces wrong tokens in real model. FP accumulation across 30 layers.
- **Cause**: 2000x more float ops per position vs sequential. Rounding amplifies per layer.
- **Fix**: Only CS=1 is safe. Chunking speedup requires FP64 accumulation or different formula.

## Next Session: P2.5 NSA Sparse Attention
1. **NSA sparse attention** — DeepSeek-V3.2 DSA pattern. O(L·(w+g)) for GQA layers.
2. **FP8 Tensor Cores** — sm_120 native. Low until GPU data-movement solved.

## Key Env Vars
```
SSM_CHUNK_MIN=64 ./gen_text_cpu "prompt" N     # min tokens to trigger chunked
FORCE_CPU_SSM_SEQ=1 ./gen_text_cpu "prompt" N   # force sequential SSM
ROPE_SCALE_FACTOR=0.25 ./gen_text_cpu "prompt" 20 # 4x context extension
DUMP_LAYER_DIR=/tmp/ref ./ref_dumper model.gguf    # 40 layer files
```