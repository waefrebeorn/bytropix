# Overnight Map — Phase 28e: Q6_K Dequant Fixed, GPU SSM Still Diverging

**Active repo:** /home/wubu/bytropix/
**Model:** Qwen3.6-35B-A3B-UD-IQ2_M.gguf (qwen35moe arch)
**Vision model:** /mnt/wslg/distro/models/qwen3.6-35b-mmproj-F16.gguf
**Current state:** Q6_K dequant FIXED ✅. GPU SSM cos-sim -0.66 vs CPU ❌. Vision encoder ported (384 LoC).

## Verifiable Facts (DO NOT RE-DERIVE)
- Last commit: `c07cf14` — Q6_K dequant offset fix
- CPU SSM path matches llama at cos-sim 0.994 (proven)
- GPU SSM produces anti-correlated output (cos-sim -0.66) — suspect state management
- Remote at `4dc985e` — 8 local commits behind
- gen_text_gpu binary exists (May 20, 1.6MB)
- gen_text CPU build broken (GPU symbols without .cu)
- Vision encoder exists: 27-layer 3D ViT, mmproj→2048, untested
- F32 waste was a DEAD CLAIM — already removed in a032a8f
- Memory leak was a FALSE POSITIVE — free() was correct
- Column-major kernel was CORRECT layout for GGUF

## Workstreams (pick one)
**A [P0] Fix GPU SSM divergence** — highest impact. GPU has working fused kernels but state management bug. Trace recurrence and conv state across layers/steps. Target: cos-sim > 0.99.

**B [P1] Fix CPU build + push** — gen_text fails at link. 8 critical GPU fixes local-only (no backup).

**C [P2] Vision integration** — build test_vision_real, verify E2E pipeline, wire multi-modal.

## Data You Should Not Re-Derive
- Q6_K dequant: fixed `d*sc*(v6-32)` not `d*sc*v6 - 32` (commit c07cf14)
- Column-major is CORRECT for GGUF output-major weight layout
- F32 waste: already `#if 0`'d in wubu_model_gpu.cu
- Vision encoder dimensions: V_HIDDEN=1152, V_OUT_HIDDEN=2048, 27 layers, temp_patch=2
- mmproj: [4608, 4608] → GELU → [4608, 2048] — maps 2 merged image tokens to text space

## Fallback
If debugging stalls, clean up: fix CPU build, push commits, write DA v11 document.
