# Plan — May 18, 2026 — POST-FIX

## Phase 0: CORE INFERENCE FIXED ✓
### Achievements:
- GQA Q/gate interleave bug fixed ✓ (cos-sim -0.51 → 0.9968)
- MoE quantized path wired ✓ (IQ2_XXS/IQ3_XXS/IQ4_XS via blob)
- Shared expert quantized path wired ✓ (Q5_K/Q6_K)
- Per-layer dump infrastructure ✓ (modded llama.cpp + bytropix)
- Layer-by-layer comparison tooling ✓ (python3 script)

## Phase 1: DA v10 Real Priorities (May 18)

### Task 1.1: Build llama reference dumper tool [P0] ✓
Replaces llama-cli with direct libllama.so linkage for fast per-layer dumps.
- ref_dumper at /home/wubu/bytropix/ref_dumper ✓ (make ref_dumper)
- Links libllama.so directly, CUDA-backed
- Dumps 40 per-layer hidden states + logits in one call
- Verified: cos-sim 0.99696 against GPU reference

### Task 1.2: GQA RoPE implementation [P1] ✓
IMRoPE for Qwen3.6 with rope.dimension_sections=[11,11,10,0], theta=10M implemented.
- Applied to Q_norm and K_norm before attention in wubu_gqa_forward (line 1113)
- OpenMP on Q-head loop ✓
- Verified: T=1 cos-sim unchanged (0.99696), T=2 forward passes correctly

### Task 1.3: gen_text pipeline [P2] ✓
- gen_text tool built: ./gen_text [prompt] [max_tokens]
- Uses verified quantized model path (wubu_model_forward_from_embd)
- SSM state carries between decode steps
- Verified multi-token generation: "The capital of France is" → " the city of Paris..."
- Output: ~0.3 tok/s on CPU for 35B model (2.4 tok/s prefill)

### Task 1.4: Fix gen_text multi-token crash [P3] ✓
- Root cause: logits buffer was `vs` floats but forward writes `B*T*vs` for T>1
- Fixed: malloc(n_prompt * vs * sizeof(float))
- Tokenizer itself was fine

## Phase 2: Performance Optimization (May 18)

### Task 2.1: Profile and optimize decode speed [P0] ✓
- PROFILE env var added: per-layer wall-clock timing (PROFILE=1)
- MoE expert loop OpenMP: 3x speedup on MoE (44ms→15ms per layer)
- Embedding file: opened once at start, closed at end (removed fopen/fclose per decode step)
- Total decode speed: **0.3→0.6 tok/s** (2x)
- Breakdown per decode step (T=1, 16 threads):
  - MoE: 15ms avg (was 44ms) — 30% of time
  - SSM: 12ms avg — 25% of time
  - GQA: 15ms avg — 10% of time
  - Output proj: 8ms — 3% of time
  - Norms/overhead: ~33%
- Next targets: SSM quantized_matmul Q8_K pre-quantization, malloc reduction

### Task 2.2: KV cache for GQA decode [P1]
- Add K/V storage between decode steps to avoid full-attention recompute
- Currently each decode step recomputes attention for all positions
- Impact: GQA is only ~10% of time, so max 10% speedup from KV cache alone
- SKIP — low priority given current bottleneck distribution

## Phase 3: Multi-Token Generation
- Implement GQA RoPE (needed for T > 1)
- Verify KV cache correctness
- Test text generation quality
