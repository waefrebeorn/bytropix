# Plan — May 18, 2026 — POST-FIX

## Phase 0: CORE INFERENCE FIXED ✓
### Achievements:
- GQA Q/gate interleave bug fixed ✓ (cos-sim -0.51 → 0.9968)
- MoE quantized path wired ✓ (IQ2_XXS/IQ3_XXS/IQ4_XS via blob)
- Shared expert quantized path wired ✓ (Q5_K/Q6_K)
- Per-layer dump infrastructure ✓ (modded llama.cpp + bytropix)
- Layer-by-layer comparison tooling ✓ (python3 script)

## Phase 1: Push to 1:1 Parity (cos-sim > 0.999)

### Task 1.1: Identify largest per-operation error sources
Current per-layer cos-sim: 0.995-0.998. Need to find which operations contribute most.

Method: run our model with F32 fallback (clear quantized ptrs) and compare F32 vs quantized path. The F32 vs reference gap shows architecture errors. The quantized vs reference shows total gap.

### Task 1.2: Quantized_matmul precision
The Q8_K input quantization + vec_dot chain may differ from llama.cpp's internal matmul.
- Compare quantized_matmul output vs llama.cpp's ggml_mul_mat for same inputs
- If gap > 0.0001 per layer: trace to specific vec_dot implementation

### Task 1.3: IQ type vec_dot parity
The self-contained C vec_dot implementations may differ from llama.cpp's SIMD versions.
- For each type (Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ3_XXS, IQ4_XS):
  - Generate same input for both bytropix and llama.cpp reference
  - Compare dot product output
- Fix any differences found

### Task 1.4: GQA RoPE
RoPE is currently SKIPPED in GQA forward (comment says "will be implemented separately").
For T=1 this doesn't matter, but for multi-token generation RoPE is essential.
- Implement IMRoPE (64 dim, sections [11,11,10,0], theta=10M)
- Apply to Q and K before attention

## Phase 2: Performance

### Task 2.1: OpenMP verification
All hot loops should have OpenMP:
- MoE per-token loop ✓
- quantized_matmul column loop ✓
- GQA attention head loop — CHECK
- SSM recurrence head loop — CHECK

### Task 2.2: Infer_text pipeline
- Build and test full text generation (infer_text)
- Verify generated text is coherent and matches llama.cpp

## Phase 3: Multi-Token Generation
- Implement GQA RoPE (needed for T > 1)
- Verify KV cache correctness
- Test text generation quality
