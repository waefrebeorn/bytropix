═══ WUBUTEXT AI — PRESTIGE RESUME (May 16 v18 — DA AUDITED) ═══
Path: /home/wubu/bytropix | Branch: master
HW: RTX 5050 6.4GB, -arch=sm_120
Build: make infer_text | Models: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf

=== RULE ===
ALL GOALS MUST BE FINISHED. Output must match llama.cpp token-for-token.
Tool-call parity: function names start from token 1. If token 1 is wrong, tool calls fail.

=== COMPLETED (verified) ===
✅ Auto-embedding: token_embd.weight extracted from GGUF at load time
✅ BOS handling: ADD_BOS env var, default=off (matches add_bos_token=false)
✅ Model type audit: 7 types, all supported
✅ P1a: Chunked DeltaNet (training-only)
✅ Unsloth Dynamic 2.0 research

=== COMPLETED (unverified) ===
❓ P0-a: Shared expert gate (compiles only)
❓ P1c: Single-pass top-k (compiles only)
❓ P2a: Warp CUDA scan (compiles only)
❓ P3a: IQ2 on-the-fly dot (4/4 tests pass)
❓ TF32 math mode, block 512, OMP loops

=== CRITICAL BUG ===
❌ Output: "Hello" → "Plot" not "Here" or any sensible continuation.
❌ Root cause: UNKNOWN after fixing:
   - Embedding file (re-extracted from GGUF)
   - BOS handling (disabled)
   - Epsilon verified (1e-6)
❌ Remaining suspects (order of likelihood):
   1. Q5_K dequant bug — 181 tensors (most attention weights)
   2. SSM recurrence formula divergence
   3. GQA dimension/indexing bug
   4. Output weight Q4_K dequant bug

=== DA FINDINGS ===
- "Noise floor" argument is WRONG for quantized models.
  Two engines running same quantized GGUF MUST produce same output at temp=0.
- Embedding file was corrupted from earlier buggy dequant extraction.
  Fixed by re-extracting from GGUF using current gguf_read_tensor_f32.
- BOS handling mismatch: bytropix added BOS, llama.cpp doesn't (add_bos=false).
  Fixed with env var.
- Output still wrong after both fixes → deeper bug in model computation.

=== TO-DO ===
1. Fix GGUF blob memory: free after tensor loading to allow token_embd + out_weight
2. Compare layer-0 h_last vs llama.cpp ref_forward
3. Verify Q5_K dequant against known test vectors
4. Reach parity with llama.cpp output
