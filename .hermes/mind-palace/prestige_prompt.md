═══ BYTROPIX — PRESTIGE RESUME (May 17 v4 — MoE expert layout bug) ═══
Path: /home/wubu/bytropix | HW: RTX 5050 6.4GB, -arch=sm_120
Build: rm -f src/cuda_kernels.o infer_text; make infer_text
Model: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf

=== COMPLETED ===
1. RoPE pairing: split-half→adjacent-pair (CPU+GPU) — FIXED
2. MRoPE sections [11,11,10,0]: frequency restart per section — FIXED
3. SSM delta net math: code structure matches llama.cpp — CLEARED
4. Output projection TRANSPOSE: weight[j*2048+k]→weight[k*248320+j] — FIXED
   (3 places: infer_text.c×2, wubu_model.c)
5. Reference extraction: dump_llama_logits now dumps logits + hidden — BUILT
6. TGT/manifold audit: no contamination in inference path — CLEARED

=== ROOT CAUSE FOUND (DA Audit May 17) ===
MoE expert tensor layout IS interleaved per expert, not contiguous:
  blk.0.ffn_gate_exps.weight dims = [2048, 512, 256]
  GGUF dims: last dim = fastest varying = expert index
  Each IQ2_XXS block (66 bytes, 256 values) = 1 value per expert at ONE position
  dequant_one_expert_contiguous reads eid*raw_per_exp — WRONG!
  Must: dequant each block → extract block_vals[eid] → store at position b

=== REMAINING ===
1. Fix MoE expert extraction stride (highest priority)
2. Rebuild and verify
3. Layer-by-layer dump if still incorrect after MoE fix

=== DA AUDIT — Questions for Future Self ===
Q1: "SSM formulas were verified correct" — Was this a TRUE verification?
  A1: ❌ "Reasonable values" in debug dumps ≠ verification. Need element-by-element
      comparison vs llama.cpp for at least layer 0 SSM output.

Q2: "GQA algorithm verified correct" — What was actually tested?
  A2: ⚠️ wubu_gqa_forward was tested vs numpy, but infer_text.c uses INLINE GQA.
      The inline code was NOT separately verified, though structurally identical.

Q3: "Weight layouts all verified" — What did we miss?
  A3: ❌ Missed MoE expert interleaving. 3D tensor layout ≠ simple 2D.
      The dequant stride assumes contiguous-per-expert but data is interleaved.

Q4: What else might be wrong that we haven't checked?
  A4: - Shared expert dequant types (check dims of ffn_gate_shexp etc.)
      - ssm_out weight access pattern (should be [VALUE_DIM, D_MODEL])
      - GPU RoPE 0.25x factor
      - ssm_a values (verified negative, range -72 to -0.019 — correct for DeltaNet)
