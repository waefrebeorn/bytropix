═══ WUBUTEXT AI — PRESTIGE RESUME (May 17 v3 — TGT/manifold cleared) ═══
Path: /home/wubu/bytropix | Branch: master
HW: RTX 5050 6.4GB, -arch=sm_120
Build: rm -f src/cuda_kernels.o infer_text; make infer_text | Models: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf

=== COMPLETED ===
1. RoPE pairing: split-half→adjacent-pair (CPU+GPU) — FIXED
2. MRoPE sections [11,11,10,0]: frequency restart per section — FIXED
3. SSM delta net math: verified identical to llama.cpp — CLEARED
4. TGT/manifold audit: no contamination in inference path — CLEARED
5. ssm_a values: all negative, range -72 to -0.019, correct for DeltaNet — VERIFIED
6. Weight layouts: all 14 components match GGUF ne[0]-innermost — VERIFIED
7. Layer mapping: (idx+1)%4==0→GQA matches llama-simple output — CONFIRMED

=== REMAINING ===
Hidden state cos-sim = 0.0167 vs reference. Bug is at a level below code reading.
Next: Layer-by-layer dump comparison to find first divergent layer.

=== BUILD ===
rm -f src/cuda_kernels.o infer_text; make infer_text
