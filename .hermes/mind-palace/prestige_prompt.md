═══ WUBUTEXT AI — PRESTIGE RESUME (May 16 v19 — DA RECERTIFIED) ═══
Path: /home/wubu/bytropix | Branch: master
HW: RTX 5050 6.4GB, -arch=sm_120
Build: make infer_text | Models: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf

=== RULE ===
Output must match llama.cpp token-for-token. Root cause unknown after 2 fix sessions.

=== DA-AUDITED STATUS (May 16 v3) ===

INFRASTRUCTURE (verified structurally vs llama.cpp):
✓ infer_text.c weight indexing: CORRECT (i + j*D_MODEL pattern everywhere)
✓ RoPE in both prefill + decode: CORRECT
✓ SSM recurrence steps: CORRECT (qkv→β/α→gate→conv→L2norm→delta→norm→out)
✓ MoE router + shared expert: CORRECT structure
✓ BOS handling: ADD_BOS env var default off = correct

BUGS FOUND:
❌ wubu_gqa_forward() weight indexing: WRONG (i*cols+j not i+j*D_MODEL)
→ Dead code for inference (infer_text.c uses inline GQA)

ROOT CAUSE: narrowed to 5 suspects
1. Q5_K dequant (181 tensors, most common type)
2. Output weight type 12 Q4_K dequant
3. SSM Q scaling factor 1/sqrt(128)
4. RMSNorm epsilon
5. TGT wrapping in GQA softmax

DA FINDINGS:
- All P0-P3 items completed at "compiles" level only. ZERO verified against llama.
- System works end-to-end (tokens in → tokens out) but output wrong.
- Model IS processing input (h_last changes with prompt length).
- No NaN/inf in any layer outputs (verified with NaN guards).

NEXT STEPS (ordered):
1. Write Q5_K dequant test vector → compare vs llama.cpp dequant of same block
2. Write Q4_K dequant test vector → verify output.weight
3. Compare layer-0 h_last vs llama.cpp ref_forward to binary-search divergence point
4. Fix wubu_gqa_forward() indexing
5. Remove TGT wrapping test

=== PREVIOUS FIXES (none verified against llama.cpp output) ===
- Shared expert gate (P0-a): loaded + sigmoid applied
- MoE contiguous dequant: fixed expert stride
- MOE=1: default changed to 1
- MAX_LAYERS clamp: 0→n_layers
- Auto-embedding: token_embd auto-extracted from GGUF
