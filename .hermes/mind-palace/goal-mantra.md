═══ GOAL PASTE (May 16 v24 — DA AUDITED) ═══
PROJECT: bytropix — Custom Qwen3.6-35B-A3B inference engine
MODEL: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (7-type mixed quant)
STATUS: All P0-P3 infrastructure done. Output WRONG — NOT tool-call ready.

=== DA AUDIT — Survivorship Bias Stripped ===
❌ "Doug" vs "Here" — CRITICAL. Root cause UNKNOWN.
❌ Embeds from pre-extracted file were CORRUPTED (extracted with buggy dequant).
    → Fixed: auto-extract token_embd.weight from GGUF at load time.
❌ BOS: add_bos_token=false in GGUF, but bytropix added BOS anyway.
    → Fixed: ADD_BOS env var, default off.
❌ h_last IDENTICAL for "Hello" and "X" prompt — forward pass not using input??
    → Actually h_last changes for multi-token prompts.
✓ Metadata epsilon=1e-6 matches bytropix hardcoded value.
✓ All 7 types (F32, Q5_K, Q6_K, IQ2_XXS, IQ3_XXS, IQ4_XS, Q4_K) supported.

=== ROOT CAUSE STILL UNKNOWN ===
Likely culprits remaining:
1. Q5_K dequant correctness (most weights use this type)
2. Model forward pass has a hidden bug (SSM recurrence, GQA, or residual add)
3. Output weight type mismatch
4. MoE layers bypassed (MOE_LAYERS=0 on CPU) → SSM-only path may be wrong

=== AUTO-EMBEDDING IMPLEMENTED ===
- wubu_model_init auto-extracts token_embd.weight from GGUF if missing/stale
- Saves to data/qwen36_embeddings_c.bin.raw for future runs
- Falls back to file if it exists and is correct size

=== TEST ===
NOGPU=1 MOE=1 ./infer_text /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "Hello" 8 1
# Still produces "Plot" not "Here" — root cause unfixed
