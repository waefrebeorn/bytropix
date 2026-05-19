# Overnight Map — May 18, 2026 — MTP MODEL DISCOVERED. ISLAND BOY NEXT.

## Session Summary
Discovered 2nd model: Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf (753 tensors, blk.40 + nextn).
MTP head: GQA + MoE + nextn.eh_proj, nextn.enorm, nextn.hnorm, nextn.shared_head_norm.
GPU output projection works (cuBLAS) but bottleneck is MoE not output proj.

## Key Insight
Memory bandwidth (DDR5 ~50GB/s) is the bottleneck for 35B model (10.7GB/step).
Solution: "Island boy" batch processing + MTP speculative decode.

## Done
1. Discovered MTP model in /models/
2. Diffed tensor names: 20 extra tensors in blk.40
3. GPU output projection via cuBLAS (GPU=1 env var)
4. Verified: GPU works but MoE is the real bottleneck
5. Phase 5 plan: Island Boy + MTP

## Next
1. Load blk.40 + nextn tensors from MTP model
2. Implement blk.40 forward (GQA + MoE + nextn projections)
3. Batch-process tokens through each layer (B=4)
4. Implement MTP speculative decode (draft via blk.40, verify batch)
5. Accept 5-token startup lag for cache warmup

## Next Session
1. Add chat template to gen_text (Gap 7)
2. Fix shared expert sigmoid gate in wubu_moe.c (Gap 5)
3. KV cache for GQA decode (minor speedup)
4. Read Unsloth UD dynamic quant blog for IQ2_M format understanding
