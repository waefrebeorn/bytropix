# state — May 16 DA session

## Done this session
- P1a: Chunked DeltaNet (training-only, not used in inference)
- auto-embedding: token_embd.weight auto-extracted from GGUF at model load
- BOS handling: env-var gated (ADD_BOS), default off (matches add_bos_token=false)
- Model type audit: all 7 types confirmed, all supported
- Unsloth Dynamic 2.0 research completed

## Critical: Output Still Wrong
- "Hello" produces first token "Plot" (should be something coherent)
- h_last changes with multi-token prompts (model IS processing input)
- Root cause UNKNOWN after BOS fix + embedding re-extract
- Remaining suspects: Q5_K dequant, SSM recurrence, GQA, residual path

## Known Issues
- Embedding file was corrupted (extracted with buggy dequant before MoE fix)
- BOS was always added (now fixed: ADD_BOS env var)
- output_weight loading fails when token_embd loaded in-memory (OOM with blob)

## Possible Root Causes (unverified)
1. Q5_K dequant has subtle bug affecting all attention weights
2. SSM recurrence formula diverges from llama.cpp
3. GQA implementation has dimension mismatch
4. Output weight type 12 (Q4_K) dequant produces repeated values

## Next Steps (if continuing)
- Compare layer-0 hidden state against llama.cpp ref_forward
- Verify Q5_K dequant with known test vector
- Fix blob memory usage to allow token_embd + output_weight simultaneously
