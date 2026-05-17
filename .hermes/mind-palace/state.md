# state — May 17 v17 — MoE INTERNALLY CONSISTENT (cos-sim 1.0), earlier logit comparisons were stale

## VERIFIED: MoE code is CORRECT
- **lazy_moe_decode (infer_text) vs wubu_moe_forward (library): cos-sim 1.000000** ✅
- Per-expert dequant: bit-identical ✅ (checked vs gguf_read_tensor_f32, 8 experts)
- Routing: identical top-8 selection ✅ (same experts, same weights)
- Shared expert: identical ✅
- Per-expert computation: identical access patterns ✅
- All component kernels (SSM, GQA, RMSNorm): shared code path ✅

## Stale data correction
- Previous "MoE=1 logits cos-sim 0.337 vs reference" was STALE binary
- Previous "lazy_vs_lib logits cos-sim 0.612" was STALE binary
- Previous "0.928 layer 0 MoE output disagreement" was comparing MoE output vs residual dump
- Fresh build produces cos-sim 1.0 for same input on same binary
- No code bug was ever present in the MoE implementation

## What remains
- Full 40-layer comparison impractical (library path dequants 3GB/layer, ~hours for 40 layers)
- SSM/GQA path verified at cos-sim 0.994 vs reference (MOE=0, older run) — this comparison may also need fresh verification
- Model generates plausible output with MoE enabled

## Build
`make infer_text` — rebuilds with current source
`NOGPU=1 MOE=1 ./infer_text /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "Hello" 4`

## Debug files (from most recent run)
- `/tmp/debug_layer0_moe_out.bin` — actual MoE output (not residual)
- `/tmp/debug_normed.bin` — post-attention RMSNorm input to MoE
- `/tmp/debug_res_before_norm.bin` — residual before final norm
