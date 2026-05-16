# state — May 16 yolo session

## Done this session
- P1a: Chunked DeltaNet recurrence implemented (wubu_ssm_chunked.c)
  - Fixed tri solve: bottom-up (L^T is upper tri)
  - Fixed decay mask: lower tri (i >= j), was upper
  - Fixed KQ: LOWER_DIAG (keep diag)
  - NOTE: chunked != sequential for multi-token chunks. Different linear system. Training-only path.
- P2a: Warp-level CUDA scan (prior session, verified clean rebuild)
- P3a: On-the-fly IQ2_XXS dequant (prior session, verified clean rebuild)
- Added wubu_ssm_sequential_recurrence() for test verification

## Model Research
- Unsloth Dynamic 2.0 confirmed: per-layer mixed quantization
- Qwen3.6-35B-A3B-UD-IQ2_M actual types:
  - F32 (361): norms, biases, SSM params
  - Q5_K (181): attention QKV/gate/output, shared FFN
  - Q6_K (70): SSM output projection, shared expert down
  - IQ2_XXS (80): MoE expert gate/up weights
  - IQ3_XXS (37): MoE expert down weights (most layers)
  - IQ4_XS (3): MoE expert down, last 3 layers (34,38,39) — higher precision
  - Q4_K (1): output.weight (lm_head)
- All 7 types supported in gguf_reader

## Performance
- CPU: prefill 4.2s, decode 1 tok/s (MOE=1, 40L)
- GPU: prefill 2.5s, decode 2.4 tok/s (MOE=1, 40L)
- GPU no-MoE: prefill 0.27s, decode 14 tok/s

## Remaining
- "Doug" vs llama "Here" — root cause unknown. NOT from attn_output_gate (already implemented).
- Possible: quantization noise at 2-3 bpw on 35B MoE, or difference in embd_norm/token_embd handling
