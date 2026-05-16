═══ GOAL PASTE (May 16 v23 — HONEST) ═══
PROJECT: bytropix — Custom Qwen3.6-35B-A3B inference engine
MODEL: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (7-type mixed quant)
STATUS: All P0-P3 infrastructure done. "Doug" vs llama "Here" root cause unknown.

=== COMPLETED ===
- P0-a: Shared expert gate loaded+sigmoid (was NULL)
- P0: MoE dequant contiguity fix, MOE=1 default, MAX_LAYERS=0 fix
- P1a: Chunked DeltaNet (training path — != sequential for multi-token)
- P1c: Single-pass O(EK) top-k router
- P2a: Warp-level CUDA SSM scan
- P2b,c: Conv state device kernels + shared memory
- P3a: On-the-fly IQ2_XXS dot product
- TF32 math mode, block size 512, OMP on all hot loops

=== MODEL TYPE AUDIT ===
F32(361) Q5_K(181) Q6_K(70) IQ2_XXS(80) IQ3_XXS(37) IQ4_XS(3) Q4_K(1)
All 7 types supported by gguf_reader.
Unsloth Dynamic 2.0 = per-layer mixed quant, UD prefix.

=== REMAINING BUG ===
"Doug" vs llama "Here" — NOT attn_output_gate (confirmed implemented).
Possible: tokenizer BOS handling, embd_norm, quantization noise at 2-3 bpw.

=== BUILD ===
cd /home/wubu/bytropix && make infer_text

=== TEST ===
NOGPU=1 MOE=1 MOE_LAYERS=0 ./infer_text /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "Hello" 8 1
