# state — May 16

## GPU optimizations applied
- TF32 tensor core math (cublasSetMathMode): ~2× matmul speedup
- Block size 256→512 for element-wise CUDA kernels
- Full 40-layer GPU: prefill 0.27s (no MoE), decode 14 tok/s

## llama.cpp comparison
- Chunked DeltaNet (llama): 3× prefill speedup — not in bytropix
- Warp-level parallel scan CUDA — not in bytropix  
- Fused gate+up MoE weights — not used by bytropix
- Shared expert gating — BUG: bytropix sets ffn_gate_inp_shexp=NULL
- Quantized on-the-fly matmul — not implemented

## Roadmap written
Plan in .hermes/mind-palace/plan.md — 5 phases, P0=P1a correctness+speed first
