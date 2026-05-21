# Overnight Map — Phase 28u: IQ1_M Model Generated, GPU Path Opens

**Active repo:** /home/wubu/bytropix/  
**Current commit:** 332bed6 (pushed to origin/master)  
**Main model:** /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (11.5GB, 40 layers)  
**IQ1_M model:** /models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf (7.7GB, 1.90 BPW)  
**MTP model:** /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf (11.9GB)  

## Session Summary (May 21, 2026 — IQ1_M Generated, GPU Path Opens)

### What Was Done
1. **IQ1_M model generated** — 7.7 GB (1.90 BPW), fits in 8GB VRAM
   - Generated via `llama-quantize` with imatrix from 2 calibration chunks
   - Imatrix: `/models/qwen_imatrix.gguf`
2. **Verified bytropix CPU path handles IQ1_M** — coherent output, 8.0 tok/s decode
   - IQ1_M decode is 43% faster than IQ2_M (8.0 vs 5.6 tok/s)
   - Same output quality on short test: "the city of Paris. It is the capital"
3. **CPU path dispatch confirmed** — `quantized_matmul.c` handles IQ1_M via dequant→F32→SGEMM

### What This Enables
IQ1_M at 7.7GB fits in RTX 5050 8GB VRAM. Full GPU inference now theoretically possible:
- Upload all weights to GPU once
- Run full forward pass on GPU without per-layer CPU↔GPU transfers
- Eliminate H2D/D2H overhead that made GPU hybrid net-negative

### Remaining GPU Blockers
1. **GPU dequant kernels for IQ1_M** — `gpu_quant_matmul.cu` only has Q5_K/Q6_K
2. **GPU MoE divergence** — 0.9888 cos-sim per layer (DA v13)
3. **GPU quantized_matmul for all weight types** needed by the model

### Next Session
1. Start adding IQ1_M GPU dequant kernel
2. Or optimize existing CPU path further
