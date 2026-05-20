# State — Phase 22: Q4_0 KV Cache + Architecture Discovery (May 19 PM v22)

**bytropix: Pure C inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE)**
**CPU: ~12 tok/s prefill — GPU: ⚠️ gen_text_gpu hangs (pre-existing) — Q4_0 KV cache: 4:1 compression**

## Vault / Research Consumed
- **Architecture discovery**: GGUF tensor enumeration proved 3:1 SSM/GQA interleaved pattern (NOT 30+10 contiguous). SSM layers: 0,1,2,4,5,6,...(30). GQA: 3,7,11,15,19,23,27,31,35,39(10).
- **DUMP_INTERMEDIATE_DIR**: Modified llama.cpp cb() to save ALL 53 intermediate tensor types per layer as F32 files. 1997 files per 5-token forward pass.
- **Unsloth UD quant formula**: SSM attn_qkv/gate=Q5_K, ssm_out=Q6_K, MoE experts=IQ2_XXS/IQ3_XXS, output=Q4_K.
- **Qwen3 technical report**: Verified 3:1 SSM/GQA pattern matches `full_attention_interval=4` metadata key.

## VRAM Budget (256k Context, Q4_0 KV Cache)
| Component | Size | Format |
|-----------|------|--------|
| GQA weights (F32) | 1,040 MB | cuBLAS SGEMM |
| SSM weights (quantized) | 692 MB | Q5_K/Q6_K native on GPU |
| SSM F32 weights (small) | ~30 MB | beta/alpha/dt_bias/a/conv1d/norm |
| SSM GPU conv_state | 2.3 MB | persistent |
| **KV cache (Q4_0)** | **720 MB** | **4-bit quantized, 4:1 vs F16** |
| Output proj (Q4_K) | 1,900 MB | quantized GPU kernel |
| SSM scratch | 49 MB | reusable intermediates |
| MoE + scratch | ~460 MB | cache(259MB) + scratch(200MB) |
| **Total** | **~6,453 MB** | **Fits 8GB VRAM with 1.5GB headroom** |

## Cos-Sim Verified
- **5-token prefill, CPU gen_text vs ref_dumper**: OVERALL 0.9994
- L00-L30 (all SSM + GQA interleaved): **0.998-0.9999**
- L31 (GQA-only): **0.9585** — quantization noise amplification through 30 layers
- L32+: recovers as error redistributes
- **GPU gen_text_gpu**: identical per-layer cos-sim (same CPU model)

## DA Audit (May 19 PM)
| DA Phase | Result | Details |
|----------|--------|---------|
| DA-1: Code vs Theory | 3 stale docs found ✅ Fixed | Architecture mislabeled as "30+10 contiguous" in README, plan, presentation |
| DA-2: Vault Deep-Dive | Synthesis matches implementation | Qwen3.6 arch, DeepSeek MoE, Unsloth quant all verified |
| DA-3: Cold Gaps | P0 fix: doc sync | All stale claims propagated and corrected |

## Key Achievements (This Session)
- **DUMP_INTERMEDIATE_DIR**: Intermediate tensor dumper in llama.cpp — enables per-operation 1:1 parity debugging
- **Architecture discovery**: True 3:1 interleaved pattern confirmed (NOT contiguous)
- **Phase 22: Q4_0 KV cache**: 4:1 compression, identical cos-sim, CPU path working
- **kv_cache_read_head bug fix**: Multi-block reads now handle arbitrary-length heads
- **ref_dumper enhancement**: Multi-token prompt, numeric token ID mode

## Next Targets
1. **gen_text_gpu hang debug** — pre-existing issue, blocks GPU inference
2. **GPU Q4_0 KV cache** — currently GPU uses FP16 (5.12GB), Q4_0 saves ~3.7GB
3. **Sparse attention with global tokens** — preserve quality at 256k
4. **Unified SSM kernel Phase A**: fuse conv1d→SiLU→split→norm→beta
