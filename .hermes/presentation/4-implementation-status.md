# bytropix Implementation Status (May 19 PM v22)

## Inference Pipeline (CPU)
| Component | Status | Detail |
|-----------|--------|--------|
| GGUF loader | ✅ | 733 tensors, 7 quant types |
| Tokenizer | ✅ | GPT-2 BPE, 248K vocab |
| SSM Gated DeltaNet | ✅ | Q5_K attn_qkv/gate, Q6_K out |
| GQA attention | ✅ | Q5_K Q/K/V/output, Q4_0 KV cache |
| IMRoPE | ✅ | sections=[11,11,10,0], θ=10M |
| MoE router (F32) | ✅ | Softmax top-8/256 |
| MoE expert dequant | ✅ | IQ2_XXS/IQ3_XXS on-demand |
| Output proj (Q4_K) | ✅ | Self-hosted vec_dot |
| **Q4_0 KV cache** | **✅ Phase 22** | **4:1 compression, cos-sim 0.9994** |
| **DUMP_INTERMEDIATE_DIR** | **✅ Phase 22** | **53 tensor types/layer** |

## GPU Pipeline
| Component | Status | Detail |
|-----------|--------|--------|
| Output proj (Q4_K) | ✅ | Custom CUDA kernel, ~0.1ms |
| GQA attention (FP16) | ✅ | cuBLAS, sliding window, tile=16384 |
| SSM recurrence | ✅ | 32 blocks × 128 threads |
| SSM conv+norm+full forward | ✅ | All 15 steps, 2 transfers/layer |
| MoE experts (IQ2_XXS) | ✅ | Per-expert cache, 259MB |
| **gen_text_gpu binary** | **❌ Hang** | **Pre-existing issue, needs debug** |

## Verification Tools
| Tool | Status | Detail |
|------|--------|--------|
| ref_dumper | ✅ | libllama.so, multi-token, intermed dump |
| layer_cos_sim | ✅ | Per-layer comparison |
| classify_layers.py | ✅ | SSM/GQA classification |
| analyze_intermediates.py | ✅ | Tensor browser |

## Vault Papers Consumed
- Qwen3 technical report
- Qwen2.5-1M (chunked prefill, RoPE extrapolation)
- DeepSeek-V3 (MoE, MTP, load balancing)
- DeepSeek-V3.2 (DSA sparse attention)
- DeepSeekMoE (normalized sigmoid gating)
- Gemma 3 (local/global attention ratio validation)
