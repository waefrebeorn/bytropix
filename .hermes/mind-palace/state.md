# State — Phase 28j: GPU MoE Denormal Fix + Compilation Fix Applied

**bytropix: GPU inference engine for Qwen3.6-35B MoE + vision multi-modal**

## FIXES APPLIED THIS SESSION
| Fix | File | Status |
|-----|------|--------|
| FORCE_CPU_SSM env var check | wubu_ssm.c:460 | ✅ Actually works now — guards GPU recurrence |
| wubu_moe.o GPU_SUPPORT | Makefile + wubu_moe.o | ✅ Was compiled without -DGPU_SUPPORT — GPU MoE path was dead code |
| GPU d_f16_f32 denormals | gpu_moe_kernel.cu:30 | ✅ IQ2_XXS blocks have 25% denormal d values. F16 denormals flushed to zero before fix |
| dump_hidden test methodology | dump_hidden.c | ✅ State contamination bug — CPU and GPU runs shared KV cache/SSM state |
| gen_text.c output proj | gen_text.c | ✅ Use CPU output proj for now (GPU output proj may have issues) |

## CURRENT STATE
| Metric | Value | Status |
|--------|-------|--------|
| GPU SSM (hybrid) decode | Cos-sim 1.0 vs CPU | ✅ WORKS |
| GPU SSM recurrence (GPU kernel) | Cos-sim 1.0 vs CPU | ✅ WORKS |
| GPU GQA prefill | Coherent output | ✅ WORKS |
| GPU MoE (with denormal fix) | Cos-sim ~0.57 vs CPU | ❌ STILL BUGGY |
| GPU MoE performance | 0.7 tok/s (vs 4.3 CPU) | ❌ STALLED (cache miss each call) |
| CPU inference (no GPU ctx) | Coherent: "Paris is the capital of France..." | ✅ WORKS |
| GPU inference (MoE disabled) | Coherent: "Paris has been the first to have..." | ✅ WORKS |
| GPU inference (MoE enabled) | Garbage: "1. 1.1.1.1..." | ❌ BROKEN |
| GPU SSM C>1 forward_full | Stub (cuBLAS error 13) | 🔧 Needs implementation |
| MTP model | Not compiled | ❌ |
| Vision encoder | 384 LoC, untested | 🟡 |

## REMAINING BUG: GPU MoE (final blocker for full GPU inference)
- Cos-sim 0.57 between GPU MoE and CPU quantized_matmul (Q8_K path)
- Algorithm formulas are identical (verified line by line)
- Denormal fix only accounts for ~2% improvement
- Performance: 7× slower than CPU (each call does H2D upload of weights due to cache miss)
- Suspected: weight pointer offset calculation, shared memory conflict, or precision accumulation

## DIRECTORIES
- `/home/wubu/bytropix/src/` — CUDA kernels + gguf reader + model
- `/home/wubu/bytropix/tools/` — gen_text.c/mtp.c, api_server
- `/home/wubu/bytropix/include/` — headers
- `/home/wubu/bytropix/.hermes/mind-palace/` — prestige docs, DA v12
- `/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (11.5GB, main)
- `/models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf` (11.9GB, +blk.40 MTP head)
