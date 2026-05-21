# DA Synthesis — May 21, 2026 — Phase 28o

## Triple Devil's Advocate Run Complete

### DA-1: Code vs Claims Audit (All Files Checked)

| File | Prior Claim | Reality After DA | Updated? |
|------|------------|------------------|----------|
| state.md | GPU vision done, P1 complete | ✅ Still current — sharpened P2 status | ✅ YES |
| goal-mantra.md | GPU hybrid net-negative | ✅ Verified — added CPU-only optimal | ✅ YES |
| plan.md | P0-P1 listed as open | ✅ P0-P1 closed, P2 reprioritized | ✅ YES |
| prestige_prompt.md | May 21 version | ✅ Added DA v13 findings, P2 queue | ✅ YES |
| ARCHITECTURE.md | May 18 — pre-GPU, pre-MTP, pre-vision | ❌ MASSIVELY OUTDATED — GPU, MTP, vision, DA v13, sm_120 bugs | ✅ REWRITTEN |
| testing.md | May 16 — "inference is broken" | ❌ STALE — CPU inference works since May 18 | ✅ REWRITTEN |
| project.md | May 16 — "inference is broken" | ❌ STALE — needs accurate state | ✅ REWRITTEN |
| research.md | May 18 — tensor types correct | ✅ Still current | ✅ Verified |
| overnight-map.md | May 21 | ✅ Current | ✅ Uncommitted changes saved |
| DA v12 | May 20 — P0 GPU MoE listed as open | ❌ DA v13 supersedes — GPU MoE is FUNDAMENTAL, not fixable | ✅ v13 is authoritative |

### DA-2: Vault Deep Dive

**Papers Read (all vault/deepseek-papers/):**
- DeepSeek-V3 (2412.19437) — MTP, auxiliary-loss-free load balancing, sigmoid gating ✅
- DeepSeek-V3.2 (2512.02556) — DSA sparse attention O(L log L) ✅
- DeepSeekMoE (2401.06066) — Sigmoid gating, shared experts, fine-grained segmentation ✅
- DeepSeek-R1 (2501.12948) — Pure RL for reasoning ✅
- synthesis.md — Cross-paper validation of 256/8 MoE config, 30:10 local:global ratio ✅

**Key gaps found:**
1. No code for DSA sparse attention (needed for 256K context)
2. No code for sigmoid gating (current router uses softmax)
3. No code for auxiliary-loss-free load balancing
4. Shared experts code exists but needs verification
5. FP8 Tensor Cores (sm_120 native) not used at all

**Tools Vault Updated:**
- ref_dumper.cpp (libllama.so reference) ✅
- ref_dumper_mtp.cpp (MTP reference) ✅
- layer_cos_sim.c (per-layer comparison) ✅
- compare_moe_expert.c (GPU vs CPU expert compare) ✅
- dump_intermediates.c, dump_hidden.c, dump_layers.c ✅
- test_vision_real.c, test_moe_layer.c ✅
- README.md with env var documentation ✅

### DA-3: Hardware Utilization Audit

**RTX 5050 (sm_120 Blackwell) — Currently Underutilized:**
| Resource | Available | Used | 
|----------|-----------|------|
| CUDA cores | ~2560 | 512 threads/block |
| Shared mem/block | 48KB | ~8KB |
| FP8 Tensor Cores | Native sm_120 | Not used (FP32 only) |
| CUDA Graphs | Graph capture | Not used |
| Async H2D | Overlap with compute | Sequential uploads |
| Multi-block parallelism | 32 blocks/SM | 1 block per kernel |

**CUDA sm_120 Bugs Documented (6 total):**
- Bug 1-3: Already in cuda-sm120-bugs skill (static __shared__, syncthreads+reduce, extern uint8_t)
- Bug 4: WDDM compute-sanitizer unavailable (Windows driver limitation)
- Bug 5: WDDM memory/context init (2-5s first call)
- Bug 6: Divergent warp primitives (__shfl_xor_sync)

**Skill Updated:** cuda-sm120-bugs — added Bugs 4-6, build flags, verified configs table

### Critical Finding: 1:1 Parity Pipeline

**The reference data infrastructure works:**
```
ref_dumper → DUMP_LAYER_DIR → ref_layer_N.bin (per-layer hidden states)
ref_dumper → DUMP_INTERMEDIATE_DIR → L<N>_<tensor>.bin (Q/K/V, attn, MoE weights)
→ 1997 files per BOS token → layer_cos_sim comparison
```

**But bytropix DUMP_LAYER_DIR writes `our_layer_N.bin` with N*tokens floats**  
**llama.cpp writes `ref_layer_N.bin` with D_MODEL floats (1 token only)**  
→ For 1:1 parity, BOTH must be run with SAME prompt length

**Makefile fix:** gen_text_cpu now builds (was broken by GPU symbol dependencies in CORE_OBJ). Added `src/wubu_moe_cpu.o` as CPU-only variant.

### Plan Forward: 1:1 Parity Path (C Code to C Code)

**Next Session — P2.1 (Llama.cpp Inline Hooks):**
The DUMP_INTERMEDIATE_DIR already dumps ALL tensor intermediates from llama.cpp. The user's need ("MODIFYING LLAMA TO GET PROPER DATA") is ALREADY IMPLEMENTED at lines 1309-1407 of ~/llama.cpp/src/llama-context.cpp. Use:
```bash
DUMP_LAYER_DIR=/tmp/ref_layers DUMP_INTERMEDIATE_DIR=/tmp/ref_interm ./ref_dumper /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
```
This gives 40 hidden states + 1997 intermediate tensors for 1:1 parity checking.

**Immediate P2 Priority:**
1. ✅ CUDA sm_120 bug skill documented (DONE)
2. 🔲 RMSNorm + SiLU GPU kernels (low impact, code cleanup)
3. 🔲 Chunked prefill (medium impact, infra exists)
4. 🔲 RoPE extrapolation 4x (simple param change)
5. 🔲 NSA sparse attention (high impact for 256K)
6. 🔲 Sigmoid gating + load balancing (DeepSeekMoE)
7. 🔲 FP8 Tensor Cores (hardware-optimal)
