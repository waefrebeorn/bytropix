# Goal Mantra — May 21 PM (Phase 29b: DA Sweep)

**Target:** CPU path verified coherent with FORCE_CPU_SSM_SEQ=1. 1:1 C-to-C parity with llama.cpp.

## STATE
| Component | Status | Detail |
|-----------|--------|--------|
| CPU text (sequential) | ✅ 3-4 tok/s | FORCE_CPU_SSM_SEQ=1. "the city of Paris..." |
| CPU text (chunked CS>1) | ❌ Garbled | FP accumulation across 30 layers |
| GPU vision encoder | ✅ 15.7s total | Only GPU win. 0.52s ViT |
| MTP spec decode | ✅ 8.5 tok/s | 4% acceptance (quant head) |
| GPU text hybrid | ⚠️ NET-NEGATIVE | 2-5x slower than CPU |
| GPU quant matmul | ✅ 4 types | Q5_K, Q6_K, Q4_K, IQ1_M |

## P0: What's Actually Next
1. Llama.cpp inline hooks — modify llama.cpp source to dump intermediates via cb() (no llama-cli). Replace ref_dumper libllama.so pattern with direct C++ hooks in llama_decode().
2. 1:1 parity via intermediate tensor comparison — use DUMP_INTERMEDIATE_DIR to compare bytropix vs reference at every processing stage.
3. Fix chunked SSM CS>1 — switch to sequential for now, fix later when needed.

## BUILD
```
make gen_text_cpu
FORCE_CPU_SSM_SEQ=1 ./gen_text_cpu "prompt" N
```

## HARDWARE
- RTX 5050 Blackwell (sm_120), 6.5-8 GB VRAM
- 16 GB system RAM, 12 CPU cores
- CUDA 13.1 toolkit, LLC (libllama.so) at ~/llama.cpp/build/bin/

## EVERY FIX
compile → run with FORCE_CPU_SSM_SEQ=1 → verify coherent output → commit
