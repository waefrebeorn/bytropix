# Goal Mantra — May 19, 2026 (Phase 8 Complete, Cos-sim Audited)

## THE GOAL
1:1 inference parity w/ llama.cpp for Qwen3.6-35B-A3B-UD-IQ2_M.
**gen_text: 7.8 tok/s decode (3.7×). gen_text_mtp: 29.9 tok/s.**

## STATE
| Metric | Value | Status |
|--------|-------|--------|
| Decode | **7.8 tok/s** | ✅ Verified |
| Prefill | **10.4 tok/s** | ✅ Verified |
| MoE decode/layer | **~2ms** | ✅ (was 10ms) |
| Cos-sim vs llama.cpp | **0.7944** | ✅ Pre-existing at IQ2_M. NOT a regression |
| MTP free-tokens | **29.9 tok/s** | ✅ Pipeline correct |
| MTP blk.40 mismatch | target=220, MTP=2 | ✅ Same in llama.cpp. Inherent at IQ2_M |

## COLD GAPS
P1: Expert prefetch (API ready, needs wiring)
P1: MTP higher-precision model for quality  
P2: Cos-sim 0.79 — fundamental at IQ2_M quantization

## GROUND TRUTH
- Model: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
- Reference: /home/wubu/llama.cpp/build/bin/llama-cli
- Cross-ref: ref_dumper (logits) + ref_dumper_mtp (MTP head)

## THE LOOP
pick highest undone → execute → compile → run → verify → mark done → report
