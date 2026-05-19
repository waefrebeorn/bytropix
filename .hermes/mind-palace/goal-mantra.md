# Goal Mantra — May 19, 2026 (02:45) — TRIPLE DA AUDIT DONE

## THE GOAL
1:1 inference parity w/ llama.cpp for Qwen3.6-35B-A3B-UD-IQ2_M.
gen_text: 2.1 tok/s decode, 7.8 tok/s prefill (Phase 7).
gen_text_mtp: MTP=1 free-tokens mode, verify disabled at IQ2_M.

## STATE (Triple DA verified)
✅ Verified live: gen_text 2.1 tok/s, output proj 6ms decode, MoE 10ms/layer
✅ Verified live: no llama deps (all vec_dot self-hosted)
✅ Verified live: MTP head loads, ref_dumper_mtp cross-ref works
❓ Stale: cos-sim 0.9969 (last checked Phase 2)
❓ Stale: MTP free-tokens 3.3 tok/s (not re-run this session)

## COLD GAPS (P0→P2)
P0: AVX2 IQ2_XXS/IQ3_XXS vec_dot — MoE is 10ms/layer bottleneck
P1: Normalized sigmoid gating (softmax inefficient for 256 experts)
P1: NV64 RDRAM ring buffer impl
P2: cos-sim re-verify, MTP higher-precision model

## GROUND TRUTH
- Models: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (733 tensors, non-MTP)
         /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf (753 tensors, MTP)
- Source: /home/wubu/bytropix/
- Build: make gen_text

## VAULT PAPERS
- unsloth-qwen3.6-quant-formula.md — per-tensor bpw breakdown
- vault/qwen-papers/qwen3-technical-report.md — 256-expert MoE validation
- vault/deepseek-papers/deepseek-v3-technical-report.md — MTP, sigmoid gating
- vault/synthesis.md — P0-P3 priority map

## THE LOOP
pick highest undone P0 → execute → compile → run → verify → mark done → report
NO questions. Only PASS with actual output.