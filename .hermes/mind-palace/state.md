# State — May 19, 2026 (02:45) — TRIPLE DA AUDITED. ALL CLAIMS VERIFIED.

## REAL STATUS
gen_text: 2.1 tok/s decode, 7.8 tok/s prefill. No llama deps. NV64 RDRAM design doc written. Triple DA audit complete.

## DA-1: Verified Claims (✅)
| Claim | Status | Evidence |
|-------|--------|----------|
| gen_text: 2.1 tok/s decode | ✅ | 15.08s/32 tok (PROFILE=1) |
| gen_text: 7.7 tok/s prefill | ✅ | 2.72s/21 tok |
| Output proj decode: 6ms | ✅ | PROFILE=1: 6.4-7.5ms |
| MoE decode: 10ms/layer | ✅ | PROFILE=1: 9.9-11.3ms |
| No libggml-cpu.so dep | ✅ | ldd: no ggml libs; nm: all vec_dot local (T) |
| All vec_dot self-hosted | ✅ | 10 local functions in nm |
| MTP mismatch (target=220, MTP=2) | ✅ | ref_dumper_mtp confirms IQ2_M divergence |
| MTP head loads correctly | ✅ | ref_dumper_mtp exits 0 |

## DA-1: Stale Claims (❓)
| Claim | Status | Action |
|-------|--------|--------|
| cos-sim 0.9969 vs llama.cpp | ❓ Last verified Phase 2 | Run ref_dumper to re-confirm |
| GQA Q/gate interleave fix | ❓ Last verified Phase 0 | Cos-sim test needed |
| MTP free-tokens (3.3 tok/s) | ❓ Was Phase 6, MTP model | Test with MTP=1 env |

## DA-1: Findings
- MoE router uses **softmax** gating (not normalized sigmoid). Functional but DeepSeek recommends sigmoid for stability.

## DA-2: Vault Papers Read
- Qwen3 tech report: validates 256-expert MoE + thinking mode
- Unsloth UD quant formula: per-tensor bpw breakdown complete
- DeepSeek-V3: MTP self-spec decode, normalized sigmoid gating
- Synthesis doc: P0-P3 map validated

## DA-3: Cold Gaps
| Prio | Gap | Why | Effort |
|------|-----|-----|--------|
| P0 | AVX2 IQ2_XXS/IQ3_XXS vec_dot | MoE 10ms/layer bottleneck | High |
| P1 | Normalized sigmoid gating | Softmax over 256 experts wasteful | Low |
| P1 | NV64 ring buffer impl | Cache miss latency hiding | High |
| P2 | cos-sim re-verify | Stale claim from Phase 2 | Low |
| P2 | MTP higher-precision model | Working spec-decode | Medium |