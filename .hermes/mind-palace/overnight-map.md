# Overnight Map — May 21 PM (Phase 29c: DUMP_INTERMEDIATE_DIR Hooks + Divergence Audit)

**Active repo:** /home/wubu/bytropix/  
**Current commit:** ec58b72 (uncommitted: DUMP_GQA_DEBUG_DIR + gen_text symlink)  
**Model:** /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (11.5GB, only model)  
**Reference:** /home/wubu/llama.cpp/build/bin/llama-simple (fast, no chat template)

## Session Summary (Phase 29c)

### New Debug Infrastructure
| Feature | Files Changed | Description |
|---------|--------------|-------------|
| Llama.cpp DUMP_INTERMEDIATE_DIR | Rebuilt libllama.so + llama-simple | Already existed in source, now compiled in. Dumps 1997 intermediates. |
| DUMP_GQA_DEBUG_DIR | `src/wubu_ssm.c` | Dumps input, Q_full, K, V, Q_norm, K_norm, attn_pre/post-gate, output |
| DUMP_GQA_PREFIX | `src/wubu_ssm.c` | Per-layer prefix via env var |
| DUMP_GQA_LAYER | `src/wubu_model.c` | Sets DUMP_GQA_PREFIX only for target layer |
| gen_text symlink | `gen_text → gen_text_cpu` | Fixes missing `gen_text` binary |

### Key Finding
**Divergence starts at L0 (cs=0.405), not L31.** Per-layer comparison:
- L0: 0.405 — first layer output already diverged
- L1-L30: drifts from 0.445 down to 0.182
- L31: 0.471 — actually CLEANER than neighbors (L30=0.182, L32=0.504)
- L38-39: improves to 0.710 and 0.496

Both systems produce the same output token ("," = token 11 for "Hello" prompt). Hidden states diverge but final token is correct.

### Llama.cpp Reference
- `llama-simple` is fast (~1s for 1-token prefill) and doesn't add chat template tokens
- `llama-completion` adds ~8 extra chat template tokens (bad for 1:1 comparison)
- `llama-cli` without `--no-conversation` also wraps in chat template

### Verified Claims
| Claim | Status | Evidence |
|-------|--------|----------|
| CPU text coherent (sequential) | ✅ | "the city of Paris..." |
| DUMP_INTERMEDIATE_DIR works | ✅ | 1997 files per forward pass |
| DUMP_GQA_DEBUG_DIR works | ✅ | Per-layer GQA intermediate dumps |
| gen_text symlink | ✅ | `gen_text → gen_text_cpu` |
| L31 root cause debunked | ✅ | Divergence starts at L0, not L31 |

## Current Blockers
1. **1:1 parity**: Hidden states diverge from L0. Need to compare token embeddings first.
2. **Chunked SSM CS>1**: Must use `FORCE_CPU_SSM_SEQ=1`.
3. **GPU text net-negative**: H2D/D2H overhead.

## Workstreams (next session)
**A — Compare token embeddings**: Add `DUMP_EMBEDDING` to bytropix gen_text_cpu, compare against reference `global_model.input_embed.bin`.

**B — Trace L0 SSM**: If embeddings match, add SSM intermediate dumps to `wubu_ssm_forward()` (similar to DUMP_GQA_DEBUG_DIR).

**C — Final logit comparison**: Compare final logits between bytropix and reference.

## Data NOT to Re-Derive
- L31 GQA is NOT the primary divergence source (cs=0.471, better than neighbors)
- Per-layer comparison script exists (see session transcript)
- Both systems output same token for "Hello" (token 11 = ",")
