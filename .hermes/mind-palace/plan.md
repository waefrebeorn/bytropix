# Plan — May 21 PM (Phase 29c: Corrected Divergence Analysis)

## P0: 1:1 Parity with llama.cpp — Corrected Approach

The per-layer comparison reveals that **divergence starts from L0 (cs=0.405)**. The previous "L31 GQA divergence" theory (0.9585 cos-sim) was based on logit-level comparison — the intermediate hidden states diverge much earlier. Fixing L31 GQA will NOT fix parity.

### P0.1: Compare Token Embeddings (New — Highest Priority)
The first layer output (L0) has cs=0.405 vs reference. Possible causes:
1. **Token embedding lookup** — bytropix loads `token_embd.weight` but may produce different values than llama.cpp due to different load path or quantization
2. **Initial RMSNorm** — L0 pre-attention norm differs

**Action:**
1. Add `DUMP_EMBEDDING_DIR` to bytropix `wubu_model_forward_from_embd()` — dump `embeddings` right after memcpy
2. Run both with same 1-token prompt
3. Compare `global_model.input_embed.bin` (ref) vs bytropix `embedding.bin`

### P0.2: Trace L0 SSM (If Embeddings Match)
If embeddings match but L0 output diverges (cs=0.405), the SSM computation in L0 is the root cause.
- Add `DUMP_SSM_DEBUG_DIR` to `wubu_ssm_forward()` — dump conv_input, qkv_mixed, alpha, beta, gate, state
- Compare against reference L0_* intermediates

### P0.3: L31 Attention Debugging (De-prioritized)
L31 shows cs=0.471, which is actually BETTER than surrounding layers (L30=0.182). Not the primary divergence point.

## P1: Structural Fixes (De-prioritized until P0 resolved)

### P1.1: Chunked SSM CS>1
Still broken. FP accumulation across 30 SSM layers. Workaround: `FORCE_CPU_SSM_SEQ=1`.

### P1.2: gen_text binary naming
✅ **DONE** — symlink `gen_text → gen_text_cpu` created.

## Per-Layer Cos-Sim (vs llama-simple, "Hello" 1-token, CPU sequential)

| Layer | Cos-Sim | Layer Type |
|-------|---------|------------|
| L0 | 0.405 | SSM |
| L1 | 0.445 | SSM |
| L2 | 0.664 | SSM |
| L3 | 0.568 | GQA |
| L4-L5 | 0.549-0.627 | SSM |
| L6 | 0.445 | SSM |
| L7 | 0.316 | GQA |
| L8-L10 | 0.310→0.142 | SSM |
| L10-L30 | 0.142→0.182 | Mixed |
| L31 | 0.471 | GQA (cleaner than neighbors) |
| L32-L38 | 0.504→0.710 | Mixed |
| L39 | 0.496 | GQA |

**Key insight**: Cos-sim drops monotonically through SSM layers (L0→L10: 0.405→0.142), then oscillates. GQA layers consistently show higher cos-sim than neighboring SSM layers. The SSM recurrence is the primary divergence source.

## Known sm_120 Hardware Bugs
1. `static __shared__` inside loops: hangs on Blackwell. Use `extern __shared__` + manual offset.
2. `__syncthreads()` after warp-leader shared-write: hangs. Use serial reduction by thread 0.
3. `extern __shared__ uint8_t` with syncthreads loops: incorrect codegen. Use `float*` instead.
