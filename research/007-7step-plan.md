# 7-Step Plan: wubuwizard Inference Engine + OS Integration

Kevin-Bacon convergence fields:
1. **Roofline 2607.02558**: decode is memory-bandwidth-bound.
2. **KIVI 2402.02750**: KV quant K-per-channel, V-per-token.
3. **DeltaNet 2406.06484**: parallel scan + chunkwise recurrence.
4. **FlashDecoding++ 2311.01282**: multi-block attention reduces KV traffic.
5. **Continuous batching Anyscale 2025**: iteration-level scheduling.
6. **Speculative decoding 2402.01528**: draft+verify.
7. **TurboQuant llama.cpp 2504.19874 + KIVI**: sub-3-bit KV, 63% reduction.

Convergent truth across all 7 fields: **bytes moved dominates tok/s**. Every step attacks bytes moved.

---

## Step 1: KIVI KV-cache quantization (DONE)
- Research: K-per-channel (KIVI), V-per-token (KIVI), asymmetric.
- Implementation: `src/wibu_kvcache_quant.c`, `include/wubu_kvcache_quant.h`
- Test: `tools/test_kivi_roundtrip.c` — 3/3 PASS (K cosine 0.999995, V cosine 0.997)
- Commit: `0858c7d`
- **Status**: committed, green.

## Step 2: SSM workspace pool (DONE)
- Research: per-call 13×malloc/free is wasted cycles in bandwidth-bound decode.
- Implementation: `src/wubu_ssm_workspace.c`, `include/wubu_ssm_workspace.h`
- Test: regression suite green (8/8 tests).
- Commit: `542adf0`
- **Status**: committed, green.

## Step 3: Model-agnostic loader + KV cache memory pool
- Research: vLLM PagedAttention + OS paging = same structure; reuse.
- TODO:
  - Make `wubu_model_init` accept HF adapter config **or** GGUF path (agnostic).
  - Pre-allocate KV cache from adapter-reported `n_layers` + `head_dim`.
  - Fix KAT-Coder 13-shard path routing (`model-00000-of-00013.safetensors`).
- Files: `src/wubu_model.c`, `src/wubu_model_adapter.c`
- Test: `test_model_adapter` verifies KAT, Agents-A1, Qwen36, BTL-3.

## Step 4: Model-agnostic gen_text decode loop
- Research: decode is bandwidth-bound; route dynamically from adapter metadata.
- TODO:
  - Make `tools/gen_text.c` read `--model` arg instead of hardcoded GGUF.
  - Detect shard layout (single/13-shard/etc.) at load time.
  - Use KV cache scheme selected by Roofline auto-selector.
- Files: `tools/gen_text.c`
- Test: `gen_text` produces finite output for all 4 models.

## Step 5: KV-styx live KV inspection
- Research: OS-level KV visibility for debugging (PagedAttention + 9P).
- TODO:
  - Register each layer's KV cache in `wubu_kv_styx` after model init.
  - `wubu_kv_styx_register("layer/0/K", ...)` for n_layers.
  - External tools read via 9P at `/n/kv/`.
- Files: `src/wubu_kv_styx.c`, `src/wubu_model.c`
- Test: `test_kv_styx` + `test_real_load` confirm registration.

## Step 6: DA-2 kernel schema gate + triple-DA audit
- Research: DA-2 = fail-closed; kernel mismatch refuses weight load.
- Already done:
  - `include/wubu_da_guard.h` — env-var gate.
  - `test_model_adapter.c` DA-2 test — 2/2 PASS.
- Commit: `7989d05`
- **Status**: committed, green.

## Step 7: WuBuOS realm kernel schema integration
- Research: `WUBU_KERNEL_SCHEMA` exported by `wubu_realm_start()` (ZealOS kernel).
- TODO:
  - `wubu_realm.c` call `setenv("WUBU_KERNEL_SCHEMA", "1", 1)` at boot.
  - wubuwizard checks env at load → fail-closed if mismatch.
- Files: `WuBuOS/src/runtime/wubu_realm.c`
- Test: realm boot + wubuwizard load in same session.
