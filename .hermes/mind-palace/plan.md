# bytropix Plan — May 17 v4 (TGT/manifold cleared, SSM math verified clean)

## CURRENT STATUS
All mathematical components verified correct. TGT/manifold not in inference path. Root cause is not in high-level math.

## Phase 0.5: First-Token Parity — MUST DO
Building layer-by-layer comparison is the only remaining move.

- [ ] **P0.5a** Build layer-dump tool using llama.cpp's `LLAMA_DUMP_LAYER_DIR` env var
  - llama.cpp already supports this (line 71-77 in qwen35moe.cpp)
  - Generates `/tmp/dump_layers/layer_N.bin` files
- [ ] **P0.5b** Add matching dump points in our engine (after each layer's residual)
  - Same format: 1-token "Hello", dump float32 vectors
- [ ] **P0.5c** Compare layer-by-layer cos-sim
  - Find first layer where cos-sim < 0.999
- [ ] **P0.5d** For the divergent layer, dump ALL intermediates:
  - QKV projection, conv output, Q/K L2-normed, delta_out, gated_norm, MoE out
- [ ] **P0.5e** Fix root cause
- [ ] **P0.5f** Re-verify full model

## Phase 1: Multi-token prefill parity
- [ ] Test on 2+ token prompts
- [ ] Verify SSM state accumulation

## Phase 2-5
Blocked on Phase 0.5.
