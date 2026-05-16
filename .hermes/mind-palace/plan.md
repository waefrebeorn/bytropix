# bytropix Roadmap — DA-Recertified May 16

## Phase 0: Critical Bug Hunt (NEW — top priority)
- [ ] P0a: Write Q5_K dequant test — compare single block vs llama.cpp reference
- [ ] P0b: Write Q4_K (type 12) dequant test — compare output.weight block
- [ ] P0c: Verify SSM Q scaling factor — check llama.cpp delta-net-base
- [ ] P0d: Verify RMSNorm epsilon — check llama.cpp default
- [ ] P0e: Remove TGT wrapping from GQA attention — compare output with/without
- [ ] P0f: Fix wubu_gqa_forward() weight indexing — i*cols+j → i+j*D_MODEL

## Phase 0.5: First-Token Parity
- [ ] Run layer-0-only inference, dump h_last, compare vs llama.cpp LLAMA_DUMP_LAYER_DIR
- [ ] Binary search: find first layer where hidden diverges from reference

## Phase 1: Inference Speed ✓ (DA: compiled only, not verified)
- [ ] P1a Chunked DeltaNet — training-only, not used
- [ ] P1b Fused Gate+Up — design decision, separate weights fine
- [ ] P1c Single-Pass Top-K — same algorithm as bubble sort, correct

## Phase 2: GPU Optimization ✓ (DA: all compiled only)
- [ ] P2a Warp CUDA scan — not used in CPU test path
- [ ] P2b Conv state kernels — not used in CPU test path
- [ ] P2c Conv1d shared mem — not used in CPU test path
- [ ] TF32/block 512 — speed only, not correctness

## Phase 3: Quant ✓ (DA: partially verified)
- [x] P3a IQ2 on-the-fly dot — 4/4 unit tests pass
- [ ] P3b K-Quant — raw_size + dequant functions. NEVER verified vs llama.cpp

## Phase 4: Architecture
- [ ] P4a Model graph — not needed until operator fusion required
- [ ] P4b KV Cache manager — not needed until 256K context

## Phase 5: Training
- [ ] P5a Backward checkpointing — not started
