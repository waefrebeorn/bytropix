# 7. Future Roadmap (May 19 PM v22 — Triple Extended)

## Phase 22 Completed ✅
- Q4_0 KV cache: 4:1 compression, cos-sim 0.9994, CPU path
- Architecture discovery: 3:1 SSM/GQA interleaved pattern
- DUMP_INTERMEDIATE_DIR: 53 tensor types per layer reference tracing
- kv_cache_read_head multi-block read fix
- ref_dumper: multi-token prompt, numeric token ID mode

---

## P0 — Must Fix (Highest Impact)

| Task | Detail | Est. Effort |
|------|--------|-------------|
| Fix gen_text_gpu hang | Debug pre-existing GPU inference hang | 1-2h |
| GPU Q4_0 KV cache | Port Q4_0 to GPU growable cache (saves 3.7GB VRAM) | 2-4h |
| L31 cos-sim investigation | Understand 0.9585 gap via intermediate comparison | 2h |

## P1 — Speed

| Task | Detail | Est. Effort |
|------|--------|-------------|
| Unified SSM kernel Phase A | Fuse conv1d→SiLU→split→norm→beta | 3-5h |
| Parallel cuBLAS streams | Overlap QKV + gate matmuls | 2h |
| Sparse attention + global tokens | Quality at 256k+ | 4-8h |

## P2 — Correctness

| Task | Detail | Est. Effort |
|------|--------|-------------|
| MoE router on GPU | F32 top-k, removes last CPU step | 1-2h |
| Chunked prefill | 3-7x prefill at 256k from Qwen2.5-1M | 4-8h |
| DSA sparse attention | O(L log L) from DeepSeek-V3.2 | 8-12h |

---

## Cold Gaps (Research → Code Priority)

| Research Concept | Priority | Gap |
|-----------------|----------|-----|
| Normed sigmoid gating (DeepSeekMoE) | P2 | Uses softmax, should be sigmoid |
| Load balancing (DeepSeek-V3) | P3 | Training-time only |
| MTP self-speculative decode | P2 | Quant noise blocks verify |

---

## Known Issues

1. gen_text_gpu hangs after model load — pre-existing, may be CUDA driver or GPU state issue
2. L31 cos-sim 0.9585 — quantization noise amplification through 30 layers, expected behavior
3. GPU FP16 KV cache not compressed — Phase 22 only covered CPU path
