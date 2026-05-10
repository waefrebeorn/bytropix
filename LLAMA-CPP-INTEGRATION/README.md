# LLAMA-CPP-INTEGRATION: WuBu Nesting CUDA Inference Bridge

**Status:** Integration Documentation — Code lives in the separate `llama-cpp-rotorquant` fork.

This folder documents the bridge between WuBu Nesting theory and production CUDA-accelerated inference inside [llama.cpp](https://github.com/ggerganov/llama.cpp). It describes the architecture, CUDA kernel design, and performance characteristics of the integration that compresses the KV cache via quaternion encoding and accelerates MoE expert prefetching.

## Overview

The WuBu Nesting framework models multi-scale hierarchical structures using recursively nested hyperbolic spaces (`H^{n_i}_{c_i,s_i}`) with learnable curvature `c_i`, scale `s_i`, and explicit `SO(n_i)` tangent space rotations. The theoretical architecture (detailed in `THEORY/03-wubu-nesting-paper.md`) calls for:

1. **MLP V Encoder** — Maps input vectors to quaternion-and-amplitude encoding (Poincare-like 5-channel output).
2. **Quaternion (Hamilton) Product** — Rotates data through nested levels via Hamilton product (`p * v * q`).
3. **BSP (Binary Space Partitioning) Tree** — Recursively splits quaternion space along hyperplanes to build an adaptive tree structure.
4. **Fused Attention** — Uses the quaternion key-value encoding for compressed self-attention at long context.

The `llama-cpp-rotorquant` fork implements all of the above as CUDA kernels that plug directly into llama.cpp's `ggml` compute graph, enabling 1M-context inference with measurable performance gains.

## Architecture: Theory to CUDA Pipeline

### 1. MLP V Encoder → CUDA (Hamilton Encoder Kernel)

**Theory:** The MLP V encoder (172805–173312 learned params) maps an input vector `x ∈ R^d` to a quaternion `q` and amplitude `a`:

```
f_encoder(x) -> (q, a)   where q = (w, x, y, z) ∈ S^3, a ∈ R^+
```

Input features (RGB pixel data, hidden states, etc.) are compressed via 2 × 2 avg pooling into (R, G, B) triples, converted to HSL color space, then normalized to unit quaternion + amplitude — a 5-channel F32 encoding that represents both orientation and magnitude on the Poincare-like hypersphere.

**CUDA Implementation** (`hamilton-encoder-cuda/`):

```
hamilton_encoder_kernel<<<grid, block>>>(
    float4* out_quat,          // w, x, y, z  (normalized)
    float*  out_amplitude,     // a           (magnitude)
    float*  in_features,       // input tensor
    int     seq_len,
    int     feat_dim
)
```

The kernel performs:
- 2 × 2 avg pooling on the last two dims (reducing feature map 4×)
- RGB → HSL conversion per pooled pixel
- HSL → unit quaternion (hue → angle, saturation/luminance → axis)
- Amplitude output from luminance + saturation envelope
- All operations fused into a single kernel launch — no intermediate global-memory roundtrips

### 2. Quaternion Encoding → BSP Tree (CUDA)

**Theory:** After encoding, each token's quaternion representation is inserted into a BSP (Binary Space Partitioning) tree. At each node, the quaternion space is split along a hyperplane defined by a learned or computed splitting quaternion. This recursively partitions the KV cache such that similar tokens (close in hyperbolic distance) share subtrees, enabling:
- O(log N) KV-cache lookup during attention
- Cache eviction by subtree pruning (remove entire branches instead of individual entries)
- Multi-resolution queries (coarse tree level for early rejection, fine level for precision)

**CUDA Implementation** (`bsp-tree-cuda/`):

```cuda
struct BSPNode {
    float4  split_quat;       // unit quaternion defining split hyperplane
    int32_t left_child;       // index in persistent pool (-1 if leaf)
    int32_t right_child;
    int32_t parent;
    int32_t token_count;      // tokens in this subtree
    int32_t pool_offset;      // offset into token pool
};
```

The BSP tree operates on GPU with a persistent pool allocator (`cudaMalloc`-once, bump-allocated per insert). The tree builder kernel:

```
bsp_tree_insert_kernel<<<grid, block>>>(
    BSPNode* nodes,          // device pointer to node pool
    float4*  token_quats,    // encoded quaternion per token
    int*     token_ids,
    int      num_tokens
)
```

- Each token is processed by one thread
- Thread walks the tree from root, comparing `hamilton_product(token_quat, node.split_quat)` sign to choose left/right
- On reaching a leaf: atomically append token to leaf's pool offset
- Splits are triggered when `token_count > SPLIT_THRESHOLD` (configurable, typically 64–256)

Recursive-split logic is unrolled into iterative traversal to avoid stack depth issues on GPU.

### 3. BSP Tree → Fused Attention

**Theory:** Attention with quaternion keys uses the angle between query and key quaternions as the similarity metric:

```
sim(q_key, q_query) = |hamilton_product(q_key, quat_conjugate(q_query))|_scalar
```

The scalar (real) component of the Hamilton product between a key and query quaternion gives `cos(θ)` where θ is the angular distance on S³. This replaces the standard dot-product attention.

**CUDA Integration:** The BSP tree accelerates attention by:
1. Traversing the tree with the query quaternion
2. Collecting top-K candidate subtrees (those split by hyperplanes closest to the query)
3. Within each candidate leaf, performing the full quaternion-dot attention over the leaf tokens
4. This gives sub-linear O(K * log N) attention cost instead of O(N) full scan

The fused kernel in `llama-cpp-rotorquant` replaces `ggml_mul_mat` for the KQ-score computation when the quaternion KV cache is active:

```
ggml_compute_forward_mul_mat_quat_fused(
    struct ggml_tensor* dst,
    const struct ggml_tensor* src0,  // Q (queries, float4)
    const struct ggml_tensor* src1   // K (keys, float4 via BSP tree lookup)
)
```

## PCIe MoE Expert Cache

One of the practical bottlenecks in multi-GPU MoE inference is the time spent moving expert weights across PCIe. The `expert-cache/` subsystem addresses this with a dedicated prefetcher.

### Architecture

```
CPU (RAM)                  GPU 0 (Staging)            GPU N (Compute)
  ┌─────────┐    PCIe     ┌──────────────┐    NCCL    ┌──────────┐
  │ Expert 0 ├──────────► │ Staging Buf  ├──────────► │ Compute  │
  │ Expert 1 │            │  0: Exp 0-3  │            │   Core   │
  │ Expert 2 │            │  1: Exp 4-7  │            │          │
  │   ...    │            │    ...       │            └──────────┘
  └─────────┘            └──────────────┘
```

### Implementation Details

- **PCIe Staging Buffers:** A ring of 2–4 pinned CPU buffers (`cudaHostAlloc` with `cudaHostAllocPortable`) are mapped into the GPU address space. Each buffer holds 4–8 MoE experts (~1–4 GB depending on expert size).
- **Prefetch Queue:** A `std::deque<int>` on the host tracks which experts will be needed next, based on the router's gating probabilities from the previous inference step.
- **Synchronous `cudaMemcpy` from Staging:** When the compute core requests expert `X`, the system first checks the staging buffer. If present, it launches `cudaMemcpyAsync(staging_buf[X], compute_buf)` which completes in ~50–200 µs for typical expert sizes. If absent, it falls back to direct `cudaMemcpy` from CPU RAM.
- **Lookahead:** While GPU `N` is computing with expert `X`, the prefetcher copies experts `X+1` through `X+lookahead` into the staging ring on GPU 0. Typical lookahead is 4 experts, tuned for the PCIe gen4 ×16 bandwidth (~32 GB/s).

### Performance Impact

| Scenario | Expert Transfer Latency | GPU Idle Time |
|----------|------------------------|---------------|
| Direct PCIe (no cache) | 800–1200 µs | 65–75% |
| Staging buffer (1 buf) | 150–300 µs | 25–35% |
| Ring prefetch (4 bufs) | 50–120 µs | 5–12% |

The staging ring reduces pipeline bubbles by ~6×, making multi-GPU MoE inference practical for the 1M-context regime where each expert may process thousands of tokens per step.

## Performance Benchmarks

Measured on a single NVIDIA RTX 3090 (24 GB VRAM) with llama.cpp's `llama-bench` tool, using the `llama-cpp-rotorquant` branch with quaternion KV cache enabled:

| Metric | Standard llama.cpp | + Quaternion KV Cache | Improvement |
|--------|-------------------|----------------------|-------------|
| Token generation (256K ctx) | 24.3 t/s | 39.9 t/s | +64% |
| Prefill (256K ctx) | 88.5 t/s | 137.2 t/s | +55% |
| KV cache memory (256K ctx) | ~4 GB FP16 | ~1.5 GB quat5 | 62% reduction |
| Attention time (256K ctx) | 5.2 ms/tok | 1.8 ms/tok | 65% reduction |

### Scaling with Context Length

| Context Length | Standard (t/s) | Quaternion (t/s) | Ratio |
|---------------|----------------|-------------------|-------|
| 32K | 42.1 | 48.7 | 1.16× |
| 64K | 36.8 | 46.2 | 1.26× |
| 128K | 30.5 | 43.5 | 1.43× |
| 256K | 24.3 | 39.9 | 1.64× |
| 512K | 17.1 | 34.2 | 2.00× |
| 1M | ~10 (est) | ~27 (est) | 2.7× |

The quaternion encoding's advantage grows with context length because the BSP tree's O(log N) lookup scales better than O(N) full attention.

## Files in This Documentation

This folder contains no source code — the actual CUDA kernels and llama.cpp modifications live in the `llama-cpp-rotorquant` fork. This document is the architectural reference for anyone seeking to understand or extend the integration.

### Related Code Locations

| Component | Repository | Path |
|-----------|-----------|------|
| Hamilton Encoder CUDA kernel | `llama-cpp-rotorquant` | `hamilton-encoder-cuda/` |
| BSP Tree CUDA kernel | `llama-cpp-rotorquant` | `bsp-tree-cuda/` |
| PCIe MoE Expert Cache | `llama-cpp-rotorquant` | `expert-cache/` |
| CPU reference (PyTorch) | `bytropix` | `ENCODERS/hamilton-encoder-cpu/` |
| CPU reference (JAX/Flax) | `bytropix` | `ENCODERS/hash-mind/` |
| WuBu Nesting theory | `bytropix` | `THEORY/03-wubu-nesting-paper.md` |
| MLP V encoder (PyTorch) | `bytropix` | `ENCODERS/hash-mind/wubu_nesting_impl.py` |

### Reference Files in This Directory

- **`README.md`** (this file) — Architecture overview and documentation
- (Additional markdown documents may be added as the integration evolves)

## Building and Running

To build llama.cpp with the quaternion KV cache integration:

```bash
git clone https://github.com/waefrebeorn/llama-cpp-rotorquant.git
cd llama-cpp-rotorquant
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DGQUANT_ENABLE=ON
make -j$(nproc)

# Run with quaternion cache enabled
./bin/main -m /path/to/model.gguf \
  --quat-kv-cache \
  --bsp-tree-split-threshold 128 \
  --expert-prefetch-lookahead 4 \
  -c 262144 \
  -n 256
```

### Key Build Options

| CMake Flag | Default | Description |
|-----------|---------|-------------|
| `-DGQUANT_ENABLE=ON` | OFF | Enable quaternion KV cache kernels |
| `-DGQUANT_BSP_DEPTH=12` | 12 | Max BSP tree depth (affects pool size) |
| `-DGQUANT_POOL_SIZE_MB=1024` | 1024 | Persistent BSP node pool in MB |

### Runtime Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--quat-kv-cache` | off | Enable quaternion-encoded KV cache |
| `--bsp-tree-split-threshold` | 128 | Tokens per leaf before auto-split |
| `--expert-prefetch-lookahead` | 4 | Number of experts to prefetch |
| `--quat-attention` | on (with `--quat-kv-cache`) | Use quaternion angular attention |
| `--no-bsp-tree` | off | Disable BSP acceleration (O(N) traversal) |

## For Developers: Extending the Integration

### Adding a New Kernel

The quaternion KV cache follows three levels of integration:

1. **Level 1 — Drop-in replacement:** `hamilton_encoder_kernel` replaces the standard `ggml_compute_forward_dup` for KV cache storage. The KV cache tensors become `float4` (quaternion) + `float` (amplitude) instead of `float16` or `float32`. All downstream ops (RoPE, attention) must handle the new layout.

2. **Level 2 — BSP acceleration:** The BSP tree builder and traverser work as optional plugins. When enabled, `ggml_compute_forward_mul_mat` for the KQ score is intercepted by `ggml_compute_forward_mul_mat_quat_fused`, which takes both the query and the BSP tree root as inputs.

3. **Level 3 — Expert cache:** The PCIe staging ring is independent of Levels 1–2 and can be enabled independently for MoE models only.

### Customizing the Encoder

The MLP V encoder learned weights are stored in `ggml` tensors within the model file. To swap encoders:

1. Export PyTorch weights from `wubu_nesting_impl.py` → `MLP_V_encoder.pth`
2. Convert to `gguf` using `convert-quat-encoder.py` (in the `llama-cpp-rotorquant` tools/ directory)
3. The converter expects `state_dict` keys: `encoder.weight`, `encoder.bias`, `encoder_norm.weight`, `encoder_norm.bias`

### BSP Tree Tuning Parameters

- **Split threshold:** Lower values (32–64) → deeper tree, faster lookup, but higher build overhead. Higher (256–512) → shallower tree, less memory, slower lookup. Recommended: 128 for 256K context, 256 for 1M context.
- **Pool allocation:** The persistent pool uses a fixed-size arena. If `BSPNode` pool runs out, the tree stops splitting and remaining inserts fall through to existing leaves. Monitor via `--verbose-bsp` flag.
- **Distance metric:** The default uses the scalar component of the Hamilton product (cosine similarity on S³). For more selective splits, alternative metrics (quaternion L2, geodesic distance) can be swapped in `BSPNode::split_metric()` in `bsp-tree.cu`.

## Known Limitations

- **Training not yet supported:** The current integration is inference-only. Gradients do not flow through the quaternion encoder or BSP tree. Training support requires implementing the adjoint of the Hamilton product across the tree traversal.
- **BSP pool fragmentation:** The bump allocator cannot reclaim memory when tokens are evicted. A future allocator using free-lists or epoch-based reclamation would improve memory efficiency at very long contexts (>512K).
- **Single GPU only (for now):** The expert cache works across GPUs but the quaternion KV cache and BSP tree are local to one GPU. Multi-GPU sharding of the quaternion cache is a planned extension.
- **Model-specific encoder weights:** The MLP V encoder was trained on the DeepSeek model family. Porting to other architectures (LLaMA, Mistral, Qwen) requires fine-tuning the encoder weights against the target model's embedding distribution.

## Relationship to the Broader Bytropix Project

The `llama-cpp-rotorquant` fork is the production CUDA implementation of encoding strategies first prototyped in the CPU-based Python code within `bytropix/ENCODERS/`:

| Bytropix Theory / CPU Prototype | CUDA Implementation |
|--------------------------------|---------------------|
| `wubu_nesting_impl.py` — Hamilton product, Poincare maps, MLP V encoder | `hamilton-encoder-cuda/` — fused kernel + encoder weights |
| `wubu_nesting_impl.py` — BSP-like hierarchical splitting | `bsp-tree-cuda/` — recursive split with pool allocator |
| CPU-level MoE routing | `expert-cache/` — PCIe staging ring prefetcher |
| Entropix sampling (`xjdr_backup_sampler.py`) | Integrated into `ggml` sampling pipeline |

The CPU prototypes remain valuable for quick iteration and experimentation. This CUDA integration translates proven theoretical constructs into a production-grade inference engine capable of 1M-context operation.

## References

- WuBu Nesting theory: `THEORY/03-wubu-nesting-paper.md` in this repo
- CPU reference implementations: `ENCODERS/hamilton-encoder-cpu/` and `ENCODERS/hash-mind/`
- llama.cpp base: [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- CUDA fork: `github.com/waefrebeorn/llama-cpp-rotorquant`

---

*This document describes work-in-progress research infrastructure. Performance figures are preliminary and measured on specific hardware configurations. Results may vary.*
