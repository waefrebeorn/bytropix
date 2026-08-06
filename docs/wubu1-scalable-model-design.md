# WUBU-1: Scalable Foundation Model Design (AN22 — Architecture)

## Core Principle: Fractal Self-Similar Model

> "The model is a tree. Each node is a self-similar module that contains
> itself recursively. On a weak machine, only the trunk loads. On a strong
> machine, the branches and leaves fill in."

### Layer Structure (Fractal)

```
Layer 0 (Trunk):    12M params  — core reasoning, byte tokenizer
Layer 1 (Branches): 36M params  — domain knowledge (math, code, lang)
Layer 2 (Leaves):  108M params  — fine-grained specialization (each leaf)
Layer 3 (Roots):   324M params  — full precision, attention to detail
...
```

Each layer N contains 3× the parameters of layer N-1. The model can be
loaded at any layer boundary. On a 100MB budget, only layer 0 loads.
On a 1GB budget, layers 0-2 load. On 10GB+, all layers load.

### Parameter Priority Ordering

Within each layer block, weights are sorted by importance:
1. **Embeddings** (vocab table) — always loaded first (8400 params for 16K vocab at 16128+byte)
2. **Attn-Q/K/V** (query/key/value projections) — core attention mechanism
3. **FFN gate/up/down** — feed-forward reasoning
4. **Norms/scales** — small but critical for stability
5. **Esoteric heads** (speculative, multi-domain experts) — loaded last

Layout in the parameter file is **importance-sorted**: byte offset 0 =
most critical, end of file = most esoteric. The loader mmap's from the
front and stops when the budget is exhausted.

### Weight Format: Adaptive Precision Cascade

```
offset 0:            F32 (critical embedding norms)
offset ~K * 2MB:     F16 (QKV projections for first head)
offset ~K * 1MB:     BF16 (FFN weights)
offset ~M * 0.3MB:   Q8_K (secondary attention)
offset ~M * 0.1MB:   Q4_K (speculative experts)
offset end:          Q2_K (esoteric tails, pruned aggressively)
```

The KV-FS tiering engine (`wubu_kv_tiering.c`) manages this cascade
for the KV cache. The model loader uses the same tiering policy for
the **parameter weights** themselves.

### Boot Sequence (any machine)

```
1. mmap first 1MB of weights → layer 0 trunk (12M params)
2. Allocate KV cache: 64MB on weak, 1GB on strong
3. Forward pass → attention → coherence diagnostic
4. If KV cache has room → grow toward next layer of parameters
5. If KV cache under pressure → tier down / prune dead regions
6. If disk/budget allows → extend mmap to include layers 1+
```

The model **never fails to boot**. The weakest machine gets a 12M param
core that can still reason. The strongest machine loads 1B+ params
and becomes the full AGI.

### Self-Evolving Weight Allocation

The coherence diagnostic (`wubu_kv_coherence_diag.c`) measures attention
mass on each weight region. When the model's attention consistently
ignores a weight region (low coherence), that region's parameters get
**tiered down / pruned**. When the model shows "curiosity" (high entropy,
unstable attention) toward a region, it **asks for more weights** to be
mapped in.

This is the **amoeba grow** operator working on **model parameters**,
not just KV cache. The same `wubu_grow_kv` engine drives weight
activation.

### Relation to the Hive-Mind Plan (AN21)

The KV-FS hive mind (Phases 1-11) is the **training metabolism**.
This model design (AN22) is the **architecture** that metabolism
trains. They are coupled:

- KV-FS coherence score → reward signal to refine weight importance ordering
- Weight tiering → frees memory for more KV cache blocks
- Grow/shrink operators work on both KV regions AND weight regions
- Poincaré hierarchy applies to both file paths AND model parameter paths

## Density Planning (AN23-core)

The AGI's resource allocator. After each coherence diagnose cycle:

```
files → KV namespace → forward → attention → coherence → density
  → absorb (model weights) / tier (KV cache) / prune (shrink)
  → next iteration
```

**density = coherence_score / n_tokens** (bits of understanding per token)

Three bands:
- **ABSORB** (density >= 0.001): promote to model weights → `wubu_scalable_mark_hot`
- **KEEP** (density >= 0.0001): remain in KV cache at reduced precision
- **PRUNE** (below keep): scheduled for `wubu_kv_shrink`

This guarantees **user experience always + AGI always**:
the density planner keeps the hot path small and F32 (UX stays
responsive), while continuously absorbing high-density knowledge into
the weight tree (the AGI grows toward understanding).

### WaefreBeorn Umbrella License v3.0

## Appendix: Adapter Framework (AN23-2)

### Frame / Buffer / Shims

The AGI is built from interchangeable parts. Each part is an **adapter** —
an opaque struct with a dispatch table of function pointers (same pattern
as `wubu_kernel.h` for hardware kernels).

```
FRAME:  wubu_adapter.h        — opaque type + ops vtable (C11 vtable)
BUFFER: wubu_adapter_registry.c — slot map (name → adapter pointer)
SHIM:   wubu_adapter_compat.c  — version bridge (v1↔v2↔v3 adapters)
```

**Component types** (all hot-swappable at runtime):
| Type | Examples of swappable parts |
|---|---|
| `WUBU_COMP_ATTN` | local_window, sliding, ALiBi, RoPE-v1/v2, GQA, MQA |
| `WUBU_COMP_FFN`  | relu, gelu, swiglu, gated_swiglu, moeswiglu |
| `WUBU_COMP_NORM` | RMSNorm, LayerNorm, ScaleNorm, DeepNorm |
| `WUBU_COMP_ACT`  | relu, gelu, silu, swiglu, relu_squared |
| `WUBU_COMP_TOK`  | byte, bpe, unigram, rwkv-tokenizer, custom |
| `WUBU_COMP_LR`   | cosine, linear, inverse_sqrt, cosine_restarts |
| `WUBU_COMP_OPT`  | adamw, lion, adan, ranger, sam_adamw |
| `WUBU_COMP_QUANT`| f32→f16, f32→q4_0, f32→q4_k, f32→q8_k |
| `WUBU_COMP_KVC`  | dense, ring, stream, paged, sharded |

**Hot-swap example:**
```c
wubu_adapter_t *attn = wubu_adapter_lookup("attn.local_window");
/* ... model is running with local_window attention ... */
wubu_adapter_t *new_attn = wubu_attn_sliding_create(...);
wubu_adapter_swap("attn.local_window", new_attn);  // atomic replace
/* ... model now uses sliding attention, no restart ... */
```

**Compile for speed, run for operation:**
- **Colonel-level (C11 design)**: opaque structs + function pointers
- **Device backends**: CUDA/Metal/Vulkan register their own impls
  via the same dispatch table — runtime swaps between CPU scalar,
  SIMD, CUDA, Metal, etc.
- **Shims**: backward-compat bridges let v1 adapters work with v2 core
