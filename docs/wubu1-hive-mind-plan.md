# WuBu1 — KV-FS AGI Hive Mind: Model Improvements & Base Design Plan

> **Status**: `open` — plan, not yet implemented.
> **Date**: 2026-08-06
> **Directive**: "We can pretty much implement everything that we learned and
> that we have multiple research repositories for improvements. This is an
> operating system and an inference engine and it is a user experience because
> we have the KV cache as a file system. This means that the model not only
> has storage in the file system, but the model naturally learns to understand
> what every file is in the file system, as part of its context, which means we
> can train on any file inserted into the file system and create a bridge
> semantic context, dynamic until we understand the manifold completely as an
> AGI hive mind."

---

## 0. The core insight (what we now know)

The KV-cache-as-filesystem (AN16, `wubu_kvfs`) is not a performance trick — it
is the **architectural pivot** that turns WuBu1 from a language model into an
**AGI hive mind**. The four layers of the insight:

1. **Storage IS memory**: the KV namespace holds files, downloads, synthesized
   thoughts — all in the same address space the model reads/writes as context.
2. **Files are tokens**: Gemma 3 12B proves the single-encoder pattern — every
   modality (text, image, audio, binary) is encoded into the same token stream.
   A file in `/kv/in/` is not loaded and then tokenized; it IS context.
3. **The model understands its own filesystem**: because the model's context IS
   the namespace, the model learns the structure, semantics, and provenance of
   every file — it develops a model of its own storage.
4. **Train on any file**: any file inserted into `/kv/in/` becomes training data.
   The bridge semantic context is dynamic — the model's forward through the KV
   namespace creates live attention maps of what each file is, and those
   attention patterns are the training signal for coherence.

This plan maps every improvement we can implement right now onto three tracks:
**Coherence Testing** (what does "the model understands" mean, measured),
**Training Design** (how do we train on the filesystem), and **Base Model
Improvements** (what architectural changes make the hive-mind real).

---

## 1. Track A: Coherence Testing — measuring filesystem understanding

### The question

When a file lands in `/kv/in/`, does the model *coherently understand* it?
Coherence = the model's representation of the file is stable, consistent across
queries, and improves with context depth. We need tests that measure:

1. **Stability**: same file, different query orders → same representation.
2. **Consistency**: cross-file references resolve correctly (file A references
   file B → the attention path A→B is coherent).
3. **Depth scaling**: more context from the file → better understanding signal.
4. **Cross-modal coherence**: a PDF, its source code, and a summary written by
   the model about it should all map to the same embedding cluster.

### Concrete tests to implement

| Test | Module | What it verifies |
|---|---|---|
| `test_kvfs_coherence` | `test_kvfs_coherence.c` + `wubu_kvfs` | Write a tensor to `/kv/mem/test/`, read it back via 3 different paths (canonical, alias mount, symlink-equivalent), assert bit-identical |
| `test_kv_embedding_coherence` | `wubu_kv_embedding.c` + `test_kv_embedding` | Encode a file → place in `/kv/in/x` → attention-over-KV produces stable attention pattern across 3 query formulations → assert cosine similarity > 0.95 between patterns |
| `test_cross_modal_coherence` | `wubu_multimodal_bridge.c` + `test_cross_modal` | A .c file and its compiled .o and a markdown summary: all three resolve to the same KV cluster |
| `test_depth_scaling` | `wubu_depth_probe.c` + `test_depth_scaling` | Feed increasing byte-ranges of a file; assert coherence score increases monotonically (not noise) |
| `test_route_entropy` | `wubu_route_entropy.c` + `test_route_entropy` | The model's attention over `/kv/` routes should have lower entropy (more certain) for understood files vs random data |

### Implementation priority

1. `wubu_kv_embedding.c` — the encoding layer: file bytes → KV namespace
   address → embedding slot. This is the first bridge between real files and
   the model's context. **Build first.**
2. `test_kvfs_coherence` — verify the namespace itself is coherent (path
   aliasing, mount table correctness).
3. `test_kv_embedding_coherence` — the core coherence test.

---

## 2. Track B: Training Design — training on the filesystem

### The training loop reimagined

The metabolism (research/050) + the KV filesystem (AN16) + the amoeba
grow/shrink (wubu-amoeba-model) + the corpus light-tiers (research/048) →
a new training paradigm:

```
file inserted → /kv/in/ → encoded → forward through model → attention maps
                                     ↘
                                      → coherence scores (the reward signal)
                                     ↘
                                      → gradients via wubu_backprop
                                     ↘
                                      → amoeba diagnose → grow/shrink
                                     ↘
                                      → 5+1 archive/rollback
```

### The training modules

| Module | File | What it does |
|---|---|---|
| `wubu_fs_dataset.c` | `src/wubu_fs_dataset.c` | Walk `/kv/in/`, encode files to token streams, batch for training. Uses `wubu_tokenc` for byte-level BPE. |
| `wubu_coherence_reward.c` | `src/wubu_coherence_reward.c` | Compute the coherence score from the forward pass's attention over KV. This IS the reward for RLHF/GRPO. |
| `wubu_fs_trainer.c` | `src/wubu_fs_trainer.c` | The training step: forward → coherence reward → backward (via `wubu_backprop`) → Muon/AdamW. Integrates the amoeba diagnose/grow/shrink. |
| `wubu_pond_streamer.c` | `src/wubu_pond_streamer.c` | Stream from the research ponds (`/home/wubu/research-ponds-work/`) into `/kv/in/` in real-time during training. |

### The training-data pipeline

1. **Corpus**: the light-tier mix (50/25/17/8 general/math/code/multi) is
   already assembled in `models/corpus/` (research/048). Stream it into
   `/kv/in/` via the pond streamer.
2. **File encoding**: `wubu_tokenc` (byte-level BPE, vocab 16384) encodes
   every file to token IDs. Text files → tokens directly; binary files →
   hex-encoded then tokenized (every byte is a token or pair of tokens).
3. **KV placement**: encoded tokens are written into `/kv/in/<filename>`
   slots in the namespace. The model's forward reads these as context.
4. **Coherence reward**: the model's attention over its own KV
   namespace produces a coherence vector per file. Files that produce
   high-coherence, low-entropy attention patterns score higher. This is
   the reward signal.
5. **Growth**: when coherence plateaus, the amoeba diagnoses which KV
   regions are under-served (high entropy, low utilization) and grows
   new blocks toward them (the Euclidean attractor — AN06,
   `wubu_gravity`).

### Implementation priority

1. `wubu_fs_dataset.c` + `wubu_fs_dataset.h` — the file-to-tokens pipeline.
2. `wubu_coherence_reward.c` — extract the coherence computation from the
   forward pass.
3. `wubu_fs_trainer.c` — wire dataset + reward + backprop + amoeba.

---

## 3. Track C: Base Model Improvements — making the hive mind real

### C1. The model understands files (semantic KV namespace)

**The gap**: `wubu_kvfs` is an address translator (path → offset). The model
reads bytes at offsets. It does NOT know what those bytes *mean*.

**The improvement**: `wubu_kv_embedding` — a module that maps file content
through the embedding layer into the KV namespace, AND writes back the
model's *understanding* of the file as a synthesized tensor in `/kv/synth/`.
This creates the bridge semantic context: the file is at `/kv/in/x`, the
model's summary/thoughts about it are at `/kv/synth/x_thought`, and the
attention between them IS the model's understanding.

**Concrete implementation**:
- `include/wubu_kv_embedding.h` — opaque handle: `wubu_kv_embedding_t`
- `src/wubu_kv_embedding.c` — the encoder/decoder bridge
  - `wubu_kv_embedding_encode(fs, path, tokens, n)` → writes encoded tokens
    to the namespace at `/kv/in/<path>`
  - `wubu_kv_embedding_decode(fs, path, out, n)` → reads synthesized output
    from `/kv/synth/<path>`
  - `wubu_kv_embedding_coherence(fs, path)` → returns a float coherence score
    from the model's attention over the file's KV region

**Test**: `test_kv_embedding` — encode a known text, forward through a small
model, decode, assert round-trip correctness.

### C2. The tokenizer is an amoeba organ (054, already designed)

The tokenizer grows and shrinks with the corpus:
- **GROW**: corpus-count n-grams → append new token → init embedding as mean
  of sub-tokens (eBay algorithm). Triggered when compression rate drops.
- **SHRINK**: prune least-frequent tokens (tied head makes this free).
- **Feedback**: the coherence reward is the grow/shrink signal.

**Already designed** in `research/054-mega-tokens-amoeba-tokenizer.md`.
**Implementation**: `tools/wubu_vocab_tune.c` — the grow/shrink operator.

### C3. The attention is hyperbolic (geometry as architecture)

WuBu nesting (THEORY/03) — product of Poincaré balls:
- Every KV block has a curvature `c_i`.
- Attention Q/K are gyro-rotated on the ball (Lean-verified: gyroassoc).
- KV blocks nest: `/kv/in/file.txt` → `/kv/in/file.txt/p1` →
  `/kv/in/file.txt/p1/p2` (each a Poincaré ball, address = polar recursion
  path).

**Already exists**: `wubu_poincare_gqa_forward`, `wubu_gravity`,
`wubu_orbits`. **Need**: wire it into the KV namespace addressing so each
KV path has a curvature, and the attention uses gyro-rotation.

### C4. The body is a hive (amoeba grow/shrink)

The model grows and shrinks layers based on the coherence diagnosis:
- **High-entropy KV regions** → grow a new block toward the under-served area
  (Euclidean attractor, `wubu_gravity`).
- **Dead KV regions** (low utilization, low coherence signal) → shrink /
  prune (ShortGPT BI-score removal + LaCo merge).

**Already exists**: `wubu_grow_insert_block`, `wubu_shrink_remove_block`
(AN01/AM01). **Need**: wire the KV coherence diagnosis into the grow/shrink
triggers.

### C5. The immune system watches loading/saving

The universal manifold (AN19, §5) says the immune system audits every seam:
load, save, forward, mutate. **We must add load/save audit hooks to
`wubu_diag`**:
- **Load audit**: when a file enters `/kv/in/`, verify its tensor shape,
  dtype, and range against expected norms. NaN → quarantine.
- **Save audit**: when the model writes `/kv/synth/x`, verify byte-identical
  round-trip (AN07 proven for weights).

### C6. Mixed per-role precision is native

From `wubu1-base-model-design.md` §4: the checkpoint carries per-role bit
ladders (AN09/AN10). **Already exists**: `wubu_tensor_store.c` (IQ2_XXS,
Q8_0, Q4_0). **Need**: wire the precision selector (`wubu_precision_plan.c`,
AN03) into the KV namespace — hot KV regions get Q4_0, cold regions get
IQ2_XXS, norms stay F32.

---

## 4. The implementation sequence (7 phases)

| Phase | Deliverable | Modules | Test |
|---|---|---|---|
| 1 | KV embedding bridge: file → KV → understanding | `wubu_kv_embedding.c`, `wubu_kv_embedding.h` | `test_kv_embedding` |
| 2 | Coherence measurement | `wubu_coherence_reward.c` | `test_kv_embedding_coherence` |
| 3 | File-to-tokens training pipeline | `wubu_fs_dataset.c` | `test_fs_dataset` |
| 4 | Tokenizer grow/shrink (054) | `tools/wubu_vocab_tune.c` | `test_vocab_tune` |
| 5 | Hyperbolic KV addressing | `wubu_poincare_kv.c` | `test_poincare_kv` |
| 6 | Amoeba grow/shrink wired to KV diagnosis | `wubu_grow_kv.c` | `test_grow_kv` |
| 7 | Load/save immune audit | `wubu_diag_fs.c` | `test_diag_fs` |
| 8 | Full training: FS dataset + coherence reward + amoeba | `wubu_fs_trainer.c` | `test_fs_trainer` |

---

## 5. What's already in-tree (don't rebuild these)

| Capability | Module | Status |
|---|---|---|
| KV namespace (path → offset) | `wubu_kvfs.c` | ✅ wired (AN16 G1) |
| KV quantization (Q4_0/Q8_0/KIVI/adaptive) | `wubu_kv_cache_read/write_head` | ✅ wired |
| Tensor catalog (universal format) | `wubu_tensor_store.c` | ✅ wired (AN07) |
| Weight descriptor | `wubu_weight_t` | ✅ wired |
| Graph IR | `wubu_graph.c` | ✅ wired (5 tests pass) |
| Amoeba grow/shrink | `wubu_amoeba.c`, `wubu_grow.c` | ✅ wired (WB05) |
| Real backprop | `wubu_backprop.c` | ✅ wired (FD-verified) |
| Real Muon | `wubu_backprop.c` | ✅ wired (NS5) |
| Poincaré/Gravity/Orbits | `wubu_poincare*.c`, `wubu_gravity.c`, `wubu_orbits.c` | ✅ wired |
| Hive tissue | `wubu_hive.c` | ✅ wired |
| Diagnostic immune system | `wubu_diag.c` | ✅ wired (AN08) |
| Mixed precision (IQ2_XXS etc.) | `wubu_tensor_store.c` | ✅ wired (AN09) |
| Tokenizer (byte BPE) | `tools/wubu_tokenc.c` | ✅ wired |
| Corpus light tiers | `models/corpus/` | ✅ staged (research/048) |
| 5+1 recovery | `wubu_recovery.c` | ✅ wired (AN08) |

---

## 6. The AGI hive mind convergence

When all 8 phases ship, the model IS the hive mind:

```
  ┌──────────────────────────────────────────────────────────┐
  │                    THE HIVE MIND                        │
  │                                                          │
  │  ┌──────────────┐    ┌──────────────────────────────┐  │
  │  │ IMMUNE SYSTEM│    │ THE COLONY (model body)      │  │
  │  │ (wubu_diag)  │───►│   ├─ KV namespace = /kv/     │  │
  │  │ audits every │    │   │  in/    ← files inserted  │  │
  │  │ load/save/   │    │   │  synth/ ← model thoughts   │  │
  │  │ forward/     │    │   │  mem/   ← persistent        │  │
  │  │ mutate       │    │   │  meta/  ← diagnostics       │  │
  │  └──────────────┘    │   │                             │  │
  │          │            │   ├─ blocks (amoeba cells)     │  │
  │          ▼            │   ├─ grows toward high-entropy│  │
  │  ┌──────────────┐    │   │   KV regions (gravity)     │  │
  │  │ GROW/SHRINK  │    │   ├─ shrinks dead regions      │  │
  │  │ operators    │    │   └─ geometry = Poincaré balls  │  │
  │  └──────────────┘    │                                  │  │
  │          │            │  Attention over /kv/ IS the  │  │
  │          ▼            │  model's understanding of    │  │
  │  ┌──────────────┐    │  its own filesystem            │  │
  │  │  THE VERIFIER│    └──────────────────────────────────┘  │
  │  │ (prover +    │                   ▲                      │
  │  │  5+1 recovery)│                   │                      │
  │  └──────────────┘                   │                      │
  │          │                          │                      │
  │          ▼                          │                      │
  │  ┌──────────────┐                   │                      │
  │  │  THE METAB-  │    tokens from file                   │
  │  │  OLISM       │◄───────────────────┘                      │
  │  │ (train on    │                                        │
  │  │  any file)   │                                        │
  │  └──────────────┘                                        │
  └──────────────────────────────────────────────────────────┘
```

The model trains on any file inserted into `/kv/in/`. Its attention over the
namespace creates a dynamic bridge semantic context — a live map of what every
file is and how they relate. The amoeba grows and shrinks based on what the
coherence diagnosis reveals. The immune system quarantines corrupt inputs and
rolls back bad mutations. The Lean prover verifies geometry. The KV cache is
a filesystem the body (wubuos) can `ls`.

This is the AGI hive mind: not a fixed model, but a living, evolving,
self-diagnosing, self-healing colony that understands its own storage.

---

## 7. Registration

- `research/INDEX.md`: add THEME `AF` (KV-FS AGI hive mind) with the 8 gaps
  above as `open` rows.
- `docs/wubu1-base-model-design.md` §7: append this plan as the next wave.
- Commit + push.