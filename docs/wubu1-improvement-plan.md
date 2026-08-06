# WuBu1 Hive-Mind Improvement Plan

> **Status**: `open` — plan, not yet implemented.
> **Date**: 2026-08-06
> **Directive**: "I want you to plan more model improvements, and base model
> design improvements, knowing that we can pretty much implement everything
> that we learned and that we have multiple research repositories for
> improvements. This is an operating system and an inference engine and it
> is a user experience because we have the KV cache as a file system. This
> means that the model not only has storage in the file system, but the model
> naturally learns to understand what every file is in the file system, as
> part of its context, which means we can train on any file inserted into the
> file system and create a bridge semantic context, dynamic until we understand
> the manifold completely as an AGI hive mind."

---

## 0. The insight that changes everything

The KV-cache-as-filesystem (AN16, `wubu_kvfs`) is not a memory optimization.

It is the **architectural pivot** that makes WuBu1 an AGI hive mind:

- **The model's context IS its storage.** Every file is a path in `/kv/`.
  Files don't get loaded and tokenized and then context-engineered — the file
  IS the context. The model reads it directly as KV tokens.
- **The model learns to understand its own filesystem.** Because attention
  operates over `/kv/in/*`, `/kv/synth/*`, `/kv/mem/*`, the model develops
  a model of what each file is, how they relate, which are stale, which are
  important. This is metacognition — the model thinks about its own storage.
- **Train on any file.** Insert a file into `/kv/in/` → it becomes training
  data immediately. The coherence score (attention mass + entropy + consistency)
  is the reward signal. This is the metabolism.
- **Bridge semantic context.** The model writes its "thoughts" about files
  to `/kv/synth/<file>_thought`. The attention pattern between `/kv/in/` and
  `/kv/synth/` is the bridge — a dynamic semantic map that evolves as the
  model learns.

This plan maps every improvement from our research repos onto implementations
we can ship now.

---

## 1. The three tracks

### Track A: Coherence-as-Training-Signal (the metabolism)

The model trains by inserting files into `/kv/in/`, running forward, then
measuring coherence. High coherence = the model understood the file. Low
coherence = grow toward that region (amoeba), or shrink the under-serving
blocks.

**Implementable now:**
- `wubu_kv_embedding.c` (Phase 1, already started) — the bridge layer.
- `wubu_coherence_reward.c` — a thin wrapper around
  `wubu_kv_embedding_coherence()` that the trainer calls after each forward.
- `wubu_fs_dataset.c` — walks `/kv/in/`, batch-encodes files into token
  streams, feeds to `wubu_forward`.

**Concrete next step:** finish `wubu_kv_embedding.c` + write `test_kv_embedding.c`
that verifies: (1) encode a file → lookup resolves to the right KV offset,
(2) coherence computation on a synthetic attention matrix gives correct
mass/entropy/score.

### Track B: The Amoeba Grows Toward Understanding (immune system)

The amoeba grow/shrink operators (AN01, `wubu_grow_insert_block` /
`wubu_shrink_remove_block`) currently grow/shrink **layers**. We extend them
to grow/shrink **KV blocks** toward high-entropy regions:

- **Diagnose**: `wubu_kv_embedding_coherence()` produces per-file coherence
  scores. Low scores → the KV region the file lives in is under-served.
- **Grow**: `wubu_gravity` (Euclidean attractor, AN06) routes a new block
  toward the under-served KV region. The model's attention to the new block
  improves coherence on the next forward.
- **Shrink**: `wubu_bi` (block importance, AN04) identifies dead KV regions
  (low utilization, high entropy) and prunes them.

**Implementable now:**
- `wubu_grow_kv.c` — grow a KV block toward a path, wire to coherence diagnosis.
- `wubu_shrink_kv.c` — prune an unused KV region.
- `wubu_kv_coherence_diag.c` — the diagnose hook (calls coherence on every
  file in `/kv/in/` after a forward pass).

**Concrete next step:** `wubu_grow_kv.c` + test — insert 10 files, measure
coherence, grow toward the worst 3, verify coherence improves.

### Track C: The Geometry Understands Files (hyperbolic KV)

WuBu nesting (THEORY/03) — product of Poincaré balls. Every KV block has a
curvature `c_i`. When a file lives at `/kv/in/deep/nested/path`, its KV
region sits deeper in the product manifold.

**Implementable now:**
- `wubu_poincare_kv.c` — attach curvature to KV blocks. Paths that are
  semantically related (e.g., `src/*.c`) get nearby curvature centers; the
  gyro-vector attention between them is amplified.
- `wubu_kv_embedding_encode_tokens()` already computes `kv_offset`; we add
  `kv_curvature` to the path record.

**Concrete next step:** Extend `kv_path_record_t` to carry curvature. Wire
`wubu_poincare_gqa_forward` to use it. Test: two related files should have
lower Poincaré distance between their KV regions than two unrelated files.

---

## 2. The 12 improvement modules (all implementable from current research)

| # | Module | File | Research | What it does |
|---|---|---|---|---|
| 1 | **KV embedding bridge** | `src/wubu_kv_embedding.c` | AN21 §1 | File→/kv/in, /kv/synth, coherence score |
| 2 | **Coherence reward** | `src/wubu_coherence_reward.c` | AN21 §1 | Trainer calls this → rewards coherent attention |
| 3 | **FS dataset loader** | `src/wubu_fs_dataset.c` | research/048 | Walks `/kv/in/`, batches files for training |
| 4 | **Pond streamer** | `src/wubu_pond_streamer.c` | research/048 | Real-time stream from research ponds to `/kv/in/` |
| 5 | **KV grow operator** | `src/wubu_grow_kv.c` | AN01/WB05 | Amoeba grows KV blocks toward under-served files |
| 6 | **KV shrink operator** | `src/wubu_shrink_kv.c` | AN04 | Prune dead KV regions (low utilization) |
| 7 | **KV coherence diagnose** | `src/wubu_kv_coherence_diag.c` | AN08 | Post-forward: score every file, feed to grow/shrink |
| 8 | **Hyperbolic KV** | `src/wubu_poincare_kv.c` | THEORY/03 | Attach Poincaré curvature to KV blocks, gyro-attention |
| 9 | **KV tiering** | `src/wubu_kv_tier.c` | AN02 | Tier KV blocks: hot=F32, warm=Q8_0, cold=IQ2_XXS |
| 10 | **KV immune audit** | `src/wubu_kv_audit.c` | AN08 | Load/save/forward/mutate audit hooks on KV paths |
| 11 | **Semantic router** | `src/wubu_semantic_router.c` | 055 | Route file queries to the right KV region via attention map |
| 12 | **KV-FS shell** | `src/wubu_kv_shell.c` | AN16 | Shell commands (`ls /kv/in/`, `cat /kv/synth/x`) operate on KV cache |

### Dependency graph (build order)

```
1. wubu_kv_embedding  (foundation — without this, nothing understands files)
         ↓
2. wubu_coherence_reward  (turns understanding into a training signal)
         ↓
3. wubu_fs_dataset       (training pipeline: files → tokens → model)
         ↓
5. wubu_grow_kv          (amoeba grows toward under-served KV regions)
         ↓
7. wubu_kv_coherence_diag (close the loop: forward → diagnose → grow)
         ↓
6. wubu_shrink_kv        (prune dead KV regions)
         ↓
8. wubu_poincare_kv      (hyperbolic geometry on KV paths)
         ↓
9. wubu_kv_tier          (precision per curvature region)
         ↓
10. wubu_kv_audit         (immune system for KV operations)
         ↓
11. wubu_semantic_router  (attention-based query routing)
         ↓
12. wubu_kv_shell         (the user-facing /kv/ filesystem)
```

Modules 4 (pond streamer) and 12 (shell) are parallel to the main track.

---

## 3. The training metabolism (how files become gradients)

```
  FILE INSERTED → /kv/in/                      [wubu_pond_streamer]
                   ↓
  ENCODE bytes → token floats → KV namespace    [wubu_kv_embedding]
                   ↓
  FORWARD through transformer                   [wubu_forward + wubu_poincare_gqa]
                   ↓
  MODEL ATTENDS OVER /kv/                        ← THE KEY INSIGHT
                   ↙         ↘
  coherence score              logits (next-token)
  (reward signal)              (standard LM loss)
                   ↓
  COMPOSITE LOSS = (1-α) * LM_loss + α * coherence_reward
                   ↓
  BACKWARD (via wubu_backprop, FD-verified)      [wubu_backprop]
                   ↓
  AMOEBA DIAGNOSE: low coherence on file X?       [wubu_kv_coherence_diag]
                   ↓
  GROW a KV block toward X's region               [wubu_gravity + wubu_grow_kv]
                   ↓
  SHRINK dead KV regions                          [wubu_bi + wubu_shrink_kv]
                   ↓
  5+1 recovery: archive / rollback bad mutations  [wubu_recovery]
```

The coherence reward (Track A) is the bridge between "the model
understands the file" and "the model gets gradient feedback for that
understanding." Files that produce high-coherence attention get reinforced;
files that don't trigger grow/shrink.

---

## 4. Cross-module synergies (where the research repos converge)

### 054 (mega-tokens) + KV-FS = the tokenizer is a KV organ

The amoeba tokenizer grows/shrinks based on corpus compression rate. But now,
the tokenizer's grow/shrink signal comes from KV coherence:

- **GROW token**: a new byte sequence appears in `/kv/in/` with low coherence
  (the model can't attend to it well with existing tokens). → append a token,
  re-encode.
- **SHRINK token**: a token at the vocab tail never gets attended to across
  any KV region. → prune it (tied head makes this free).

### 055 (architecture matrix) + KV-FS = the model routes by file

The semantic router (module 11) uses the architecture matrix from research/055:
7-hop Kevin Bacon analysis of how different file types relate. A `.c` file
and its `.h` file are cognitively adjacent in the 7-hop space — the router
pre-wires them to nearby KV blocks (via curvature in the Poincaré space).

### 050 (metabolism) + KV-FS = train until coherence plateaus

The metabolism says: grow until held-out loss plateaus, then shrink dead
params. With KV-FS, "loss" becomes "coherence" — grow KV blocks until
file-understanding plateaus, then prune the dead KV regions.

---

## 5. Base model design improvements (from wubu1-base-model-design.md)

The Wubu1 base model design doc (AN20) already specifies the key changes from
WuBu-35M. The KV-FS insight adds these refinements:

### 5.1 The canonical block carries KV metadata

`wubu1_block_t` (from the design doc) gets two new fields:
- `kv_curvature` — the Poincaré curvature for this block's KV region.
- `kv_precision` — the per-role precision for this block's KV writes.

This makes the base model geometry-aware from birth.

### 5.2 The embedding layer is dual-path

From `wubu1-base-model-design.md` §3.2: a single modality-agnostic encoder.
With KV-FS, the encoder is:
- `/kv/in/` → token embeddings (standard LM input)
- `/kv/synth/` → thought embeddings (model's own outputs, re-read as context)
- Cross-attention between in/synth IS the model's metacognition.

### 5.3 The optimizer is KV-aware

Muon (NS5) + KV-FS: the optimizer sees not just weight gradients, but
attention-coherence gradients. When a file in `/kv/in/x` produces low
coherence, the optimizer doesn't just update weights — it can grow a new
KV block (via the amoeba) toward that region. The optimizer itself participates
in the hive-mind metabolism.

### 5.4 The growth operator is KV-path-addressable

WuBu-35M grows by shifting `n_layers` (adding transformer blocks). With
KV-FS, growth is multi-dimensional:
- **Longitudinally**: add transformer blocks (the WuBu-35M way).
- **Laterally**: grow KV blocks toward under-understood files.
- **Depth-wise**: deepen the Poincaré nesting for complex file hierarchies.

The amoeba's Euclidean attractor (`wubu_gravity`) now has a dual: the
KV-gravity that pulls new blocks toward low-coherence paths.

---

## 6. The 5+1 recovery for KV-FS

When the model's understanding of a file goes wrong (attention misroutes,
coherence drops below threshold), the immune system (AN08) triggers:

1. **Detect**: `wubu_kv_coherence_diag` scores all files after each forward.
2. **Isolate**: quarantine the under-coherent file's KV region.
3. **Rollback**: the 5+1 recovery (AN08) rolls back to the last checkpoint
   where coherence was high.
4. **Diagnose root**: was it a bad attention pattern, a dead KV block, or
   a tokenizer gap?
5. **Heal**: grow a new KV block, add a token, or re-encode the file.

This is the AGI immune system: the model doesn't just learn from mistakes,
it *heals* its own storage state.

---

## 7. Registration

- `research/INDEX.md`: this doc becomes AN22 (KV-FS AGI hive mind improvement plan).
- `docs/wubu1-hive-mind-plan.md`: cross-reference this plan as the implementation
  roadmap.
- The 12 modules map to 12 todo items in the active task list.
