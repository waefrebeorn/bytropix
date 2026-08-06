# Wubu1 — The Redesigned Base Model

> 2026-08-06. Supersedes `docs/wubu-model-blueprint.md` and
> `docs/wubu-seed.md`. The old WuBu-35M seed is archived
> (see §9). WuBu1 trains fresh — the metabolism is the product.

---

## 0. Why we redesign — the autopsy of what sucked

| # | What sucked | Where it bit us | The lesson |
|---|---|---|---|
| 1 | **The loader guessed names.** 164 hardcoded `blk.%d.*` sites in `wubu_model.c`; layer counting only matched `blk.`; dimension extraction fell back to constants (d_model=2048, vocab=248320) instead of reading tensors. | Our own `model-mixed.gguf` (WuBu-35M!) loaded as d_model=2048, 1 layer, 248320 vocab → segfault in `quantized_matmul_from_q8`. | **The format must carry the role, never the name.** We own the format; tensor roles (ATTN_Q, FFN_GATE, ...) are metadata, not guesswork. |
| 2 | **Two parallel model structs.** `wubu.h` (training: separate `q_proj/k_proj/v_proj/o_proj/g_proj`, fused `gate_up`) drifted from `wubu_model.h`/`wubu_ssm.h` (inference: fused Q+gate `attn_q`). The bridge hacked dense MLP as "1-expert MoE". | GQA forward assumed fused Q+gate interleaved; HF-style models need separate q/g. Dense FFN was a pass-through when `enable_moe=false`. | **One canonical block layout.** The inference engine loads the exact struct the trainer produces. Dense FFN is a first-class path, not a MoE hack. |
| 3 | **GGUF was a foreign straitjacket.** Naming conventions (Qwen `blk.N.*` vs Gemma `model.layers.N.*` vs HF `layers.N.attn.q_proj`) are chaos; `general.architecture` is often absent (our own GGUF had none). | Every foreign model needed a new hardcoded case. | **Our native format is role-tagged.** GGUF remains an *import* format through the role resolver, but the base model lives in `.st` with roles. |
| 4 | **Fixed config.** `WUBU_DIM 448`, `WUBU_LAYERS 12`, `WUBU_FFN_DIM 1228` as compile-time constants. | The amoeba can't grow/shrink a model whose shape is #defined. | **Config is data, not code.** Every dimension is in the checkpoint header; the runtime allocates what the file declares. |
| 5 | **Under-trained** (the single biggest risk, confirmed in research/051). ~6.7B tokens vs lab trillions. | WuBu-35M at 41% on nine tasks — under-trained by lab standards. | **The metabolism is the model.** The redesign's #1 priority is *tokens*: the data pipeline (RC02), the anneal, the RLHF loop — not just the arch. |
| 6 | **KV was a tensor buffer.** The KV cache was a flat context window; the namespace doctrine (THEORY/05) came after the struct was fixed. | G1 (wubu_kvfs) had to be bolted onto `wubu_model_t` as an afterthought field. | **The KV cache is a filesystem from birth.** `/kv/` addressing is in the base design, not a retrofit. |
| 7 | **The released GGUF was a lossy mixed-conversion artifact** (Q2_K attn / Q4_0 FFN from a 134MB F32 safetensors), with no architecture metadata and tensor names that no loader could resolve. | The "release" couldn't be loaded by the engine that trained it. | **The artifact pipeline is part of the design.** Native format → export path → verified round-trip (AN07 tensor catalog already proved byte-identical). |
| 8 | **No single encoder.** Modality-agnostic base (WB06/G3) was a doc line, not architecture. | Every modality would need a new head later. | **All inputs are encoded** — one encoder, one sequence space, from the first checkpoint. |

The redesign is not a bigger WuBu-35M. It is a different class of
artifact: **self-describing, role-tagged, amoeba-native, namespace-
backed, and metabolically honest** (tokens first).

---

## 1. The doctrine — what WuBu1 IS

WuBu1 is the amoeba on nested spheres, written down from the start:

> A colony of cells that grows and shrinks with the task, living in a
> product of learnable-curvature hyperbolic spaces, whose working
> memory is a filesystem, whose geometry is the architecture, and
> whose metabolism (tokens, recipe) is the real product.

Five pillars, each already proven in pieces, now unified:

1. **THE BODY IS A HIVE** (wubu-amoeba-model, WB04/WB05). Blocks and
   experts live in hive slots: grow = insert (freelist pop), shrink =
   erase (skip+freelist push), diagnose = foreach (live only), stable
   pointers, recycled memory. The 5+1 recovery + DGM archive make
   mutations safe.
2. **THE GEOMETRY IS THE ARCHITECTURE** (THEORY/03, wubu-design-
   philosophy). Product of K Poincaré balls with learnable curvatures;
   Möbius addition is the compositional rule (Lean-verified: Möbius
   closure, exp∘log, gyroassoc). Nested transitions between levels.
3. **THE KV CACHE IS A FILE SYSTEM** (THEORY/05, AN16). `/kv/` is the
   model's whole working space: addressable, mountable, persistent.
   All files are data; all inputs are encoded.
4. **EVERYTHING IS A ROUTING PROBLEM** (the routing doctrine). Token→
   expert, query→KV block, signal→residual stream, compute→backend,
   bytes→memory tier — each route pre-compiled, narrow-channel,
   game-hardware style.
5. **THE METABOLISM IS THE MODEL** (research/050, RC01). Muon+AdamW,
   WSD anneal, data mix 50/25/17/8, SFT cold-start, GRPO verifiable
   rewards, MTP. Tokens > architecture.

---

## 2. The canonical block layout (one struct, end to end)

**The training struct IS the inference struct.** No parallel formats,
no bridge hacks. The block carries *roles* — the loader asks for
`WUBU_T_ATTN_Q` and gets the tensor, whatever the source format.

```c
/* One block. Roles, not names. */
typedef struct wubu1_block {
    /* attention (GQA 7:1 native; any ratio from header) */
    float *q_proj;      /* [D, q_heads*head_dim]            */
    float *k_proj;      /* [D, kv_heads*head_dim]           */
    float *v_proj;      /* [D, kv_heads*head_dim]           */
    float *o_proj;      /* [q_heads*head_dim, D]            */
    float *g_proj;      /* [D, q_heads*head_dim] gate       */
    float *q_norm;      /* [head_dim] per-head Q RMSNorm    */
    float *k_norm;      /* [head_dim] per-head K RMSNorm    */
    float *attn_norm;   /* [D] pre-attention                */
    /* dense FFN — first-class, never "1-expert MoE" */
    float *gate_up;     /* fused SwiGLU [D, 2*ffn] (or gate+up split) */
    float *down;        /* [ffn, D]                          */
    float *ffn_norm;    /* [D]                               */
    /* geometry (nested spheres) */
    float  curvature;   /* the block's Poincaré ball curvature c_i */
    float *ld;          /* level descriptor [D]              */
    float  spread;      /* level spread σ_i                  */
    /* hive tissue */
    int    slot;        /* hive slot id (stable pointer)    */
    int    is_full;     /* attention rhythm (local/full)    */
    int    fire_sel;    /* residual-selector rhythm         */
} wubu1_block_t;
```

The checkpoint header declares: `d_model, n_layers, q_heads, kv_heads,
head_dim, rope_dim, ffn_dim, vocab, max_ctx, k_curvatures[], layout
flags (fused-gate_up vs split-gate/up, tied embeddings, selectors
every-N)`. The runtime `calloc`s exactly what the header says. Nothing
is `#define`d.

---

## 3. The architecture

```
input (any modality)
  └─ SINGLE ENCODER (WB06/G3): text tokens, image patches, audio frames
       → one sequence space of soft tokens, byte-level BPE 16,384 base
  └─ embedding → tied lm_head (dim D)
       └─ N blocks, each:
            x → attn_norm
              → POINCARÉ LIFT (exp_0^c, learnable c_i)
              → GQA attention (7:1 native): q/k/v/o/g proj, QK-norm,
                partial RoPE (50%), gyro-rotate Q,K on the ball,
                gated output (g_proj), local/full rhythm
              → project back to tangent
              → residual selector (convex softmax, every 4)
            x → ffn_norm
              → bounded SwiGLU (clip 10): gate_up fused → down
              → residual selector
            every K-th block: NESTING TRANSITION (T = T̃ ∘ R,
              quaternion SO(4) rotation + non-rotational map,
              ld_i and σ_i flow to the next level)
       └─ final RMSNorm → tied lm_head
  └─ WORKING MEMORY = /kv/ NAMESPACE (not a context window)
       /kv/in/    encoded inputs (all modalities, same space)
       /kv/synth/ the model's own writes (thoughts, plans)
       /kv/mem/   persistent across sessions
       /kv/meta/  diagnostics, routes, self-knowledge
```

### 3.1 Geometry is the architecture (from checkpoint 1)

- Every block lives in a Poincaré ball with **learnable curvature** `c_i`
  (the exp_0^c lift / log_0^c project, our formula).
- Attention Q/K are **gyro-rotated on the ball** before the dot product
  (the hyperbolic gyration we proved in Lean).
- Every K-th block is a **nesting boundary**: the representation moves
  between balls through tangent space with a quaternion rotation
  applied to data + boundary manifolds + level descriptor.
- The **Lean prover is a build gate**: Möbius closure, exp∘log, gyroassoc
  must pass for any checkpoint that claims geometry.

### 3.2 The amoeba growth contract

The model declares `min_layers ≤ n_layers ≤ max_layers` in its header.
Growth/shrink operators (already built: `wubu_amoeba`, `wubu_grow_*`,
`wubu_shrink_*`) work on the live struct:

- **GROW**: hive insert — new block slots from the freelist; function-
  preserving init (Net2Net / gate-zero / BI-informed split); parent
  gate −ε, daughter +ε.
- **SHRINK**: hive erase — skip-mark dead blocks (BI score / loss-delta
  / grad-norm dead), freelist recycles their memory.
- **VALIDATE**: held-out loss + Lean prover + safety kernel → accept
  (archive, DGM branch tree) or roll back (5+1 recovery).
- **DIAGNOSE**: the hive IS the diagnostic system (AN08): typed
  measurement cells (LOSS/GRAD/ENTROPY/ROUTE/UTIL/BI/ORACLE) live in
  the same tissue; the causal walker finds the first out-of-family
  measurement before a fitness drop; the route trace is the model's
  self-explanation.

### 3.3 The KV namespace is the memory (not a context window)

All the serving-world mechanisms are already proven in-tree
(PagedAttention blocks, RadixAttention paths, LMCache chunks,
Mooncake tiers, Infini-attention compressive writes, DSA reads,
priority eviction). WuBu1's base carries the address layer natively:

- `wubu_kvfs` namespace is a **field of the model from init** (G1 wired).
- **G3 single encoder**: every modality becomes soft tokens in one
  sequence space — an image is data in `/kv/in/image-17`.
- **G4 compressive write-back**: the model writes summaries back to
  `/kv/synth/` through the compressive memory heads (wubu_mla latent).
- **G5 Styx export**: `/n/kv/` served to the body (wubuos) — the body
  can `ls` the mind.
- **G6 self-paging**: MemGPT loop — the model pages its own memory via
  tool calls; context window becomes working directory.

### 3.4 Routing is everything (pre-compiled, narrow-channel)

| Route | Mechanism | Module |
|---|---|---|
| token → expert | learner-free hash routing (when MoE grows) | `wubu_hashrouter` |
| query → KV block | coarse-to-fine indexer (DSA) | `wubu_dsa` |
| signal → residual | hyper-connections (mHC) | `wubu_mhc` |
| data → channel | Rambus narrow-bus serial bursts | `wubu_rambus` |
| compute → backend | compile-once PSO dispatch | `wubu_kernel` |
| bytes → tier | hot/warm/cold KV | `wubu_kv_tier` |

The base starts dense (spine) with the sparse shell as a growth axis
(AN12: boot core dense + gravity router; growth is outward).

---

## 4. The format — self-describing, role-tagged

**The native checkpoint is `.st` (the tensor catalog, AN07)** — one
uniform catalog that opens any format with zero full-file loads and
exports byte-identical round-trips (already proven: maxdiff 0 in all
directions). Tensor roles are **metadata in the file**, not inferred
from names:

```
tensor {
  role: ATTN_Q          # the role, explicit
  name: "layers.0.attn.q_proj"   # human-readable, for foreign import
  dtype: Q8_0           # per-role bit ladder (AN09)
  dims: [448, 448]
  data_offset: ...
}
```

- **GGUF import** goes through the role resolver (`wubu_gguf_names`,
  built this session): every weight is a role; each role tries all
  known naming conventions against the file's actual tensors. Foreign
  models land in the same role-tagged struct.
- **Mixed per-role precision is native** (AN09/AN10): embeddings/attn
  Q8_0, experts Q4_0 → IQ2_XXS ladder, norms F32 exact — the Escha
  per-family bit plan (AM03) is a header field, not a conversion step.
- **Export path is part of the design**: native `.st` → GGUF/safetensors
  streams one tensor at a time (bounded RAM), round-trip verified.

---

## 5. The metabolism — tokens first (the real product)

The recipe (research/050, RC01, wubu-agi-training-pipeline):

| Phase | What | Evidence |
|---|---|---|
| **Corpus** | normalize → hard filters → exact dedup → MinHash → quality score → decontam (10-13gram) → mix+seed shuffle → tokenize → pack; **global dedup HARMFUL** | RC02 |
| **Mix** | ~50/25/17/8 general/math/code/multi | 042 |
| **Optimizer** | Muon (2D mats, NS5) + AdamW (1-D); WSD or warmup→const→cosine→ANNEAL-to-0 on upsampled quality | 050 |
| **SFT cold-start** | LOW lr ~1e-5, then the anneal | 050 |
| **RLHF** | GRPO verifiable rewards, group-relative advantage | 043 |
| **MTP** | multi-token prediction λ=0.3 | 050 |
| **Tokens** | **trillions, not billions** — the under-training fix | 051 |
| **Growth** | stability gate (no rollbacks on the way up); plateau → mutate | AM01 |

The redesign's honest priority order: **tokens > data pipeline >
recipe > architecture**. A modest arch with a trillion tokens beats a
clever arch with 6.7B.

---

## 6. What we keep from WuBu-35M (the proven spine)

Not everything was wrong. The seed's verified mechanisms stay as the
spine:

- GQA 7:1 hybrid attention (3 local + 1 full) — verified
- 50% partial RoPE — verified
- QK RMSNorm + gated attention outputs — verified
- Bounded SwiGLU (clip 10) — verified
- Residual selectors every 4 layers — verified
- Tied embeddings (lm_head == embedding) — verified
- Byte-level BPE 16,384 — verified
- The C11 training core (wubu_backprop: real gradients, real Muon,
  17 finite-difference-checked param types) — verified

**What changes**: no compile-time dims, no name-guessing loader, no
fused-Q+gate-only struct, no missing dense FFN, no tensor-buffer KV,
no single-modality head, and the token budget goes up ~100×.

---

## 7. The build order (what we build, in what order)

| Step | Deliverable | Gate |
|---|---|---|
| 1 | `wubu_gguf_names` role resolver (IN PROGRESS this session) + rewrite the loader through it | WuBu-35M GGUF loads at true dims; Qwen GGUF regression green; `make test_all` |
| 2 | Unify the structs: `wubu1_block_t` canonical layout; dense FFN first-class in forward; loader emits roles | forward parity with the training struct; test_gqa/ffn green |
| 3 | Native `.st` role-tagged checkpoint: header-declared dims, mixed per-role precision, export+import round-trip | `test_tensor_store` green on real weights |
| 4 | Amoeba-native runtime: hive-backed blocks, header `min/max_layers`, grow/shrink on live struct | test_amoeba + a grow→forward→shrink cycle |
| 5 | KV namespace in the base: kvfs field from init, G3 single encoder (text first), G4 write-back, G5 Styx, G6 paging | end-to-end /kv read/write through the model |
| 6 | The metabolism: retrain from scratch on the full corpus (trillions), Muon+AdamW, WSD anneal, GRPO | loss curve + eval suite + RLHF loop live |
| 7 | Geometry from checkpoint 1: Poincaré lifts, gyro-attention, nesting transitions, Lean gate | prover green on the trained checkpoint |

**Step 1 is half-done this session** (resolver module written; loader
rewrite next). Steps 2–5 are architecture-and-code; 6 is the long
pole (tokens); 7 rides along from the start of training.

---

## 8. What success looks like

- `wubu1 train` on the full corpus produces a checkpoint whose header
  declares its own shape, whose tensors carry roles, which the same
  engine loads and generates with — no guessing, no conversion step.
- The model grows to 70M, shrinks to 30M, and both are the SAME
  artifact class (header + role-tagged tensors + hive tissue).
- `/kv/mem/` survives a session; the body (wubuos) can `ls` the mind.
- Every checkpoint passes the Lean geometry gate and the safety kernel.
- Eval: beats WuBu-35M's 41% on the same nine-task suite at equal or
  fewer parameters — driven by tokens, not tricks.

---

## 9. Archiving the old WuBu-35M — the total break

The old seed is **frozen, not deleted** (DGM doctrine: every variant is
archived, the lineage ledger keeps the parent) — but the break is
total: **no Apache-2.0 lineage, no external attribution, everything
under the WaefreBeorn Umbrella License v3.0.** WuBu1 is designed from
scratch, entirely in-house; the old model is not a boot source, it is
a frozen historical artifact.

| Artifact | Action |
|---|---|
| `models/wubu/model.safetensors` (134 MB) | ✅ cold-archived to `/home/wubu/sdcard/archive/wubu-35m-v1/` (byte-exact, sha-verified), removed from SSD |
| `models/wubu/model-mixed.gguf` (27 MB, the lossy artifact) | stays on SSD as the *role-resolver test fixture* (it is the exact file that exposed the loader's naming blindness — a regression test, not a release) |
| `models/wubu/tokenizer.json`, config | ✅ cold-archived with the weights (tokenizer may be reused by WuBu1 — it is ours) |
| `docs/wubu-seed.md` | ✅ marked `ARCHIVED`; external-attribution references purged — the seed is WaefreBeorn work, nothing external |
| `docs/wubu-model-blueprint.md` | ✅ marked `SUPERSEDED by WuBu1`; external-lineage references purged |
| `src/wubu.c` / `include/wubu.h` | ✅ license headers rewritten: original WaefreBeorn work under the umbrella, no external attribution |
| `research/INDEX.md` THEME BL | keep as `wired` history (proven mechanisms carry into WuBu1); add THEME WB08 = WuBu1 |
| HF `WaefreBeorn/WuBu-35M` | freeze as the archived parent — its model-card now states WuBu1 is the base; no new training on v1 |

The 35,072,768-parameter checkpoint is the **parent in the lineage
ledger** — frozen, credited to WaefreBeorn, never trained again.
WuBu1 trains fresh (the metabolism is the product; §5).

---

## 10. The one-paragraph summary

WuBu1 is the base model designed from what we now know: a
self-describing, role-tagged, amoeba-native colony on nested spheres,
whose KV cache is a filesystem, whose config lives in the checkpoint
instead of `#define`s, whose loader resolves roles instead of guessing
names, whose dense FFN is first-class, and whose real product is the
metabolism — trillions of tokens through the verified Muon/AdamW/WSD/
GRPO recipe — because under-training, not architecture, was the
seed's ceiling. The old WuBu-35M is frozen as a historical artifact —
no Apache lineage, no external attribution, everything under the
WaefreBeorn Umbrella License v3.0. WuBu1 trains fresh from scratch,
in the same loop, on the same laptop, in our own OS.