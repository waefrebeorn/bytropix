# THE UNIVERSAL MANIFOLD — one mathematical substrate for every model

> 2026-08-06. User directive: "we need to go through pretty much every
> part of our model design, every part of our model loading, every part
> of our model saving, and every part of our model fine-tuning — and it
> needs to be adaptable and diagnostic like an amoeba, because we're
> gonna have to load music models, video models, diffusion models,
> coders, decoders, generative adversarial networks. We're basically a
> mathematical engine subtext type."
>
> This document is the research synthesis (online, 2026-08-06) + the
> design that follows. It does NOT describe new per-architecture
> loaders — it describes the ONE substrate every format and every
> architecture reduces to.

## 0. The one-sentence thesis

**Every model — LLM, music, video, image diffusion, coder, decoder, GAN,
encoder — is the same thing at the substrate level: a directed graph of
typed tensor operations, with weights stored in a type-tagged tensor
container, trained by delta-adapter fine-tuning on top of a frozen
backbone, and watched by a diagnostic immune system.**

That is the "mathematical engine subtext." Everything else — architecture
names, modality heads, training regimes — is a layer of sugar above it.
wubuwizard already has most of the substrate. This document maps the
rest and proves the claim with the world's designs.

---

## 1. What the world actually does (the online research, 2026-08-06)

### 1.1 Containers: GGUF and safetensors are already universal

**GGUF** (Georgi Gerganov Universal Format, `ggml-org/ggml` spec) is a
*typed tensor container*: header (magic, version, tensor count, KV
metadata) → tensor infos (name ≤64B, n_dims ≤4, dims[], ggml_type,
data_offset) → one mmap-able data blob. The ggml_type enum is the
canonical type space: F32=0, F16=1, Q4_0=2 … IQ1_M=29, BF16=30,
TQ1_0=34, TQ2_0=35, MXFP4=39, NVFP4=40, Q1_0=41, TQ3_1S=45, TQ4_1S=46,
Q2_0=47. **Nothing in GGUF says "language model"** — the container
stores any tensor graph. It is a universal manifold *format*.

**safetensors** (HuggingFace) is the zero-copy sibling: JSON header
mapping names → shape/dtype/data_offsets, then a concatenated blob.
Same contract as GGUF minus the type table: **a name-addressable tensor
catalog with per-tensor dtype + shape + offset**.

**What we already have**: `gguf_reader.c` (GGUF v3, all types incl.
BF16/MXFP4/NVFP4/TQ, `gguf_dequantize` universal dequantizer,
`quantized_matmul` type-agnostic dispatcher, 137/137 tensors verified
bit-exact vs the reader path) and `wubu_tensor_store.c` — the ONE
catalog that opens safetensors/GGUF/.st with zero full-file loads and
streams byte-identical round-trips in all directions (AN07, `wired`).
Plus `wubu_weight.c` (2026-08-06): the single `wubu_weight_t`
descriptor `(data, type, n_elems)` every loader fills identically.

### 1.2 Graphs: ONNX and MLIR — the "subtext" is a typed op graph

**ONNX** (Open Neural Network Exchange): a *graph IR*. Nodes are ops
(Gemm, Conv, LayerNorm, Softmax…), edges are typed tensors, and the
model file = graph + initializer weights. ONNX Runtime loads ANY ONNX
model (BERT, UNet, GAN, Whisper, DiT) with the SAME engine: convert to
in-memory graph → provider-independent optimizations → per-node
dispatch to registered execution providers (CUDA/CPU/DirectML/…).

**MLIR** (Multi-Level IR, LLVM): the lesson is *progressive lowering
with dialects*. One IR, many abstraction levels (linalg → affine →
LLVM), and any domain gets a dialect. **The hourglass pattern**: one
neck (the IR), wide inputs (all frameworks), wide outputs (all
hardware). AN15 already proved this at the ISA level (llama.cpp runs
one GGUF on 15 backends) and the driver space (one MIR, N ISA drivers).

**What this means for us**: the missing piece in wubuwizard is the
**graph layer** — a `wubu_graph` IR where a model is nodes (ops) +
edges (typed tensors) + a topological order. Today the engine binds
tensors by NAME into hardcoded structs (`gqa_layer_weights`,
`ssm_layer_weights`) — that is the per-architecture coupling the
universal manifold must break. With a graph IR, a MusicGen checkpoint
and a DiT checkpoint load through the SAME loader; only the op set
differs, and the op set is just a dispatch table.

### 1.3 Modalities: every domain is a tokenizer + one backbone

The convergence is remarkable — **every modality now reduces to
"encode to discrete tokens / patches → run ONE transformer backbone →
decode."**

| Domain | Model | The reduction |
|---|---|---|
| Text | LLMs | vocab tokens → transformer |
| Music/audio | **MusicGen** (Meta, 2306.05284) | audio → EnCodec neural codec → discrete RVQ tokens → single-stage autoregressive transformer LM |
| Image | **DiT** (2212.09748) | latent VAE patches → ViT backbone with adaLN-Zero |
| Video | **Sora** (OpenAI) | video → latent → spacetime patches → diffusion *transformer* |
| Audio+video | AV-DiT (2502.03897) | joint audio-video DiT, task tokens |
| Vision | ViT, SigLIP | pixel patches → transformer |

The backbone is the SAME transformer math; only the codec/head changes.
**This is the amoeba body**: one core, swappable sensory pseudopods.

### 1.4 Fine-tuning: deltas are already universal

**LoRA** (2106.09685) and **PEFT** (HuggingFace): rank-decomposed delta
matrices W′ = W + (α/r)·B·A applied to ANY weight tensor, any
architecture. **"LoRA works on any deep learning model architecture"**
(community consensus; also demonstrated on diffusion models —
ProLoRA 2506.04244 transfers LoRA adapters between diffusion models
without retraining). The delta is a *tensor*, stored in the same
container formats, applied at the same matmul seam. **Fine-tuning is
not per-architecture either** — it is a delta-adapter algebra over the
universal weight descriptor. wubuwizard already has `wubu_lora.c`
(BTL-3 two-step orchestration `wired`, `make test_btl3_lora`).

### 1.5 GANs: a training regime, not a new substrate

GAN = generator network + discriminator network, adversarial min-max
loss. Both networks are **ordinary graphs of tensor ops** — the
substrate is identical; only the loss and the two-body training loop
differ. Encoder-decoder GANs (Auto-Encoding GANs) add an encoder —
again, three ordinary graphs. **No new loading/saving/fine-tuning
mechanism is needed**; only a training-regime descriptor.

### 1.6 Self-healing: detect → diagnose → heal (the amoeba immune system)

The industry's self-healing ML pipelines (6-layer control loop:
inference → detection (KS-tests, z-scores, Bayesian uncertainty) →
healing action selection (retrain / rollback / fallback) → validate →
deploy → log) match wubuwizard's existing `wubu_diag` hive diagnostic
system exactly: ring-bounded measurement cells, z-score anomaly
detection, the causal walker finding root cause, the 5+1 rollback.
**The amoeba immune system is already wired** (AN08). The gap: it must
watch *loading and saving* too, not just training.

---

## 2. The design: one substrate, five layers

```
┌────────────────────────────────────────────────────────────────┐
│ 5 DIAGNOSTIC IMMUNE SYSTEM (wubu_diag)                          │
│    audits every seam: load / save / forward / backward / mutate │
├────────────────────────────────────────────────────────────────┤
│ 4 FINE-TUNING: delta-adapter algebra                            │
│    LoRA / full / frozen — deltas are tensors on wubu_weight_t   │
│    any training regime (SL, GRPO, GAN min-max, diffusion)       │
├────────────────────────────────────────────────────────────────┤
│ 3 THE GRAPH IR (wubu_graph)  ← THE MISSING PIECE               │
│    nodes = ops, edges = typed tensors, topo order              │
│    modality heads = subgraphs (codec / patch / vocab)          │
├────────────────────────────────────────────────────────────────┤
│ 2 THE TENSOR CATALOG (wubu_tensor_store + wubu_weight_t)        │
│    ANY container: GGUF / safetensors / .st — name→tensor,      │
│    one descriptor (data, type, n_elems), universal dequant     │
├────────────────────────────────────────────────────────────────┤
│ 1 THE OP LATTICE (wubu_kernel + quantized_matmul)              │
│    matmul / conv / norm / softmax / rope / codec ops           │
│    CPU scalar → SIMD → CUDA → Vulkan → … (runtime registry)    │
└────────────────────────────────────────────────────────────────┘
```

### Layer 1 — the op lattice (exists, extended)

`wubu_kernel_run(WUBU_KERN_*, …)` dispatch table + `quantized_matmul`
type-agnostic dispatcher + `gguf_dequantize` universal dequantizer.
**Extension**: add the op kinds the new modalities need, each as an
entry in the SAME registry — conv (codecs/UNets), conv-transpose
(decoders), time-embedding (diffusion), cross-attention (DiT), STFT/
waveform ops (audio), pooling (vision). Ops are just function-pointer
table entries with typed tensor in/out — no architecture coupling.

### Layer 2 — the tensor catalog (exists)

`wubu_tensor_store` opens any container zero-copy, name-addressable,
streaming export (AN07). `wubu_weight_t` is the single descriptor.
**Extension**: none structural — add container readers (e.g. raw .bin
state_dicts, ONNX) as catalog backends; the catalog IS the loader.

### Layer 3 — the graph IR (MISSING — the heart of the build)

A model becomes: **a list of typed tensor nodes** (each a
`wubu_weight_t` + name + role tag), **a list of op nodes** (op kind,
input tensor refs, output tensor refs, op params), **a topological
order**. Two sub-layers:

- **Binders** (importers): GGUF metadata + tensor names →
  graph; ONNX nodes → graph; HF config.json + safetensors →
  graph. This is where the old per-arch loaders (`wubu_model_adapter`)
  become *binders that emit a graph*, and new models (MusicGen, DiT,
  Sora-style) get NEW binders only — never new engine code.
- **Executors**: topological walk of op nodes → layer-1 dispatch.
  The existing `wubu_ssm_forward` / `wubu_gqa_forward` /
  `wubu_moe_forward` become *optimized subgraph executors* (fast paths
  the graph IR can pattern-match), not the only way to run.

**Why the graph fixes the class of bugs we just fixed**: the GQA/GGUF
`_q`/`_raw` triplication existed because structs were hand-wired to
names. In a graph IR, a weight is one descriptor, one type tag, one
consumer (`wubu_weight_to_f32` / `wubu_weight_matmul`). There is no
second representation to get out of sync — by construction.

### Layer 4 — delta-adapter fine-tuning (exists, generalized)

`wubu_lora` + `wubu_train` + the backward modules
(`wubu_*_backward.c`) + Muon/GRPO. **Generalization**: deltas are
`wubu_weight_t` too (or F32 mirrors); `wubu_model_apply_lora` becomes
`wubu_graph_apply_deltas(graph, deltas[])`; training regimes (SL,
GRPO, GAN min-max, diffusion denoising) are **regime descriptors** on
the same graph. "Train them a little to test them" = apply a delta to
any loaded model, run the diagnostic battery, keep or roll back.

### Layer 5 — the diagnostic immune system (exists, extended)

`wubu_diag` (AN08) + the amoeba (`wubu_amoeba`, WB05) + the 5+1
recovery + Triple-DA. **Extension**: audit hooks at the four seams —
**load** (shape/dtype/NaN/range per tensor vs catalog metadata —
already proven by test_universal_weight, which caught the Q4_0 nibble
bug), **save** (byte-identical round-trip check — AN07 proven),
**forward** (activation NaN/finite guards per node), **mutate**
(grow/shrink validated by fitness gate). The immune system treats a
corrupt tensor the way it treats a dead cell: quarantine, diagnose
root cause (the causal walker), roll back via the 5+1.

---

## 3. Mapping the world's models onto the substrate

| Model | Container | Graph | Fine-tune | Diagnosis |
|---|---|---|---|---|
| Any LLM (WuBu-35M, Qwen, KAT) | GGUF/safetensors | attention+FFN blocks (binder) | LoRA/delta `wired` | AN08 + load audit |
| MusicGen (music) | HF safetensors | EnCodec (conv) + AR transformer LM | delta on LM | load audit + activation guards |
| DiT / SD (image diffusion) | HF safetensors / ONNX | patch embed → DiT blocks → linear out; VAE codec subgraph | delta on DiT (ProLoRA pattern) | same |
| Sora-style (video) | HF safetensors | spacetime patch embed → DiT + temporal attn | delta | same |
| GANs (image/style) | safetensors/ONNX | generator graph + discriminator graph | min-max regime | two-body audit |
| Encoder-decoder (T5, Whisper) | safetensors | enc subgraph + dec subgraph | delta | same |

**Every row uses the same five layers.** The only per-model work is a
binder (map names→graph) — never the engine, never the container,
never the fine-tune path, never the immune system.

---

## 4. The amoeba principle applied to the manifold

- **Grow**: a new model family = a new binder + new op entries. The
  engine body doesn't change; the pseudopod extends.
- **Shrink**: unused op entries / binders are registry slots, pruned
  like dead cells.
- **Adapt**: dtype/backend auto-selection already exists
  (`wubu_weight_direct_ok`, kernel registry); a quantized tensor whose
  type the direct path can't handle falls back to dequant+SGEMM —
  the manifold never asserts on a file it doesn't know, it degrades
  and reports.
- **Diagnose**: Triple-DA at every seam; the load audit IS the first
  line (proven: it caught a real Q4_0 dequant bug the same day the
  layer landed).

---

## 5. Build order (wired-gate discipline, each step tested)

1. **`wubu_graph` core** (new): typed tensor node + op node +
   topo order + minimal executor. Test: build a 3-op graph
   (matmul→relu→matmul), run it, assert outputs vs hand-computed.
   *(next session)*
2. **GGUF binder**: load model-mixed.gguf → graph (all 137 tensors
   become nodes, verified by test_universal_weight).
3. **Optimized-subgraph fast path**: pattern-match
   attention+FFN blocks → route to existing `wubu_*_forward` — the
   graph IR becomes the loader, the old forwards become accelerators.
4. **ONNX binder** (graph import — the universal interchange).
5. **Modality binders**: first a small DiT-style graph (patch embed +
   N transformer blocks + out proj) and a codec-style conv graph —
   proves music/video/diffusion share the loader.
6. **Regime descriptors**: SL → GRPO → GAN min-max → diffusion
   denoising as graph-level training loops.
7. **Seam audits**: load/save/forward/mutate hooks in `wubu_diag`.

---

## 6. Sources (online research, 2026-08-06)

- GGUF spec: `ggml-org/ggml` docs/gguf.md (typed tensor container,
  ggml_type space incl. BF16=30, TQ, MXFP4, NVFP4)
- HF GGUF parsing, model-format comparisons (GGUF/safetensors/ONNX)
- ONNX IR + ONNX Runtime high-level design (graph nodes → in-memory
  IR → execution-provider dispatch)
- MLIR project docs + Wikipedia (dialects, progressive lowering,
  multiple abstraction levels in one IR)
- DiT — Peebles & Xie, 2212.09748 (latent patches → ViT backbone,
  adaLN-Zero); encord DiT guide; VDT/Sora coverage
- Sora — OpenAI "Video generation models as world simulators"
  (spacetime patches, diffusion transformer)
- AV-DiT — 2502.03897 (joint audio-video DiT)
- MusicGen — Meta 2306.05284 + AudioCraft docs (EnCodec RVQ tokens →
  single-stage AR transformer LM)
- LoRA — Hu et al. 2106.09685; PEFT docs ("LoRA works on any
  architecture"); ProLoRA 2506.04244 (LoRA transfer across diffusion)
- GAN structure — IBM think, Google ML GAN overview (generator +
  discriminator = two ordinary graphs)
- Self-healing ML pipelines — 6-layer control loop (detect → heal →
  validate), drift KS-tests/z-scores
- In-repo: research/INDEX.md AN07 (tensor catalog), AN08 (hive
  diagnostics), AN13-15 (driver space, hourglass), WB05 (amoeba);
  docs/wubu-amoeba-design.md; wubuwizard-inference skill
