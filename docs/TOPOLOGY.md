# WuBu Project Topology — the master map of BOTH repositories

> 2026-08-03. The user's directive: "start cohesively organizing our
> project between our two repositories." This is the authoritative
> map: what lives where, the layer boundaries, the data flow, and the
> canonical placement rules. It supersedes scattered READMEs when they
> conflict.

```
╔══════════════════════════════════════════════════════════════════╗
║                    THE WUBU UNIVERSE                              ║
║                                                                  ║
║   ┌──────────────────────────┐   ┌──────────────────────────┐    ║
║   │   wubuwizard (THE BRAIN) │   │    wubuos (THE BODY)     │    ║
║   │   inference + training   │   │   kernel + shell + GUI   │    ║
║   │   research + math vault  │   │   firmware + attestation │    ║
║   └──────────┬───────────────┘   └──────────┬───────────────┘    ║
║              │  trained weights,            │                    ║
║              │  model cards, evals          │  Live Colonel      ║
║              ▼                              ▼  (ring-0 REPL)     ║
║   WuBu-35M (HF) ◄──────────────►  WuBuOS metal (measured boot)   ║
║                                                                  ║
║   satellites: BearRL, WuBuContainer, multi-device-os,            ║
║   mythos-fable, reactos-study, gnome-study, mujoco_local,        ║
║   physics, bytropix-*, VulkanShaderCUDA                          ║
╚══════════════════════════════════════════════════════════════════╝
```

## THE ONE-SENTENCE SPLIT

**wubuwizard = the BRAIN** (everything that thinks: model code,
training, research, math, inference engines, the corpus pipeline).
**wubuos = the BODY** (everything that acts: the kernel, the shell,
the GUI, the firmware that boots it, the measured-boot chain, the
recovery substrate, the container isolation).

The Brain trains; the Body runs. The Live Colonel (ring-0 REPL in
wubuos) is where the Body hosts the Brain.

---

## 1. wubuwizard (THE BRAIN) — /home/wubu/wubuwizard

### 1.1 The layers

| Path | Role | Contents |
|---|---|---|
| `src/` + `include/` | the ENGINE | **277 C modules / 273 headers** — every algorithm, every kernel, every data structure. Opaque structs, minimal includes, pure C11. |
| `tools/*.c` | the CLI + tests | **516 C tools** — one test per module (`test_<module>.c`), operational CLIs (`wubu_cli`, `wubu_train_cli`, `gen_text*`, `infer_*`). |
| `tools/*.py` | the harnesses | **84 Python tools** — corpus fetch/extract (`wubu_*`), API clients (`nvidia_nim`, `openrouter_rlhf`), viz. |
| `research/` | the paper library | **40+ research notes** (`001-kv-entropy` …) — each with Triple-DA, implementation status (`wired`/gap), ties. |
| `THEORY/` | our own papers | the WuBu Nesting (層疊嵌套) papers, foundational philosophy, axiomatic emergent theory, `papers/` (DeepSeek lineage, Möbius transformers, …). |
| `MATH/` | the proof vault | `lean/wubu_proofs/` — the Lean-verified theorems (Poincaré ball, Möbius, gyration, MLA compression). |
| `WUBUNEST_V2/` | python training prototypes | the numpy/torch nesting experiments that became the C11 `wubu_nest`. |
| `docs/` | the brain's docs | model blueprint, model card, live-stream/free-API ledger, improvement plans. |
| `vault/` | collected references | api-server notes, quantization formats, bins/tools. |
| `manifests/` | model configs | `Qwen_Qwen3.6-27B`, `Kwaipilot_KAT*`, `InternScience*` — the bigger-brother line. |
| `models/` | local weights | `wubu/` (the WuBu seed: safetensors + tokenizer), the reference checkpoints. |
| `python/` | small helpers | tokenizer extraction etc. |
| `DEMOS/ DRAFT/ DIAGRAMS/` | sketches | the early prototyping (kept for lineage; most logic now lives in `src/`). |

### 1.2 The engine modules (the 277)

The `src/wubu_*.c` modules cluster by theme (the naming convention:
`wubu_<theme>_<thing>.c`):

| Cluster | Modules (representative) |
|---|---|
| **attention** | `wubu_attn_kernels`, `wubu_attn_gate`, `wubu_attn_tune`, `wubu_attnres`, `wubu_cross_attn` |
| **KV cache** (11+) | `wubu_kv_cache`, `wubu_kv_evict`, `wubu_kv_compress`, `wubu_kv_tier`, `wubu_kv_quant`, `wubu_paged_kv`, `wubu_mla`, `wubu_4kv`, `wubu_ring_attn` |
| **MoE** (6+) | `wubu_moe`, `wubu_moe2`, `wubu_moe_grouped`, `wubu_moe_hyperbolic`, `wubu_latentmoe`, `wubu_ssd_moe` |
| **SSM** (4+) | `wubu_ssm_scan`, `wubu_ssm_recurrence`, `wubu_nested_ssm`, `wubu_chunked_ssm` |
| **speculative** (4+) | `wubu_spec_decode`, `wubu_spec_tuner`, `wubu_spec_variants`, `wubu_medusa` |
| **quantization** | `quantized_matmul`, `quantized_dot_generic`, `wubu_awq`, `wubu_gptq`, `wubu_smoothquant`, `wubu_nf4`, `wubu_mxfp4`, `dequant_iq2_xxs` |
| **hyperbolic/nesting** | `wubu_hyper`, `wubu_nest`, `wubu_mobius_linear`, `wubu_poincare_gqa`, `wubu_hyperbolic_output_proj`, `rsgd` |
| **model core** | `wubu` (the seed), `wubu_train`, `wubu_backprop`, `wubu_model`, `wubu_gemma4`, `wubu_tokenizer_hf` |
| **the AGI organs** | `wubu_hive` (memory), `wubu_moe2` (agents), `wubu_prover2` (verifier), `wubu_agi` (the loop), `wubu_deltanet` (linear mixer) |
| **agentic OS** | `wubu_agentic_kv`, `wubu_agentic_mem`, `wubu_agentic_os`, `wubu_agentauth`, `wubu_agentid` |
| **misc** | `wubu_arena`, `wubu_audio`, `wubu_bandit`, `wubu_actor_critic`, `wubu_ecs`, `wubu_hopfield`, `wubu_energy`, `wubu_freeenergy`, `thread_pool`, `tile_manager` |

### 1.3 The AGI brain pipeline (the flow)

```
corpus (SD card: /home/wubu/sdcard/corpus/)
  ├─ text/        raw Cosmopedia shards (wubu_extract.py)
  ├─ tokens/      .tok uint16 streams (wubu_tokenc C11 BPE)
  ├─ finemath-live.tok / openmath-live.tok   (wubu_stream.py live)
  └─ checkpoints/ seed.st-NNN.st (every 10 steps, the 5+1 slots)

trainer (tools/wubu_train_cli.c + src/wubu_train.c
         + src/wubu_backprop.c)
  └─ WuBu-35M safetensors -> trained .st checkpoints -> HF
       (WaefreBeorn/WuBu-35M, weights + tokenizer + LICENSE + card)

oracles (tools/nvidia_nim.py, tools/openrouter_rlhf.py)
  └─ the RLHF reward: WuBu drafts -> frontier scores -> trainer
```

---

## 2. wubuos (THE BODY) — /home/wubu/wubuos

### 2.1 The layers

| Path | Role | Contents |
|---|---|---|
| `src/kernel/` | the KERNEL | **~90 modules**: boot/crt0, memory, tasking, interrupts (APIC/PIC/PIT), AHCI, FAT32 family (10 modules), TXFS, VMM, SMP, klog, libc, serial, swap, sync, WDT, TSS, vdso, the human HX family (`wubu_psych`, `wubu_tutor`, `wubu_bonzi_study`), the recovery (`wubu_recovery`), the AGI kernel (`wubu_agi_kernel`), the hive port (`wubu_hive`), the math (`wubu_math`). |
| `src/firmware/` | WuBuFW | the UEFI firmware from scratch (no EDK2): fw_* modules (PCI, NVMe, AHCI, XHCI, GOP, TPM, secureboot, sha256, acpi), `fw_agi` + attestation, the chainloader, `wubufw.fd` — **the measured boot chain (28/28 conformance, real kernel boots)**. |
| `src/apps/` | the GUI apps | canvas (full editor), explorer, notepad, calc, regedit, taskmgr, repl, the bonzi/comfy/cmd/control suites, the Tandem shared-desktop window. |
| `src/gui/` | the windowing | Win98/XP chrome, theme engine (`wubu_theme`), rendering. |
| `src/bridge/` | the VSL bridge | the ReactOS NT syscall -> VSL transliteration, the syscall handlers. |
| `src/compiler/` | the HolyC compiler | lexer, parser, codegen, PTX — "My Seed" (the compiler that compiles). |
| `src/runtime/ src/hosted/ src/shell/` | the hosted layer | the scaffold for Linux/Windows/macOS parity, the 9P namespace. |
| `src/worldsim/ src/bear/` | the RL world | cartpole physics, GAAD training, curriculum. |
| `docs/compendium/` | the institutional memory | 00-philosophy, 01-reference (GENERATED by make docs), 02-architecture, 03-learned (the prestige ledger: worked/didn't-work), 04-roadmap, 05-sources. |
| `holyc-include/` `vendor/` `reference/` | the reference | ZealOS headers, upstream comparisons. |

### 2.2 The boot chain (the verified spine)

```
WuBuFW (src/firmware) measures the kernel
  -> PCR4 + AuthentiCode (TPM)
  -> chainloader reads KERNEL.ELF off the ESP
  -> SHA-256 -> attestation handoff in low RAM
  -> ExitBootServices -> crt0 -> kernel_main
  -> AGI supervisor with the root-of-trust gate LIVE
     (verified: make test_agi_metal = PASS, measured boot green)
```

### 2.3 The kernel's AGI organs

| Module | Role |
|---|---|
| `wubu_recovery` | the 5+1 rollback (five slots + the Jesus state) — mistakes are safe |
| `wubu_psych` | the HX-A user model + HX-B adaptive timing |
| `wubu_tutor` | HX-C learning/education |
| `wubu_bonzi_study` | HX-D companion |
| `wubu_agi_kernel` | the AGI supervisor (ring-0, attestation-gated) |
| `wubu_hive` | the hive port (the AGI's memory, kernel-side) |
| `wubu_verifier` | the DA-2 fail-closed verification |
| `wubu_attest` | the root-of-trust attestation |
| `wubu_hid/input` | the human's mouse + keyboard (the human keeps control) |

---

## 3. THE BOUNDARIES (what goes where)

**The Brain owns (wubuwizard only):**
- ALL model code (forward/backward/training), ALL quantization, ALL
  inference engines, the tokenizer, the oracles (NVIDIA/OpenRouter),
  the corpus pipeline, the research, the math proofs.
- The hive lives HERE as the reference implementation (`src/wubu_hive.c`).

**The Body owns (wubuos only):**
- The kernel, the firmware, the boot chain, the GUI, the shell, the
  compiler, the container isolation, the recovery substrate.
- The hive lives HERE as the metal port (`src/kernel/wubu_hive.c`) —
  same API, no-heap (the kernel allocator), for the ring-0 brain.

**The bridge (both):**
- `WuBu-35M` weights flow Brain -> HF -> (Body hosts them on metal).
- The Live Colonel (Body, ring-0) loads the Brain's model file.
- The 9P namespace (`/n/kv/`, `/n/models/`) exposes the Brain's state
  to the Body's tools (per WUBUOS_INTEGRATION.md).
- `wubu_agi` (Brain: the learning loop) and `wubu_agi_kernel` (Body:
  the supervisor) are the two halves of the same AGI: the Brain
  learns, the Body protects and acts.

**Satellite repos (context, not core):**
- `BearRL` — RL training experiments (the cartpole GAAD work).
- `WuBuContainer` — container isolation prototypes (now in kernel).
- `multi-device-os`, `mythos-fable` — kernel lineage studies.
- `reactos-study`, `gnome-study` — upstream gap analyses.
- `physics`, `mujoco_local` — physics/RL grounding.
- `bytropix-*` — the bytropix integration work.
- `VulkanShaderCUDA` — the Vulkan compute path.

---

## 4. THE PLACEMENT RULES (canonical)

1. **A new algorithm goes in wubuwizard** `src/wubu_<theme>.c` +
   `include/wubu_<theme>.h` + `tools/test_<theme>.c`. No exceptions.
2. **A new kernel primitive goes in wubuos** `src/kernel/wubu_*.c`.
   If it must also run in the Brain, port it (same API, metal impl).
3. **Research notes** go in `wubuwizard/research/NNN-name.md` with the
   Triple-DA + `wired`/gap status. Papers go in `THEORY/papers/`.
   Proofs go in `MATH/lean/wubu_proofs/`.
4. **The prestige ledger** (worked/didn't-work) goes in
   `wubuos/docs/compendium/03-learned/`.
5. **Model artifacts** (weights, cards) go on HuggingFace under
   `WaefreBeorn/`; the local copies live in `wubuwizard/models/`.
6. **Corpus data**: ACTIVE working copies live on the SSD at
   `/home/wubu/models/corpus/` (master manifest `CORPUS.md` there:
   Tier 0 pretrain tokens, Tier 1 SFT pack, Tier 2 agentic pack).
   The SD card (`/home/wubu/sdcard/corpus/`) is the COLD raw archive;
   `/home/wubu/sdcard/archive/` holds finalized cold tarballs
   (research ponds, qwen36 embeddings). Never clone git or write
   active work on the SD card (drvfs has no chmod; 256KB clusters).
   Nothing corpus goes in a repo.
7. **Secrets** live in `~/.hermes/profiles/mind-palace/secrets/`
   (0600), NEVER in any repo.
8. **Test binaries** are never committed (gitignore covers `/test_*`).
9. **The research ponds** (701 MB pure text, 7 ponds × 100 MB) are the
   READING substrate — `/home/wubu/research-ponds-work/` (SSD active,
   SD `archive/` cold). PONDS.md is the catalog; grep the ponds for
   the failing subject, sources.json maps file → paper/repo.

## 5. THE AUDIT FINDINGS (2026-08-03, from the full-repo survey)

1. **No topology doc existed** — this file fixes that. The repo roots
   had grown organically; the boundaries were implicit.
2. **The Brain's training core had 3 real gaps** (found by reading
   `src/wubu_train.c`):
   - `layer_grad()` gave EVERY layer the same outer product — not
     backprop; the full backward pass is the `wubu_backprop`
     milestone in progress.
   - `muon_update()` was momentum SGD — no Newton-Schulz (the Muon
     paper's entire point).
   - no gradients flowed through the attention path.
3. **The Body is healthy**: 468+ C files / 91 test targets / measured
   boot verified / monoliths dissolved. The Brain's `test_*` binaries
   are gitignored correctly.
4. **The hive exists in BOTH repos** — intentional (reference vs metal
   port), now documented as the boundary contract.

## 6. NEXT ACTIONS (the cohesive path)

1. Finish `wubu_backprop` (real backward + real Muon) — the
   Brain's training gap.
2. Wire the RLHF oracle rewards (NVIDIA/OpenRouter) into the trainer —
   the Brain's RLHF loop.
3. Port the trained WuBu checkpoints to the Body (Live Colonel loads
   the safetensors via the 9P namespace) — the Brain→Body bridge.
4. Run `make docs` so `docs/compendium/01-reference` regenerates with
   the new modules.

<!-- repodoc:BEGIN -->
## Module map (auto-generated)

| `src/src/bench.c` | GPU Output Projection — hidden @ output_weight^T via cuBLAS |
| `src/src/dequant_iq2_xxs.c` | IQ2_XXS block-level operations for on-the-fly dequant dot product. |
| `src/src/gaad_nesting_llm.c` | static int64_t golden_split_pos(int64_t length) { |
| `src/src/gguf_reader.c` | From ggml-common.h — lookup table for 1.5625 bpw dequantization |
| `src/src/kv_paged_attention.c` | - Prefix caching for shared prompts |
| `src/src/qlearner.c` | Reward = 1/(loss + eps): lower loss = higher reward. |
| `src/src/quantized_dot_generic.c` | Self-contained generic + SIMD implementations of quantized dot products. |
| `src/src/quantized_matmul.c` | For each output column, quantizes the F32 input to Q8_K then calls |
| `src/src/quantized_matmul_fixed.c` | col_stride_bytes: byte stride between columns (0 = packed) |
| `src/src/rsgd.c` | the Poincaré ball. The key steps: |
| `src/src/safetensors_reader.c` | F32 / F16 / BF16 / I8..I64 tensors to float32. |
| `src/src/safetensors_writer.c` | [ uint64 LE header_len ][ header JSON ][ padding to 8 ][ raw blob ] |
| `src/src/thread_pool.c` | ── Thread pool using OpenMP ─────────────────────────────────── |
| `src/src/tile_manager.c` | - Tiles = 64×64 "pixel" blocks (64 tokens each) |
| `src/src/wubu.c` | (c) 2026 Harshal Singh). Pure C11, no third-party deps. The forward |
| `src/src/wubu_4kv.c` | 1. KV-cache is memory-bandwidth-bound in decode (Roofline 2607.02558). |
| `src/src/wubu_acq.c` | - EI(x) = (μ-f*)Φ((μ-f*)/σ) + σ·φ((μ-f*)/σ)  [closed form] |
| `src/src/wubu_active.c` | - FF05: uncertainty sampling = query argmax σ(x); QBC = query argmax |
| `src/src/wubu_actor_critic.c` | - GG03: critic learns V(s) via TD: δ = r + γV(s') - V(s). Actor updates |
| `src/src/wubu_affinity.c` | C11, self-contained (Linux). No god headers. |
| `src/src/wubu_agentauth.c` | Agents in a multi-agent system exchange messages; without authentication a |
| `src/src/wubu_agentic_kv.c` | LMCache-vision-hash / LOOK-M / agentic-compaction 7-hop): |
| `src/src/wubu_agentic_mem.c` | - AE01 episodic->semantic consolidation: an episodic event is "distillable" |
| `src/src/wubu_agentic_os.c` | - AD01 9P capability enforcement: each agent gets a bounded subtree of the |
| `src/src/wubu_agentid.c` | - DD03: each CoAgent gets a verifiable identity (ID + name + capability |
| `src/src/wubu_agi.c` | 1. observe: push an observation into the hive |
| `src/src/wubu_align.c` | if (x > 0) return -logf(1.0f / (1.0f + expf(-x))); |
| `src/src/wubu_ambig.c` | static const wubu_us_slot_t *find_slot(const wubu_us_slot_t *state, |
| `src/src/wubu_amoeba.c` | can use the hive." The amoeba's cells ARE hive slots: |
| `src/src/wubu_arena.c` | Self-contained C11. See header. |
| `src/src/wubu_attn_gate.c` | *dynamically* to the attention output. Suppresses attention sinks and |
| `src/src/wubu_attn_kernels.c` | - P11 int2 KV dequant: KV stored as 2-bit (4 levels) per component with a |
| `src/src/wubu_attn_tune.c` | - L06 Quest: sub-linear attention by selecting, per query block, the top-k |
| `src/src/wubu_attnres.c` | C11, self-contained. AttnRes lets a layer READ representations written by |
| `src/src/wubu_audio.c` | hz→mel, power spectrogram, log-scale): |
| `src/src/wubu_awq.c` | Compression and Acceleration", MLSys 2024. |
| `src/src/wubu_backprop.c` | WuBu seed (12 layers, 7 Q heads / 1 KV head GQA, 448-dim, 16384 |
| `src/src/wubu_bandit.c` | - FF06: each "config family" (attention variant, quant scheme) is an arm. |
| `src/src/wubu_bf16_gemv.c` | dispatch + F32 fallback. C11. No third-party deps; uses <immintrin.h> only |
| `src/src/wubu_bft.c` | Two-Fold BFT, n=3f+1 threshold): |
| `src/src/wubu_bi.c` | norm change at layer l). Low BI = redundant layer (ShortGPT removes |
| `src/src/wubu_bo.c` | - FF03: maintains a candidate set, scores each with the acquisition function |
| `src/src/wubu_bonzi.c` | int wubu_bonzi_mood_step(wubu_bonzi_mood_t *m, float t_val, float t_ar, |
| `src/src/wubu_bonzi2.c` | int wubu_bonzi_sentiment(const float *text_feat, const float *voice_feat, |
| `src/src/wubu_bridge.c` | int wubu_br_mood_retrieve(const float *mood_patterns, int n_moods, |
| `src/src/wubu_bridge2.c` | Agnostic: a bridge-table (the JE emotion event → external driver), |
| `src/src/wubu_cache_advice.c` | C11, self-contained. Upgrades the ds4-ssd LRU slot-bank with a learned |
| `src/src/wubu_capacity_wall.c` | binding constraint oscillates between weight-I/O (W) and KV-I/O (K) and |
| `src/src/wubu_capzero.c` | - AF02 deny-by-default tool registry: an agent holds an explicit capability |
| `src/src/wubu_causal.c` | temporal/belief, logic engines, PDDL planning, abductive/counter-abductive, |
| `src/src/wubu_cegis.c` | - EE03: ∃f.∀x,y. φ(f,x,y). Loop: synthesize candidate f from grammar |
| `src/src/wubu_chunked_prefill.c` | Unveiled" (arXiv:2607.02558); disaggregated PD papers. |
| `src/src/wubu_cla.c` | C11, self-contained. CLA reduces KV cache by sharing K/V tensors across |
| `src/src/wubu_codeexec.c` | - AX07: verify generated code before it enters the decode loop. |
| `src/src/wubu_codesynth.c` | - AX10: the agent receives a textual spec (operation + func name), |
| `src/src/wubu_compress.c` | int wubu_comp_llmlingua(const float *perplexities, int n, float th, |
| `src/src/wubu_compress2.c` | int wubu_comp2_llmlingua(const float *perplexities, int n, float th, |
| `src/src/wubu_contbatch.c` | - HH04: schedule at iteration (token) granularity, not request granularity. |
| `src/src/wubu_continuous_batching.c` | C11, self-contained. Implements continuous batching (vLLM-style): |
| `src/src/wubu_coord.c` | access-control, intent-lock-before-edit, conflict-resolution 7-hop): |
| `src/src/wubu_credit.c` | - AH12: given a frozen reference model's answer-predictability before/after |
| `src/src/wubu_credit_sft.c` | (Orchard): a trajectory that never resolved still contains productive |
| `src/src/wubu_cross_attn.c` | (text) and K/V come from an encoder (vision/audio). Uses the same |
| `src/src/wubu_ctx_manage.c` | (L16 elastic context / N07 tiered-cache advisor / N14 MoD router). C11. |
| `src/src/wubu_ctxvm.c` | - AF08 4-level context hierarchy: L1 gen window, L2 session, L3 long-term, |
| `src/src/wubu_cuda_graph.c` | (Area E, items E.41/E.42/E.43/E.50). C11 planning logic is testable on CPU; |
| `src/src/wubu_db_cross.c` | gaps each import a database concept into the KV/decode engine as a small, |
| `src/src/wubu_dbstate.c` | static const char *find(const wubu_db_slot_t *state, int nslots, |
| `src/src/wubu_dedup.c` | A polynomial rolling hash over the window; the hash table maps the |
| `src/src/wubu_delta_net.c` | C11, self-contained. Implements the DeltaNet fast-weight update: |
| `src/src/wubu_deltanet.c` | int wubu_deltanet_state_init(wubu_deltanet_state_t *st, int n_heads, |
| `src/src/wubu_der.c` | int wubu_der_push(wubu_der_buffer_t *b, const float *teacher_logits, int ndim) |
| `src/src/wubu_dgm.c` | - AX01: DGM empirical gate -- verified=1 only when gen_text returns 0 |
| `src/src/wubu_dims.c` | See wubu_dims.h. The loader sets WUBU_DIMS from real tensor shapes; |
| `src/src/wubu_dims_gpu_stub.c` | CPU-only stub for the GPU dims sync symbol so CPU builds/tests link |
| `src/src/wubu_distill.c` | - BB04: teacher snapshot + KL divergence soft-target loss. |
| `src/src/wubu_dn2.c` | - S02 Gated DeltaNet-2: decouples erase and write. Two gates e (erase) and |
| `src/src/wubu_dqn.c` | - GG05: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]. Off-policy TD(0). |
| `src/src/wubu_dsa.c` | indexer (DSA indexer). Self-contained C11 (libc + libm only). |
| `src/src/wubu_eagle.c` | Draft model = truncated target model (fewer layers). |
| `src/src/wubu_early_exit.c` | See header. Self-contained C11. |
| `src/src/wubu_ecs.c` | wubu_ecs_t *wubu_ecs_create(int cap) { |
| `src/src/wubu_energy.c` | energy ledger can later be fed by real RAPL/CMU counters). |
| `src/src/wubu_epcap.c` | int wubu_epcap(const int *cost, int n, int budget, int *out) |
| `src/src/wubu_equiv_check.c` | int wubu_equiv_vectors(const float *a, const float *b, int n, |
| `src/src/wubu_eval.c` | int wubu_eval_run(const wubu_db_goal_t *goals, const wubu_eval_traj_t *trajs, |
| `src/src/wubu_eval_qat.c` | QAT-STE / per-channel / noise-injection 7-hop): |
| `src/src/wubu_evict2026.c` | int wubu_ev_pool_obs(const float *attn, int n, int w, float *out) |
| `src/src/wubu_evict2026b.c` | float wubu_ev_norm(float raw, float lo, float hi) |
| `src/src/wubu_evict2026c.c` | int wubu_evictc_h2o(const float *attention, int n, float th, int *keep) |
| `src/src/wubu_evolve.c` | - AX06: propose→verify→commit→regress loop. |
| `src/src/wubu_ewc.c` | - BB02: Elastic Weight Consolidation on the 15-dim sweep space. |
| `src/src/wubu_experibuf.c` | - BB01: reservoir-sampled ring buffer of past sweep configurations. |
| `src/src/wubu_expert_allreduce.c` | void wubu_allreduce_sum(const float *const *partials, int nranks, int len, float *out) { |
| `src/src/wubu_expert_choice.c` | Mixture-of-Experts", Google, 2024; Switch Transformer top-1 routing; |
| `src/src/wubu_fast_attn.c` | At 512K context, the per-query-position malloc(n_q_heads * attend_len * 4) |
| `src/src/wubu_flash_prefill.c` | Attention with IO-Awareness", NeurIPS 2022. |
| `src/src/wubu_flashdecode.c` | Self-contained C11. See header. Default chunk gives ~8 parallel KV chunks. |
| `src/src/wubu_fmt.c` | static int json_ok(const char *out) |
| `src/src/wubu_fp8.c` | uint8_t wubu_fp8_e4m3_from_f32(float x) { |
| `src/src/wubu_fraud.c` | evidence submission, trust decay, dispute arbitration): |
| `src/src/wubu_freeenergy.c` | inference (Theme IN). C11, deterministic. |
| `src/src/wubu_fuzz.c` | int wubu_fuzz_mutate(const char *in, char *out, int cap, uint32_t seed) |
| `src/src/wubu_fuzz2.c` | float wubu_fz2_tradeoff(float robustness, float quality, float w) |
| `src/src/wubu_gamebud.c` | clock_gettime(CLOCK_MONOTONIC, &ts); |
| `src/src/wubu_gemm.c` | - A panel packed into contiguous row-major (improves streaming + FMA |
| `src/src/wubu_gemma4_model.c` | Architecture: 48 layers, 40 sliding-window (HEAD_DIM=256) + 8 full-attention (HEAD_DIM=512). |
| `src/src/wubu_gemv_tune.c` | Pure C, routes through wubu_roofline for the B*-ridge decision. |
| `src/src/wubu_generate.c` | (doc 018 / K01). Self-contained C11. See header. |
| `src/src/wubu_gp.c` | - FF01: RBF kernel k(x,x') = σ²_f exp(-||x-x'||²/(2ℓ²)) + noise·δ. |
| `src/src/wubu_gptq.c` | Generative Pre-trained Transformers", ICLR 2023. |
| `src/src/wubu_grow.c` | the per-block weight byte size (all the block buffers) */ |
| `src/src/wubu_hadamard.c` | return n > 0 && (n & (n - 1)) == 0; |
| `src/src/wubu_hashrouter.c` | token. Slot k hashes (token_id, pos, salt_k, seed) with our own |
| `src/src/wubu_hive.c` | "live" when skip[s] == 0. Erase: skip[s] = 1, live--, and push the |
| `src/src/wubu_hopfield.c` | C11, deterministic, no third-party deps. |
| `src/src/wubu_hopfield2.c` | int wubu_hf_rk4_step(const float *state, const float *field, int dim, |
| `src/src/wubu_hopfield3.c` | static float dot(const float *a, const float *b, int d) |
| `src/src/wubu_hopfield4.c` | Implements the 26 remaining IP gaps (IP05-IP67 minus those already in |
| `src/src/wubu_hugepage.c` | bandwidth-bound and TLB-footprint-heavy; 2MB hugepages cut TLB misses and |
| `src/src/wubu_hwcaps.c` | See header. Self-contained C11. Raw CPUID, no third-party deps. |
| `src/src/wubu_hybrid.c` | int wubu_hyb_falcon(const float *attn_out, const float *ssm_out, |
| `src/src/wubu_hyper.c` | mobius_add_1d c x y = ((1 + 2cx·y + c·y²)·x + (1 - c·x²)·y) |
| `src/src/wubu_hyperbolic_output_proj.c` | exp_map(v): output[i] = tanh(||v||/R) * R/||v|| * v[i] |
| `src/src/wubu_imgenc.c` | static float lcg_randf(unsigned *seed) { |
| `src/src/wubu_integrate.c` | modules into the live decode path (option c: exploit discovered gaps). |
| `src/src/wubu_invariant.c` | - EE05: given a trace of loop states (var1, var2) at each iteration, discover |
| `src/src/wubu_kda.c` | C11, self-contained. KDA = DeltaNet with CHANNEL-WISE decay: each key channel |
| `src/src/wubu_kereq.c` | C11, self-contained. Genuine (if lightweight) SYMBOLIC prover: represents each |
| `src/src/wubu_kernel.c` | Adopted the kernel dispatch table pattern from waste_kernels[] |
| `src/src/wubu_kernel_backends.c` | register at runtime via wubu_kernel_register(). The engine never |
| `src/src/wubu_kv2026.c` | - Q02 ChunkKV: group consecutive KV tokens into semantic chunks, score each |
| `src/src/wubu_kv2026b.c` | - Q01 CentroidKV: cluster KV tokens by cosine similarity to a learned (here: |
| `src/src/wubu_kv2026c.c` | - Q11 DASH-KV: hash-based token-level attention scheduling. We compute a |
| `src/src/wubu_kv_adaptive.c` | LLMs via Entropy-Aware Cache Compression", ISCA 2025. |
| `src/src/wubu_kv_budget.c` | + footprint forecaster (L18 / L19 / N03 / N17). |
| `src/src/wubu_kv_cacheline.c` | starts on a cache-line boundary. This eliminates partial cache-line |
| `src/src/wubu_kv_compress.c` | slots carry little attention; retaining the *attention-mass-weighted* subset |
| `src/src/wubu_kv_evict.c` | See header for the policy. Self-contained C11. |
| `src/src/wubu_kv_runtime.c` | global g_kv_scheme instead of a compile-time #if, so the engine can pick the |
| `src/src/wubu_kv_select.c` | Pure C, routes through the tested wubu_roofline module. |
| `src/src/wubu_kv_shield.c` | cache accessed by untrusted indices (e.g. attacker-controlled attention spans, |
| `src/src/wubu_kv_styx.c` | KV-cache allocator (`wubu_kv_runtime.c`) and WuBuOS's 9P namespace. |
| `src/src/wubu_kv_tier.c` | HOT  = existing gqa_k_cache / gqa_v_cache (CPU RAM, current tokens) |
| `src/src/wubu_kv_transfer.c` | for a completed prefix to a transfer buffer (mmap'd temp file); a decode |
| `src/src/wubu_kvcache_quant.c` | KV-cache movement dominates bytes moved per token. |
| `src/src/wubu_kvquant.c` | C11, self-contained. FP8 (e4m3) and INT4-with-rotation KV storage. |
| `src/src/wubu_kvvq.c` | Self-contained C11. See header. |
| `src/src/wubu_latency.c` | - AF05 latency class (HRT/SRT/DT) + EDF/RM-ready scheduler hook: earliest- |
| `src/src/wubu_latentmoe.c` | C11, self-contained. 896 routed experts, top-k=16 active per token, PLUS a |
| `src/src/wubu_layer_skip.c` | y = x + gate * F(x)   where gate ∈ [0,1] |
| `src/src/wubu_linattn.c` | static float dot(const float *a, const float *b, int d) |
| `src/src/wubu_linattn2.c` | int wubu_la2_delta_write(float *state, int d, const float *k, const float *v, |
| `src/src/wubu_linear_attn.c` | These replace the O(n^2) attention with an O(n) recurrent state update. The |
| `src/src/wubu_lm_infinite.c` | - L13 LM-Infinite: landmark ("soft prompt") tokens are injected every `stride` |
| `src/src/wubu_lmcache.c` | latency via prefix offload + prefill/decode disaggregation. |
| `src/src/wubu_lookahead.c` | draft model, scan recent token history for a repeated n-gram and propose the |
| `src/src/wubu_loopguard.c` | LLM06/ASI02 tool-abuse cap, ASI08/strata JIT+HITL 7-hop): |
| `src/src/wubu_lora.c` | B^T @ A has shape [out_f, in_f] (matches W). Applied in place. |
| `src/src/wubu_lruk.c` | KV cache is a buffer pool; the right eviction policy is LRU-k (keep the k most |
| `src/src/wubu_masked_ce.c` | int wubu_masked_ce(const float *logits, const uint16_t *tokens, |
| `src/src/wubu_medusa.c` | - HH05: attach lightweight draft heads to the target's last layer → propose |
| `src/src/wubu_mega.c` | C11, self-contained. MEGA = single-head gated attention + multi-headed EMA |
| `src/src/wubu_mem_budget.c` | the safe KV cache size and forward buffer budget, never OOMs. |
| `src/src/wubu_metacog.c` | int wubu_mc_init(wubu_metacog_t *m, int n_agents) |
| `src/src/wubu_metagame.c` | fitness, faked-log lesson, self-improvement delta 7-hop): |
| `src/src/wubu_metagame2.c` | int wubu_meta_regulate(const float *policy_conf, int n, float th, int *action) |
| `src/src/wubu_mhc.c` | C11, self-contained. mHC widens the residual stream by factor `exp` and mixes |
| `src/src/wubu_mhc_mh.c` | manifold-constrained (row-softmax) mixing matrix, gated writes, and an |
| `src/src/wubu_misc_gaps.c` | P12/P13). C11, no third-party deps. |
| `src/src/wubu_mix.c` | long wubu_mix_build(const char **paths, const float *weights, int n, |
| `src/src/wubu_mla.c` | Mixture-of-Experts Language Model", arXiv:2405.04434. |
| `src/src/wubu_mm_adapter.c` | - CC04/CC06: projects vision/audio embeddings into text space (via |
| `src/src/wubu_mm_align.c` | - CC03: learned linear projection maps vision/audio features into the |
| `src/src/wubu_mm_kv.c` | - CC05: assembles the multimodal token prefix (vision + audio token IDs) |
| `src/src/wubu_mobius.c` | void wubu_mobius_add(const float *x, const float *y, int d, float R, float *z) { |
| `src/src/wubu_mobius_gyrate.c` | Optimized Möbius gyration using precomputed dot products. |
| `src/src/wubu_mobius_linear.c` | Helper: exp_map backward (matching interface from PGA backend) |
| `src/src/wubu_mobius_new.c` | τ = 1 + 2c⟨x,y⟩ + c²||x||²||y||² |
| `src/src/wubu_model.c` | Global tensor naming convention (set during model init) |
| `src/src/wubu_model_adapter.c` | self-contained, opaque). Hand-parses the JSON we care about: |
| `src/src/wubu_model_safetensors_bridge.c` | into wubuwizard's wubu_model_t and run them through the EXISTING |
| `src/src/wubu_moe.c` | GPU MoE expert forward (declared in wubu_model_gpu.cu, C linkage) |
| `src/src/wubu_moe2.c` | int wubu_moe2_route(const wubu_moe2_t *moe, const float *x, |
| `src/src/wubu_moe_backward.c` | Handles NULL expert weight pointers gracefully (skips that section). |
| `src/src/wubu_moe_grouped.c` | (Area D, items D.31/D.37/D.38). C11, self-contained. |
| `src/src/wubu_moe_hyperbolic.c` | Helper: map Euclidean vector to Poincaré ball via exp_map |
| `src/src/wubu_moe_hyperbolic_backward.c` | Poincaré router backward pass (single-level + nested 2-level). |
| `src/src/wubu_moe_rag.c` | KV-Packet / RACC / CAG / cross-doc-isolation 7-hop): |
| `src/src/wubu_moeroute.c` | - HH03: top-k routing with capacity factor C (each expert ≤ C tokens). |
| `src/src/wubu_moondream.c` | Self-contained C11 implementation of the MoonDream 3 bridge. |
| `src/src/wubu_more_spec.c` | (M07/M08/M09/M10/M15/M17/M18/M19/M20). C11. |
| `src/src/wubu_mxfp4.c` | C11, self-contained. MXFP4: 32-element blocks, each element E2M1 (1s/2e/1m), |
| `src/src/wubu_nest.c` | wubu_quat_t wubu_quat_mul(wubu_quat_t a, wubu_quat_t b) |
| `src/src/wubu_nested_ssm.c` | Nested SSM Forward Implementation |
| `src/src/wubu_nested_ssm_backward.c` | Nested SSM Forward-Save + Backward (BPTT through K Poincaré balls) |
| `src/src/wubu_neurom.c` | int wubu_neurom_encode(float value, float rate_max, float dt, int n_bins, |
| `src/src/wubu_nf4.c` | Quantization: normalize to [-1,1] via block absmax, then nearest-level |
| `src/src/wubu_ngram.c` | Pure C11, self-contained, zero external model weights. |
| `src/src/wubu_ngram_cascade.c` | Pure C11, self-contained. Uses prompt n-gram statistics to draft tokens. |
| `src/src/wubu_numerical_audit.c` | 1. No NaN / Inf in output (unless input has NaN/Inf) |
| `src/src/wubu_nvfp4.c` | uint8_t wubu_nvfp4_from_f32(float x) { |
| `src/src/wubu_paged_kv.c` | C11, self-contained. Implements vLLM-style paged attention bookkeeping: |
| `src/src/wubu_pagedkv.c` | - HH02: split KV into fixed-size blocks (16 tokens); logical block table → |
| `src/src/wubu_parallel_spec.c` | - V01 EAGLE-3 feature drafting: instead of drafting tokens, predict the next |
| `src/src/wubu_passk.c` | log C(n, k) via the log-gamma -- the counts are huge, the ratio is not */ |
| `src/src/wubu_pd_serve.c` | dynamic compute / mixture-of-depths (AC01-AC03). C11. |
| `src/src/wubu_pd_split.c` | C11, self-contained. Splits inference into a compute-bound prefill pool and a |
| `src/src/wubu_pim.c` | int wubu_pim_offload(int op_kind, long bytes, long compute_flops, |
| `src/src/wubu_pim2.c` | int wubu_pim2_bits(float sensitivity, float th_lo, float th_hi) |
| `src/src/wubu_planediv.c` | - AG02 control/data-plane separation: every input is tagged control-plane |
| `src/src/wubu_plateau.c` | float wubu_plateau_slope(const float *losses, int n, int window) |
| `src/src/wubu_poincare_gqa.c` | Dequant a [rows, cols] BF16/F16 matrix into F32 [cols, rows] (transposed), |
| `src/src/wubu_poincare_gqa_backward.c` | Helper: forward declarations for static helpers |
| `src/src/wubu_poincare_ssm_backward.c` | Poincaré SSM Backward (gyration chain rule) |
| `src/src/wubu_polar_pso.c` | serial bit reading for PolarQuant KV cache. |
| `src/src/wubu_polarquant.c` | paper (arXiv:2502.02617). Pairs of coordinates are transformed to |
| `src/src/wubu_policy.c` | - GG02: linear softmax policy π(a|s) = softmax(W·s + b). Baseline b(s) |
| `src/src/wubu_ppo.c` | - GG04: ratio r = π_θ(a|s)/π_θ_old(a|s). L = min(r·A, clip(r,1-ε,1+ε)·A). |
| `src/src/wubu_pref.c` | static float lg(float x) { return logf(1.0f + expf(-x)); } |
| `src/src/wubu_pref2.c` | static float lg(float x) { return logf(1.0f + expf(-x)); } |
| `src/src/wubu_prefix_cache.c` | Pure C11, self-contained. Uses FNV-1a 64-bit hash (no OpenSSL dep). |
| `src/src/wubu_priority.c` | the shame list (rolled-back events) prevents repeating failures, the |
| `src/src/wubu_prover.c` | - EE04: a lightweight propositional + arithmetic prover. Given premises and |
| `src/src/wubu_prover2.c` | checking in C11. The model proposes steps; the verifier accepts or |
| `src/src/wubu_q4k_m.c` | C11, self-contained. Matches GGUF Q4_K layout exactly. |
| `src/src/wubu_q8.c` | C11, self-contained. Q8_0 is effectively lossless (~0.5% vs FP16) at half |
| `src/src/wubu_quant_selector.c` | (N04 batch-size-aware, N05 context-length precision ladder, N09 PMC roofline |
| `src/src/wubu_quantkv.c` | - HH06: KV cache is memory-bound at 512K ctx. INT8 per-group (symmetric) |
| `src/src/wubu_rambus.c` | Interleaved banks + row-buffer banking + RDRAM-cycle cost model. C11. |
| `src/src/wubu_recency.c` | float wubu_recency_weight(long i, long n, float base, float power) |
| `src/src/wubu_reinforce.c` | - GG01: ∇J(θ) = E[Σ_t ∇log π(a_t|s_t) · (G_t - b)]. Monte-Carlo returns |
| `src/src/wubu_repetition.c` | as a ring buffer. repeat_penalty scans the recent window; DRY hashes |
| `src/src/wubu_resource.c` | degradation 70B->14B->7B 7-hop): |
| `src/src/wubu_reverify.c` | int wubu_reverify_init(wubu_reverify_t *r, double shift_thresh, |
| `src/src/wubu_ring_attn.c` | over 1M+ token contexts using the ring communication pattern. |
| `src/src/wubu_rollout.c` | int wubu_rollout_alloc(const float *succ, int n, int budget, |
| `src/src/wubu_roofline.c` | C11, self-contained. Implements the data-movement framework from the I/O |
| `src/src/wubu_rope_prefetch.c` | position encoding means K vectors at nearby positions have similar |
| `src/src/wubu_rotate.c` | Self-contained C11. See header for the invariance proof. |
| `src/src/wubu_rsi.c` | int wubu_rsi_gate(float verifier_score, float th, int *consecutive_fails) |
| `src/src/wubu_safekern.c` | - AF11 non-tamperable interrupt: a stop signal that lives OUTSIDE the agent's |
| `src/src/wubu_safetensors_model.c` | wubuwizard forward pass consumes. Dequantizes F16/BF16/F32 on the fly |
| `src/src/wubu_safetensors_shard.c` | See wubu_safetensors_shard.h. Self-contained; uses safetensors_reader. |
| `src/src/wubu_sandbox_safekern.c` | - AX08: bridge between sandbox isolation and safekern capabilities. |
| `src/src/wubu_save.c` | every trained checkpoint was a private .st dump no standard tooling |
| `src/src/wubu_scheduler.c` | batching + iteration-level KV-cache merge. Model-agnostic: operates |
| `src/src/wubu_seed.c` | static uint64_t splitmix64(uint64_t *x) |
| `src/src/wubu_self_cascade.c` | Pure C11. Calls a provided small-model forward function. |
| `src/src/wubu_semcons.c` | distributed semantic agreement, smart contract signalling): |
| `src/src/wubu_serve.c` | int wubu_serve_admit(long used_tokens, long budget, long req_tokens) |
| `src/src/wubu_serve2.c` | float wubu_serve2_fairness(long achieved, long entitled) |
| `src/src/wubu_si.c` | int wubu_si_init(wubu_si_t *s, const double *params, int ndim, double lambda) |
| `src/src/wubu_sindy.c` | - EE02: from trajectory (x_t, dx/dt) builds a candidate library (const, |
| `src/src/wubu_smoothquant.c` | Self-contained C11. See header. |
| `src/src/wubu_smt_check.c` | "Equivalence Checking of ML GPU Kernels". |
| `src/src/wubu_soa.c` | Arrays) for cache-friendly channel-wise access. In AoS, token i's hidden |
| `src/src/wubu_sparse_attn.c` | (L11 NSA / L12 MoBA). Self-contained C11. |
| `src/src/wubu_spawn.c` | C11, self-contained (no god headers). |
| `src/src/wubu_spec_cascade.c` | Pure C11, self-contained. Two cascade flavors: |
| `src/src/wubu_spec_decode.c` | proposal + target model verification via rejection sampling. |
| `src/src/wubu_spec_tuner.c` | should track the *measured* acceptance rate. If acceptance is high, raise K; |
| `src/src/wubu_spec_variants.c` | remaining M-family gaps are *combinations* of machinery already wired this |
| `src/src/wubu_specdec.c` | - HH01: draft model proposes K tokens; target verifies all in ONE forward |
| `src/src/wubu_ssd_moe.c` | See include/wubu_ssd_moe.h. Self-contained; C11; opaque ctx. |
| `src/src/wubu_ssm.c` | Global tensor naming convention (defined here for CORE_OBJ visibility) |
| `src/src/wubu_ssm_chunked.c` | written in matrix (outer-product / rank-1) form: |
| `src/src/wubu_ssm_scan.c` | C11, self-contained. Parallel (Blelloch) prefix scan over chunked SSM |
| `src/src/wubu_ssm_workspace.c` | static wubu_ssm_workspace_t g_pool[WUBU_SSM_WORKSPACE_MAX_LAYERS]; |
| `src/src/wubu_stream_kv.c` | 2026): at long context decode is KV-bandwidth/capacity bound. StreamingLLM |
| `src/src/wubu_symbolic.c` | - AW07: a Prolog-ish engine -- facts (predicate(args)) + rules |
| `src/src/wubu_symreg.c` | - EE01: discovers closed-form equations from (x, y) data. We implement a |
| `src/src/wubu_synth.c` | - AX05: spec→C11 code generation with compile-time verification. |
| `src/src/wubu_sys2026.c` | C11. Policy cores (hardware plumbing abstracted; the decision logic is real). |
| `src/src/wubu_sys_tune.c` | - L10 SeerAttention: per-head dynamic sparse attention -- predict each head's |
| `src/src/wubu_tandem.c` | Two stages (A=prefill/RSP, B=decode/RDP) run in tandem over a ring handoff. |
| `src/src/wubu_taskbd.c` | - BB03: detect task boundaries via performance divergence. When the |
| `src/src/wubu_tensor_store.c` | model file never loads weights -- it builds a name->(offset,dtype,shape) |
| `src/src/wubu_ternary.c` | int wubu_ternary_qat(const float *w, int n, float alpha, int8_t *out) |
| `src/src/wubu_thread_spec.c` | Two pinned thread pools (prefill / decode). See header. Self-contained C11. |
| `src/src/wubu_threshsig.c` | - DD02: simplified threshold signature scheme. Each agent produces a |
| `src/src/wubu_token.c` | int wubu_tok_bit_bpe_cost(int byte_len, int bits_per_symbol) |
| `src/src/wubu_token2.c` | float wubu_tok2_bench(long tokens, long chars) |
| `src/src/wubu_tokenizer.c` | Qwen3.6 exact byte-to-token mapping from original tokenizer.json |
| `src/src/wubu_tokenizer_hf.c` | Self-contained: embeds a tiny, correct recursive-descent JSON scanner |
| `src/src/wubu_tooluse.c` | - AX04: tool schema registry -- name+description+JSON Schema input, |
| `src/src/wubu_train.c` | grows here: the REAL backprop (wubu_backprop) + the REAL Muon |
| `src/src/wubu_traj_grpo.c` | recipe core). Group-relative advantage over the G trajectories: |
| `src/src/wubu_traj_sft.c` | The input is COPIED (never modified in place -- the in-place NUL |
| `src/src/wubu_tst.c` | TST: Token Superposition Training Implementation |
| `src/src/wubu_ttc.c` | - Q08 PolyKV: a shared, asymmetrically-compressed KV pool across agents. |
| `src/src/wubu_turboquant.c` | frame-based planning, and LRU eviction for the TurboQuant+/RotorQuant |
| `src/src/wubu_ubus.c` | Backends: CPU scalar (always), CPU OpenMP (12 threads), GPU cuBLAS |
| `src/src/wubu_uq.c` | - FF04: bootstrap ensemble over sweep replays → variance σ_uc² = 1/(B-1)Σ(f_b-μ)². |
| `src/src/wubu_user_sim.c` | static const char *find_slot(const wubu_us_user_t *u, |
| `src/src/wubu_uuid.c` | for 74 bits of randomness (only used once at startup — subsequent UUIDs |
| `src/src/wubu_value.c` | - GG06: Bellman optimality: V*(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V*(s')]. |
| `src/src/wubu_vecsearch.c` | PQ/RaBitQ/SQ quantization, FlashAttention, similarity metrics, |
| `src/src/wubu_verify.c` | - AX09: a lightweight formal gate — assertion-based invariant checking |
| `src/src/wubu_vision.c` | int wubu_vision_selector(const float *scores, int n, float th, int *keep) |
| `src/src/wubu_vision_moondream.c` | patch_embed → 27× ViT block → post_ln → proj_mlp → exp_map → Poincaré |
| `src/src/wubu_width.c` | old block in its top-left corner EXACTLY (no scaling) and zeroes the |
| `src/src/wubu_wm_kv.c` | (N02) + per-layer compute budget floor (N08). |
| `src/src/wubu_worldmodel.c` | 7-hop): pure LLM reasoning fails at agency because it is OPEN-LOOP -- it |
| `src/src/wubu_yarn.c` | C11, self-contained. Extends a model's trained context to longer lengths by |
<!-- repodoc:END -->
