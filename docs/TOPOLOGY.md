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
