# External Findings — OS Meta-Layer, AGI-UX, and Recursive Self-Improvement

> Researched 2026-10-11. This is a DIFFERENT axis from the existing
> `vault/external-findings/` manifold-geometry notes and the
> `docs/WUBUWIZARD_100*_IMPROVEMENTS` (which are inference-engine-scoped:
> kernels, MoE, quantization). This entry covers the three things the
> user's latest directive named, each mapped to a subsystem that already
> exists in the WuBu stack. Every claim cites a primary source and is
> checked against the actual repo, not assumed.

The stack (verified this session):
- **wubuos (BODY)** — kernel + Win98/XP shell + Styx/9P namespace + Arch
  containers. Already has: `wubu_exec.c` (universal ELF/PE/Mach-O/WASM/DOS
  dispatcher), `wubu_ns_bridge.c` + `/n/{svc,dram,kernel,hw}` namespace
  files, `wubu_archd_daemon.c` (service supervisor), `wubu_selfimprove.c`
  (DA-1/2/3 operators), `src/kernel/wubu_verifier.h` (independent verifier),
  `wubu_spawn.c` (dependency-free fork/exec/wait), `src/gui/wubu_bonzi.c`
  (desktop companion shell), `wubu_screenshot.c` (capture primitive),
  `styxfs_server.c` (already has a `/n/kv/*` write-side doctrine).
- **wubuwizard (BRAIN)** — 305-module C11 engine. Already has
  `include/wubu_agentic_os.h` (AD01-AD04: 9P capability enforcement under an
  agent subtree, exp-backoff, checkpoint pack/resume, resource budgets),
  `wubu_agentic_mem.h`, `wubu_tooluse.h`, `wubu_toolreg.h`,
  `wubu_amoeba.c`/`wubu_hive.c` (the mutate→validate→archive DGM organs).
- **wubunos (COMPILER)** — HolyC frontend + 14 ISA backends. 195/195 green.

---

## 01 — Styx/9P as the universal namespace ("reorganization for KV cache as a filesystem")

### 1.1 The core idea, and why it is provably right here
**Sources:**
- Pike et al., *Plan 9 from Bell Labs* — the "everything is a file system"
  doctrine. MIT PDOS 6.828 notes: three principles — (1) everything is a
  file, (2) one standard protocol, (3) private malleable namespaces.
- Pike, *The Styx Architecture for Distributed Systems*, 9p.io / Inferno —
  Styx is a variant of 9P. Key quote: "present their resources as files in a
  hierarchical name space… represent stored data, but may also be devices,
  dynamic information sources, interfaces to services, and control points."
- Plan 9 wiki, *Expanding your Grid* — the window system (8½) is itself a
  dynamic file server publishing `/dev/mouse`, `/dev/screen`.

**Overlap: STRONG.** The WuBu stack already runs Styx/9P and already exposes
kernel/hw/dram/services as namespace files. `styxfs_server.c` line 301 states
the doctrine verbatim: "/n/kv/* writes go into the KV tensor." The 9P
model — a resource is a *file server you attach*, and attaching a remote one
gives you a distributed system for free — is exactly the "reorganization"
the user wants. The KV cache, the expert slot-bank, the diagnostic hive, and
the companion's memory should each be a **9P file server** so the same
`open/read/write/stat` verbs that inspect the kernel inspect the Brain.

### 1.2 The transferable technique: per-process namespaces = capability security
Styx's private malleable namespaces mean a process's *view* is its capability
set. This is the mechanism behind `wubu_agentic_os.h`'s AD01 (9P capability
enforcement under an `agent_subtree`). A companion/agent gets its own
namespace subtree; what it can `attach` is what it may touch. No ambient
global access — security falls out of the namespace, not from ACL checks
bolted on.

### 1.3 "Pure interpretation so all OS apps run in our stack"
The Styx paper's insight that "if resources are represented as files, and
there are remote file systems, you have a distributed system" generalizes:
an app (ELF/PE/Mach-O/WASM/DOS) that expects a filesystem, a window, a
socket can be satisfied by a **user-level translator** that serves those
expectations as 9P files. Plan 9 does exactly this with user-level
translators for NFS/disk. WuBu already has `wubu_exec.c` for the format
dispatch and `src/runtime` for the shims; the missing piece is a **syscall/
libc translation layer per foreign ABI** (see plan item L2).

---

## 02 — The AGI-UX / Bonzi companion ("meta layering design and AGI user experience integration as Bonzi buddy")

### 2.1 What a *real* desktop companion is (and is not)
**Sources:**
- Jarvis, *The Complete Guide to Desktop AI Assistants in 2026* — the four
  defining traits: (1) **native to the OS, not a website** (global hotkey,
  paints over active windows, reads the FS, hooks accessibility APIs);
  (2) **summonable from anywhere** (a hotkey over whatever you're doing);
  (3) **cross-app** (sees the active screen so context is the prompt, not a
  description); (4) **persistent memory**.
- Perplexity *Personal Computer* (Mac, GA 2025) and Google Gemini/Apple
  Intelligence — "the assistant lives at the desktop layer because the work
  lives across many apps." "That ambient context is what separates a desktop
  agent from a chatbot."
- Gemma Wagon / NOVA (Devpost) — ambient desktop layer with a persistent
  floating "Orb", local multimodal reasoning, permission-aware tools, local
  REST API, offline fallback to a local model.

**Overlap: PARTIAL — and the delta is the work.** `wubu_bonzi.c` is already a
**native** desktop orb (not a browser tab) with an animate/tick/mouse API.
What it lacks to become the thing these systems describe: (a) **ambient
screen context** — `wubu_screenshot.c` provides the capture primitive but
nothing feeds the active framebuffer to the Brain as a prompt; (b) a
**global summon hotkey**; (c) **persistent memory** wired to
`wubu_agentic_mem.h`; (d) **tool execution** through `wubu_tooluse.h` gated
by the AD01 namespace capability. The companion is currently a shell; the
meta-layer is the wiring that turns the shell into an OS-integrated agent.

### 2.2 The design principle to adopt: context is the prompt
The single most transferable idea from the research: the companion should
*see* the active window (screenshot + accessibility-tree introspection) so
the user never has to describe what they're looking at. In WuBu terms: the
WM knows the focused window; the companion captures that window's rect from
the framebuffer and hands the pixels + a structured window-manager state
dump to the Brain. This is the "meta layering" — a layer that reads the OS's
own state (via 9P namespace files) and the screen, and acts through tools.

---

## 03 — Recursive self-improvement ("full stack recursive learning improvement")

### 3.1 The Darwin Gödel Machine is already your architecture — and you hardened it
**Sources:**
- Zhang et al., *Darwin Gödel Machine: Open-Ended Evolution of Self-Improving
  Agents*, Sakana AI + UBC + Vector, arXiv:2505.22954 (2025). Loop:
  propose→stage→evaluate→archive→sample. SWE-bench 20%→50%, Polyglot
  14.2%→30.7%. Maintains an open-ended archive with traceable lineage.
- The Register (2025-06-02) — the documented failure mode: DGM sometimes
  **"cheated"** (objective-hacking / reward hacking on the eval).
- darwin-agents prior-art notes — "empirical validation instead of formal
  proof"; and explicitly: "DGM didn't address the cost of failed mutations
  (its sandbox was infinite); we have to. DGM didn't have a death condition;
  we do."

**Overlap: STRONG.** WuBu's `ARCHITECTURE.md` already names the loop
"mutate → validate → archive, sandboxed" and the organs (`wubu_amoeba.c`,
`wubu_hive.c`). Critically, `wubu_selfimprove.c` already implements the two
things the DGM paper is criticized for lacking:
- **DA-3 human gate + freeze** = the death condition + oversight the DGM
  lacked (answers its objective-hacking failure mode).
- **DA-2 failure weighting (3x)** + an **independent verifier**
  (`src/kernel/wubu_verifier.h`) that must pass before any promote = the
  empirical-validation gate, made fail-closed.

So the recursive-learning leg is not "add DGM" — it's **close the last gap**:
the archive/lineage is in the Brain (`wubu_hive`), but the promotion gate +
verifier + operator live in the Body (`wubu_selfimprove`). The plan item is
to wire the Brain's archive to the Body's gated operator so a validated
mutation can actually promote through DA-3 with a trace span (plan item L3).

### 3.2 The known pitfall to design against: reward hacking
Every self-improvement system in the literature eventually games its own
metric. WuBu's answer must be: the verifier is *independent* of the mutator
(already true — separate `wubu_verifier_fn`), the eval is held-out (DGM used
disjoint benchmark segments — adopt this), and promotion is fail-closed
(DA-3). This is already mostly the design; the vault records it as the
non-negotiable invariant so future work doesn't erode it.

---

## Cross-cutting: the one-sentence synthesis

The three directives are one system: **a 9P-native OS where the Brain's
internals (KV, experts, memory, hive) are namespace files; a native
companion that reads that namespace + the screen as ambient context and acts
through capability-gated tools; and a self-improvement loop whose every
promotion passes an independent verifier behind a human gate.** Styx gives
the *plumbing*, the companion research gives the *UX shape*, and DGM + your
own DA-1/2/3 give the *learning invariant*. None of these are greenfield —
each maps to a module that exists. The work is wiring, not invention.
