# SI Alignment Matrix — United States SI goals ↔ WuBu goals

> 2026-10-18. Sources were fetched, not remembered: EO 14434 (29 Sep 2026),
> EO 14409 (2 Jun 2026), the NSPM-11 fact sheet (5 Jun 2026), *America's AI
> Action Plan* (Jul 2025, PDF), and NIST's SI pages / AI RMF.
>
> "SI" is the US government's term of art per EO 14434 §1–3. Internally we
> still say AGI; in policy-facing documents we say SI. The EO's working
> definition is the existing 15 U.S.C. §9401(3) definition of AI until a
> statutory SI definition supersedes it.

## 1. What the US actually asks for (verified)

| # | Source | Objective | What it concretely asks |
|---|---|---|---|
| U1 | Action Plan Pillar I | Accelerate innovation | remove regulatory barriers; no mandatory licensing (EO 14409 §3(c) is explicit) |
| U2 | Action Plan Pillar I | Free speech / objective truth | systems must not be built for ideological bias; procurement condition |
| U3 | Action Plan Pillar I | Open-source & open-weight | encourage release of open models and weights |
| U4 | Action Plan Pillar I | Adoption & workforce | adoption across economy; workers gain, not replaced |
| U5 | Action Plan Pillar I | Manufacturing & science | next-gen manufacturing; AI-enabled science; world-class datasets |
| U6 | Action Plan Pillar I | Interpretability, control, robustness | explicit research investment |
| U7 | Action Plan Pillar I | Evaluations ecosystem | build public/private evaluation capacity |
| U8 | Action Plan Pillar I | Government & DoD adoption | accelerate federal and defense use |
| U9 | Action Plan Pillar I | IP & synthetic media | protect innovations; combat synthetic media in courts |
| U10 | Action Plan Pillar II | Infrastructure | permitting, energy, semiconductors |
| U11 | Action Plan Pillar III | Diplomacy, standards, exports | American SI as global gold standard |
| U12 | Action Plan Pillar III | National-security evaluations | CBRNE + cyber risk evaluation of frontier models |
| U13 | Action Plan Pillar III | Biosecurity | nucleic-acid synthesis screening, customer verification |
| U14 | EO 14409 (Jun 2026) | Cyber defense | harden systems; AI cyber clearinghouse; covered frontier model framework (voluntary, 30-day pre-release access) |
| U15 | EO 14409 §4 | Criminal misuse | enforce computer-crime statutes against AI-enabled intrusion |
| U16 | NSPM-11 (Jun 2026) | Defense adoption | robust, steerable, controllable; clear accountability; no vendor can disable a fielded system without approval |
| U17 | NSPM-11 | Civil liberties | no censorship, no ideological bias, no unlawful surveillance |
| U18 | EO 14434 | Terminology & definition | use "SI"; propose a statutory definition within 60 days |
| U19 | NIST / CAISSI | Standards & T&E | measurement science, testing/evaluation ecosystem, voluntary standards, international alignment |
| U20 | March 2026 framework | Children, communities, creators | protect children; strengthen communities; support creators |

## 2. Where we already align (keyword-mapped candidates, not adjudicated)

Counts are goals whose title/criterion/module match the theme (case-insensitive).
They are candidates for review, not a claim that each goal satisfies the policy.

| US objective | matching goals | representative goal IDs | strongest domains |
|---|---|---|---|
| U6 interpretability/control/robustness | 65 | 10, 12, 27, 38, 116, 119 | COMPILER 15, AGIEDR 7, UX 6 |
| U7 evaluations ecosystem | 117 | 7, 18, 59, 80, 91, 96 | COMPILER 36, STACK 17, BRAIN 13 |
| U3 open-weight/reproducibility | 32 | 197, 249, 258, 261, 270 | BRAIN 7, SECURITY 5, STACK 4 |
| U5 science & datasets | 13 | 150, 160, 180, 188, 239 | COMPILER 4, BRAIN 4, PATHS 2 |
| U14 cyber defense | 24 | 353, 365, 583, 584, 585 | SECURITY 8, AGIEDR 6, OMARCHY 4 |
| U10 critical infrastructure | 27 | 15, 27, 84, 184, 483 | UX 9, WORLDSIM 4, FS 4 |
| U10 infrastructure & hardware | 46 | 3, 20, 21, 28, 61 | COMPILER 12, DRIVERS 9, KERNEL 7 |
| U8/U16 government & defense | 51 | 81, 89, 142, 143, 144 | AGIEDR 16, DRIVERS 6, AGENT 5 |
| U11 standards & exports | 72 | 1, 10, 12, 13, 14 | NAMESPACE 28, KERNEL 15, FS 9 |
| U4 workforce & education | 15 | 211, 294, 361, 663 | STACK 6, PATHS 4, BRAIN 2 |
| U5 manufacturing | 7 | 170, 173, 175, 192 | COMPILER 4 |
| U13 biosecurity | **2** | 871, 964 | NETWORK 1, DRIVERS 1 |
| U9 synthetic media provenance | 72 | 3, 7, 18, 39 | DRIVERS 11, NAMESPACE 10 |
| U20 child & community safety | 36 | 41, 317, 342, 360 | RUNTIME 10, FS 7, SECURITY 5 |
| U9 IP & licensing | 35 | 32, 34, 51, 86 | NETWORK 7, KERNEL 5, BRAIN 5 |
| U14 incident readiness | 50 | 35, 57, 67, 184 | UX 7, AGIEDR 6, AGENT 5 |

## 3. The mapping that matters: policy → goal → armor → gap

| US objective | Our vehicle | Goal IDs | Armor invariant | Status | Gap / next action |
|---|---|---|---|---|---|
| U6 interpretability/control | `wubu_manifoldviz`, probes, `wubu_armor` | 10, 12, 27, 38 | Truth + non-domination | partial | no mechanistic-interp suite over the manifold; build probe suite next |
| U7 evaluations | `gauntlet_runner`, `test_all`, oracles | 7, 18, 59, 80 | Truth + repair | **strong** | publish a machine-readable eval report (JSON) per run |
| U3 open-weight | GGUF/safetensors loaders, umbrella licence | 197, 249, 258 | Truth + solidarity | strong | document the exact reproduction recipe per release |
| U5 datasets/science | corpus pipeline, `wubu_synth` | 150, 160, 180 | Truth | partial | no dataset provenance ledger; add one |
| U14/U15 cyber | `wubu_safekern`, seccomp containers, `wubu_capzero` | 353, 365, 583 | Faith (shield) + repair | partial | port the AI-cyber clearinghouse concept: a local vuln scan/disclosure path |
| U10 infrastructure | thermal governor, NUMA, driver frontier | 3, 20, 21, 28 | Sabbath (limits) | partial | energy-per-token already measured; add a hard thermal budget API |
| U8/U16 defense adoption | capability authority, NT object manager | 81, 89, 142 | Salvation (identity) + consent | partial | "no vendor can disable a fielded system" needs a local override/escrow policy |
| U11 standards | 9P/Styx, POSIX-ish paths, `zealos_parity.h` | 1, 13, 14 | Truth | strong | write an SI conformance profile (what a WuBuOS SI node guarantees) |
| U13 biosecurity | **none** | 871, 964 | Dignity/repair | **gap** | add a nucleic-acid-synthesis screening policy hook at the device/agent boundary |
| U5 manufacturing | PLC/robot drivers? | 170, 173, 175 | Solidarity | **weak** | decide whether to serve manufacturing at all; if yes, a real-time fieldbus goal |
| U4 workforce | docs/tutorials | 211, 294, 361 | Solidarity | weak | a "learn the stack" path that is not just docs |
| U9 synthetic media | attestation | 3, 7, 18 | Truth | partial | content provenance (C2PA-style) in the FS namespace |
| U20 children | content guardrails | 41, 317, 342 | Dignity + repair | partial | explicit child-safety acceptance tests |
| U18 terminology | this doc | — | — | **action** | policy-facing docs use "SI"; code keeps AGI naming |

## 4. What we must NOT do (the policy forbids it, and so does the armor)

- **No mandatory licensing or preclearance.** EO 14409 §3(c) says nothing in
  the order authorizes a mandatory licensing, preclearance, or permitting
  regime for model development or release. Any internal gate that becomes a
  permission-to-exist is out of bounds.
- **No censorship or ideological bias** (U2, U17). The armor's truth invariant
  forbids deception; it does not license a commissar.
- **No unlawful surveillance** (U17). The hidden acts are not ours to judge
  (ST I-II q.91 a.4).
- **No vendor kill-switch over a fielded system** (U16) — and, symmetrically,
  no vendor lock-in: our licence and the jubilee invariant require a release
  path.
- **No biosecurity theatre.** If we add screening, it must be enforceable at a
  real boundary, not a checkbox.

## 5. The OS thesis (why this is an operating-system project)

The US plan is a *stack* strategy: innovation, infrastructure, and international
leadership. A complete OS is the only artifact that touches all three at once:

- **Body (WuBuOS)** supplies namespaces, scheduling, memory, devices, security
  and UX — the substrate a local SI runs on (U10, U14, U16).
- **Compiler (WuBuNOS)** supplies the ISA backends and the JIT/MIR that make the
  SI portable across hardware — the sovereignty layer (U11).
- **Brain (wubuwizard)** supplies the model, the KV filesystem, the manifold
  geometry and the training loop (U5, U6, U7).
- **Armor** supplies the normative contract (U2, U17, U20).
- **Licence + G39** supply the civic and governance layer (U9, U11).

That is the alignment: not a document that says we agree with the policy, but a
stack where each policy objective has an owning subsystem, a test, and a named
gap.

## 6. Honest status

- The keyword counts are **candidates**, not compliance. A goal can mention
  "eval" and still be a stub.
- Two objectives are genuinely unaddressed: **biosecurity** and
  **manufacturing**. Workforce is thin.
- The defence-adoption objective raises a hard design question (vendor disable
  authority) that we have not answered; it is listed as an open gap rather than
  hand-waved.
- Nothing here is legal advice or a claim of government endorsement.

## Refresh log

- **2026-10-18 (portfolio cycle 2)** — armor suite re-run green inside the full
  gate. No objective status changed. New work this cycle hardens two rows
  indirectly: AGENT #336 (cross-process promotion lock) supports the
  *evaluations* objective by making promotion exclusive across hive processes;
  BRAIN #244 (public orthogonalization primitive) supports *standards* by
  giving the optimizer a testable convergence contract. Biosecurity,
  manufacturing and workforce remain the honest gaps.
