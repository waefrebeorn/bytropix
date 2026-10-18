# WuBu STAR — the model as a white dwarf

> Canon set 2026-10-18 by the user. The reference object is **WD J1330+6435**,
> the first star modelled to appear green. Image:
> `docs/assets/wubu-star-reference.jpg`.

## The reference object

| Property | Value | Source |
|---|---|---|
| Designation | WD J1330+6435 (SDSS J1330+6435) | Wikipedia |
| Type | **DZ** white dwarf — metal-polluted, nearly pure helium atmosphere (H < 1 in 10⁴) | Wikipedia; arXiv 2206.05258 |
| Distance | 281.6 ly / 86.35 pc, constellation Draco | Wikipedia |
| Effective temperature | **4,310 ± 190 K** (one analysis gives 3,660 ± 50 K) | Wikipedia; arXiv 2206.05258 |
| Spectrum | strong **Na I, Ca, Li** absorption; metals accreted from planetary debris | Wikipedia; RAS 2026 |
| Flux peak | ~**500 nm** | RAS 2026 colour study |
| Modelled colour | **chartreuse green**, `#DDFFB8` (figure) / `#EAFFA1` (Hollands model) | figure; RAS 2026 |
| Distinction | first star of any kind reported to appear green | RAS 2026; Wikipedia |

The green is not emission. Broad sodium absorption eats the orange, calcium eats
the violet, and what survives peaks near 500 nm. **The colour is a hole in the
spectrum** — the star is green because of what it absorbed.

## Why this object, not a main-sequence star

A white dwarf is a **remnant**. It has finished fusing. It is supported not by
fusion pressure but by **electron degeneracy pressure**, and it lives by
radiating away stored heat until it crystallizes. It got its metals by **eating
its own planets**.

That is a far better description of a trained model than a young star is:

| White-dwarf physics | WuBu STAR |
|---|---|
| Crystallized carbon/oxygen lattice | the quantized weight lattice at the core |
| Electron degeneracy pressure | the quantization floor (Q8/Q4): compression past it is a phase change, not a saving |
| **No fusion** | inference does not create capability; it radiates stored capability |
| Cooling curve (Mestel), latent heat at crystallization | the lifecycle: hot and plastic → cool and ordered; crystallization releases the last useful heat |
| Accretion of planetary debris | fine-tuning and corpus ingestion |
| Metals dye the atmosphere | **the model's colour is its provenance** — what you fed it is visible in the spectrum |
| Sedimentation of heavy elements | forgetting / consolidation: what sinks out of the atmosphere |
| Planetary nebula (ejected envelope) | shed weights and corpora, exported outward as artifacts |
| Chandrasekhar limit (~1.4 M☉) | the capacity limit: accrete past it without shedding and you get a runaway, not a bigger star |
| Nova / Type Ia | overfit event / catastrophic reset |
| Habitable zone migrates **inward** as it cools | capability migrates toward the core as the model cools — distillation inward while the shed shell expands outward |

The last row is the inversion that matters. The system does expand — but the
expansion is the **ejected envelope**, while the star itself contracts and its
habitable zone moves inward. WuBu STAR grows by **shedding and compacting**, not
by inflating. A model that only accretes ends as a supernova.

## Name canon

| Name | What it is | Not |
|---|---|---|
| **WuBu** | the AGI as a whole | not a checkpoint |
| **WuBu STAR** | the **model** — WD J1330+6435 as a physics system | not a file format, not a size |
| **WuBuOS** | the **body**: kernel, namespace, GUI, containers | not the model |
| **WuBuNOS** | the **compiler**: HolyC frontend, MIR, ISA backends | not a runtime |
| **wubuwizard** | the **brain repo**: engine, training, research | not the OS |
| **WuBu solar system** | the orbiting bodies: experts, adapters, tools, corpora | not a fixed MoE |

`WuBu-35M` is a *mass* of STAR, not its definition. The checkpoint is one
weighing of the star.

## The spectrum as the model card

| Spectral feature | Wavelength | Model meaning |
|---|---|---|
| Ca absorption (violet) | ~430 nm | the earliest, hardest refusals — the armor |
| Flux peak | ~500 nm | the working band: what the model is actually good at |
| Broad Na I doublet (orange) | ~590 nm | the absorbed middle: capability deliberately not emitted |
| Narrow dip | ~660 nm | the sharp failure mode; needs a named guard |
| Colour | `#DDFFB8` | identity accent for UI, docs, plots |

Machine-readable form: `docs/WUBU-STAR-CARD.json`.

## Design rules that follow

1. **No fusion at inference.** Every claim about capability must be a claim about
   stored heat. Measure the energy budget; do not imply generation.
2. **Quantization is degeneracy pressure.** There is a floor below which the
   lattice phase-changes. Q4 is that floor for this body; going lower is a new
   star, not a smaller file.
3. **Keep a pollution ledger.** If the colour is provenance, every corpus that
   changes the weights must be recorded with the line it dyes — dataset, tokens,
   and the eval that moved. Untracked accretion is how a model's identity becomes
   unexplainable.
4. **Identity is absorptive.** The green is a hole. Refusal policy is as
   definitional as output policy; the armor is part of the spectrum, not a wrapper.
5. **Shed before the limit.** Distillation and corpus export are not housekeeping;
   they are the planetary nebula. A star that cannot shed cannot stay a star.

## What exists, what is frame

Exists: the router already scores by potential (`src/wubu_router_physics.c`),
hyperbolic expert geometry (`src/wubu_moe_hyperbolic.c`), KV-FS/9P as the corona
and wind, `wubu_decision` as restraint and `wubu_rlm` as reach.

Frame, not code: experts do not yet have **mass, radius, or angular position**;
there is no pollution ledger; there is no crystallization/cooling model of the
weight lattice.

## First three concrete steps

1. **Star card as data** — `docs/WUBU-STAR-CARD.json` (name, hex, spectrum,
   temperature, type) so UI and docs read identity from one file. **Done.**
2. **Pollution ledger** — `tools/pollution_ledger.py` + `docs/POLLUTION-LEDGER.json`.
   Scans the 28-dataset pack, preserves eval-only flags, and refuses to claim a
   spectral line without a recorded before/after eval. **Done as tooling:** all
   28 datasets are present, **0 measured**, 6 eval-only — the honest baseline.
3. **Orbital expert body** — `include/wubu_orbit.h` + `src/wubu_orbit.c`:
   mass, position in the Poincaré disk, symplectic-Euler N-body integration,
   Boltzmann routing over gravitational potential, Kepler periods, resonance
   detection, and cooling migration to a core floor. `test_orbit` 22/22 (circular
   orbit drifts 0.26% over 5,000 steps). **Done as a module; not yet wired into
   `wubu_router_physics.c`** — the existing router still uses its own potential.

## Honest caveats

- Temperature estimates disagree (4,310 ± 190 K vs 3,660 ± 50 K); this document
  quotes both rather than averaging them.
- The figure's hex (`#DDFFB8`) and the colour study's model (`#EAFFA1`) differ;
  the figure is canonical for identity, the study value is cited.
- The solar-system object model is a frame. No orbiting bodies exist in code yet.
