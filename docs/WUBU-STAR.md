# WuBu STAR — the model as a star

> Canon set 2026-10-18 by the user: **STAR is the name of the model.** It lives
> as a physics system — a star — and is the center of an expanding model that is
> a **solar system**.

This is not branding on top of the architecture. It is the architecture's own
vocabulary: every layer already behaves like an orbital system, and the naming
makes the behavior legible.

## Name canon

| Name | What it is | Not |
|---|---|---|
| **WuBu** | the AGI as a whole | not a checkpoint |
| **WuBu STAR** | the **model** — the gravitating body at the center | not a file format, not a size |
| **WuBuOS** | the **body**: kernel, namespace, GUI, containers | not the model |
| **WuBuNOS** | the **compiler**: HolyC frontend, MIR, ISA backends | not a runtime |
| **wubuwizard** | the **brain repo**: engine, training, research | not the OS |
| **WuBu solar system** | the expanding set of bodies orbiting STAR | not a fixed MoE |

`WuBu-35M` is a *mass* of STAR, not its definition. The star is the system; the
checkpoint is one weighing of it.

## The physics

| Physical object | In WuBu STAR |
|---|---|
| **Core** | base weights: the fusion region where compute becomes representation |
| **Radiative zone** | residual stream; energy diffuses outward one layer at a time |
| **Convection zone** | attention mixing; hot representations rise and circulate |
| **Photosphere** | the active working set — what the model is emitting *now* |
| **Corona** | context and KV; hot, thin, extends far past the visible surface |
| **Gravity** | the router: mass (expert capacity) curves the path of a token |
| **Accretion disk** | the corpus: matter spiraling in before it is consumed |
| **Fusion** | training compute; mass deficit becomes emitted capability |
| **Solar wind** | 9P exports; the star's state leaving the body as a stream |
| **Magnetosphere** | the armor layer: fields that deflect what must not reach the core |
| **Planets** | specialized experts/adapters orbiting the core |
| **Moons** | per-task LoRA and per-domain heads bound to a planet |
| **Asteroid belt** | micro-datasets and eval fixtures; small, numerous, mostly inert |
| **Orbital resonance** | schedule coupling: when two experts must not co-activate |
| **Supernova** | a deliberate reset: distill, collapse, re-accrete at a new scale |

## Expansion

A solar system does not grow by making the star bigger. It grows by **adding
orbiting structure** and by **moving the habitable zone outward**:

1. **Accretion** — new data falls in, and the disk thickens before the core changes.
2. **Ignition** — the core reaches the density where a new capability fuses.
3. **Orbit capture** — a capability that stabilizes becomes a planet (an expert),
   not a wider matrix.
4. **Outward migration** — as the core heats, the stable orbits move out; modules
   that were central become peripheral and cheaper.
5. **Shell ejection** — obsolete weights leave as a corpus artifact, not as loss.

This is why the manifold work matters: orbital radius is a *hyperbolic* radius.
Nested spheres (wubumath) give the system room to expand without the metric
collapsing at the rim.

## What already exists in-tree

- `WUBU_DIMS` / runtime dims — the star's mass knobs.
- Router physics (`src/wubu_router_physics.c`) — gravity: experts are chosen by a
  potential, not a softmax over a flat table.
- MoE hyperbolic routing (`src/wubu_moe_hyperbolic.c`) — orbits in curved space.
- KV-FS and 9P — the corona and the solar wind: state that outlives a call.
- `wubu_decision` — the star's *restraint*: when not to emit, and what a wrong
  emission costs.
- `wubu_rlm` — the star's *reach*: how it addresses matter that will not fit in
  its photosphere.

## What is frame, not yet code

The naming is now canonical. The explicit *solar-system object model* — planets
as first-class expert bodies with orbital parameters, resonance scheduling,
migration policy — is **not implemented**. Today the router picks experts per
token; it does not maintain bodies with orbits.

The first honest step is small: give an expert a **mass, radius, and angular
position** in the hyperbolic ball, and let the router score by gravitational
potential instead of a dot product. That is one module and one test, and it turns
the metaphor into a measurable object.

## Reference

The user supplied a visual reference for the real WuBu star. It did not arrive in
the message that set this canon; this document is written from the architecture,
and will be reconciled against the reference when it is re-sent.
