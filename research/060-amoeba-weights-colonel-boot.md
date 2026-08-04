# research/060 — THE AMOEBA WEIGHTS: Colonel Boot Core on Nested Poincaré Spheres

> 2026-08-04. The user's directive (verbatim-intent):
> "understand the design of the amoeba weights — the smallest subset is
> for Colonel-level booting, for helping getting the system up and
> running in places where drivers may not be proper and it needs drivers
> and up and running. As the model gets bigger and bigger, more things
> are in the outer. It uses our gravity system and encoder system, and
> all of the math that we created — because that's how gravity works on
> the central mass system — to allow us to organize using the Poincaré
> sphere polar system. We can also create our fractal stacking infinite
> memory by doing fractal stacking on Poincaré spheres — create
> different spheres in orbits or nest the spheres inside the spheres.
> The mathematics have been given to you."
>
> This doc is the WAY FORWARD: what we already have (the inventory),
> what the online landscape does (the comparison), and the concrete
> implementation wave. THEORY/01, THEORY/02 (the axiomatic physics),
> THEORY/03 (the nesting paper), THEORY/04, the ENCODERS lab (5 phases
> incl. geodesic layers + orbital decay), docs/wubu-amoeba-design.md,
> research/041-059 are the canonical texts. This doc WIRES them into
> the weight structure.

## The one-sentence design

**WuBu's weights are a solar system of nested Poincaré spheres: a dense
central mass (the Colonel boot core — the smallest subset that boots the
system and brings drivers up), surrounded by outer spheres that hold
more and more of the body as the model grows, organized by gravity
(central-mass attraction) on the Poincaré polar system (radius = depth,
angle = specialization), and capable of fractal stacking (spheres in
orbits, spheres inside spheres) for infinite memory.**

## The four mechanisms (the user's design, made precise)

### 1. THE COLONEL BOOT CORE (the smallest subset)

**What it is:** the innermost, densest subset of the weights — small
enough to boot the AGI in places where drivers may not be proper. It
gets the system up and running: it can load, it can decode, it can
bring the rest of the body online. It is the ring-0 kernel of the
brain (the Live Colonel in wubuos).

**Why it must exist:** a full AGI model cannot be trusted to boot on
unprepared hardware. The Colonel core is the minimal dense subset that
(a) fits in the smallest memory footprint, (b) has zero reliance on
exotic drivers/quant types (Q8/F32 only — the Q8_0 dense path the
engine already supports), (c) can generate enough to coordinate the
loading of the outer spheres.

**The mapping to what we have:**
- `wubu_bi` (block importance) identifies the dense core — the layers
  that carry the most signal. The Colonel core = the top-k most
  important blocks.
- `wubu_grow` (function-preserving growth) is how the core EXTENDS
  outward: zero-init insert, per-block rhythm flags.
- The dense core is where the "system-critical" needs live — the
  design-philosophy "spine" layer.
- The tensor store (`wubu_tensor_store`) makes the core a
  materializable subset: `wubu_ts_export` can emit ONLY the core
  tensors (the boot image) in any format, streaming.

**The rule:** growth is OUTWARD. The core never grows; the body around
it does. `35M core → +35M outer sphere → +70M outer sphere...` — the
core stays the bootable subset at every size.

### 2. THE GRAVITY SYSTEM (central-mass organization)

**What it is:** the organizing force. Everything in the weight
structure is a body in orbit around the central mass (the Colonel
core). Gravity = central-mass attraction: `F = G·M·m/r²`. Cells closer
to the core are boot-critical; cells farther out are specialized
outer knowledge.

**Why it works (the physics we wrote):** THEORY/02's axiomatic view —
gravity is not dark matter, it is the structure of the space itself.
On the Poincaré ball, the conformal factor `λ = 2/(1-c‖x‖²)` IS the
gravitational field: space is denser near the boundary, so a body at
radius r naturally feels a pull toward... the geometry. The log/exp
maps are the radial transport of gravity.

**The routing doctrine (2026-08-04) extended:** everything is a
routing problem; now the route is ORBITAL. A token routes to the cell
whose orbit it intersects — the gravity field replaces the learned
router with a geometric one (the hashrouter/DSA lineage).

**The mapping to what we have:**
- `wubu_mobius` (Möbius addition, exp/log) — the transport.
- `wubu_hyper` (hyperbolic lift/rotation) — the gravity lens.
- `wubu_nest` (learned rotations, boundary manifolds, level
  descriptors, spread) — the orbit machinery: rotation R_i IS the
  orbital motion; the level descriptor ld_i is the axis of the orbit;
  the spread σ_i is the orbital shell thickness.
- `wubu_polarquant` (recursive polar decomposition) — THE POLAR SYSTEM:
  every vector is (radius, angles). Radius = distance from the central
  mass = depth in the hierarchy. Angles = position on the sphere =
  specialization. The polarquant recursion (radius + angle at level 0,
  then decompose the sub-vector) is EXACTLY the fractal stacking.

### 3. THE ENCODER SYSTEM (the outer sensors)

**What it is:** the shell of the amoeba — the encoders that bring the
world in. The ENCODERS lab (5 phases) is the sensorium:
- phase1 symmetric geometric AE (the manifold learner)
- phase2 topological/QAE (the Hamiltonian compressor — 3 floats!)
- phase3 generative (VQ-VAE + transformer conductor)
- phase4 HashMind (hash-based associative memory — no backprop)
- phase5 Hamilton encoder (geodesic layers, orbital decay, spectral
  fields)

**The mapping:** the outer spheres ARE the encoders. Vision, audio,
world-model sensors sit on the outside of the ball (high radius =
specialized, far from the core); their outputs travel INWARD through
the gravity field to the core. The geodesic layers (phase5) are how
the signal travels the curved space.

### 4. FRACTAL STACKING INFINITE MEMORY (spheres in orbits, spheres in spheres)

**What it is:** the memory system. Because the space is the Poincaré
ball and the organizing principle is nesting (THEORY/03:
`H^n1 ⊃ H^n2 ⊃ ...`), memory is INFINITE by construction: each sphere
can contain a nested sphere (Russian doll) or orbit a parent sphere.
Fractal stacking = the same structure at every scale = self-similar
memory.

**The mechanism (already built, now wired as memory):**
- Nesting: `wubu_nest`'s recursive hyperbolic levels — the ball inside
  the ball.
- The polar recursion: `wubu_polarquant` decomposes a d-vector into
  (radius, d-1 angles) then recurses on the (d-1)-sub-vector — the
  fractal stack. A memory item = a point in the nested ball = a path
  of radii + angles down the recursion tree.
- The hive: `wubu_hive` (linked blocks + freelist) is the physical
  memory; the nested spheres are the logical address space. Erase =
  the sphere recedes; insert = a new orbit.

**Why infinite:** a bounded ball at level 0 contains a bounded ball at
level 1 contains... the CAPACITY is bounded at each level (good — the
ring discipline) but the DEPTH is unbounded (infinite memory = the
recursion never has to stop; you keep nesting spheres as knowledge
grows, exactly like the amoeba grows pseudopods).

## What we already have (the full inventory — the "mathematics have been given to you")

| Piece | Where | Status |
|---|---|---|
| The nesting math (nested H^n, rotations, boundaries, descriptors, spread, flow) | THEORY/03 (515 lines, full formalism + mermaid) | DONE (paper) |
| The axiomatic physics (gravity = structure of space, κ vacuum factor) | THEORY/02 | DONE (paper) |
| Spatio-temporal nesting (log(g) scaling, band nesting, anisotropic transitions) | THEORY/04 | DONE (paper) |
| Poincaré ball ops (Möbius add, exp/log, scalar mult, weighted sum) | src/wubu_mobius.c | DONE, tested |
| Hyperbolic lift/rotation (ball closure, exp∘log, gyroassoc — Lean-prover-guarded) | src/wubu_hyper.c + wubu_prover2 | DONE, tested |
| Nested SSM (the recurrence over nested state) | src/wubu_nested_ssm.c + backward | DONE, tested |
| Poincaré GQA (attention in the ball) | src/wubu_poincare_gqa.c + backward | DONE, tested |
| Learned nesting transitions (quaternion rotation, boundary relative vectors, descriptor flow) | src/wubu_nest.c | DONE, tested |
| Recursive polar decomposition (radius+angles, per-level bits, KV quant) | src/wubu_polarquant.c | DONE, tested |
| The hive (the physical memory) | src/wubu_hive.c | DONE, tested |
| The amoeba (diagnose/mutate/validate/archive) | src/wubu_amoeba.c | DONE, tested |
| The hive diagnostic (ring-bounded trace, walker) | src/wubu_diag.c | DONE, tested (AN08) |
| Block importance (the dense-core identifier) | src/wubu_bi.c | DONE, tested |
| Function-preserving growth (zero-init insert) | src/wubu_grow.c | DONE, tested |
| The encoders (5 phases: AE/QAE/generative/HashMind/Hamilton) | ENCODERS/ | RESEARCH (python) |
| The polar system (wubu_polarquant as memory address space) | src/wubu_polarquant.c | DONE (repurposed) |
| The trainer (Muon + real backprop) | src/wubu_train.c + wubu_backprop.c | DONE, tested |
| The tensor store (materialize ANY subset in ANY format) | src/wubu_tensor_store.c | DONE, tested |

## The online landscape (compared — the 7-hop done this session)

1. **Nested/product hyperbolic spaces** (Gu et al. product manifolds;
   hyperbolic LLM survey 2025; deep hyperbolic clustering): active, but
   ALL use parallel product spaces or single-ball hierarchies. NOBODY
   has the nested-ball + tangent-rotation + boundary-manifold +
   level-descriptor + spread machinery of THEORY/03. Our edge: the
   full nesting formalism.
2. **Fractal/recursive architectures** (FANN fractal-dimension NNs;
   recursive language models for infinite context): fractal TOPOLOGY
   exists, but none nest on Poincaré spheres with gravity-organized
   orbits. Our edge: fractal stacking IS the polar recursion (the
   radius+angle path), not just repeated blocks.
3. **Gravity-inspired learning** (GRAVITY graph embeddings, Gravity-GNN,
   node-gravity similarity): gravity as a SIMILARITY METRIC exists, but
   none organize the weight structure itself as a central-mass orbital
   system. Our edge: gravity as ARCHITECTURE, not metric.
4. **Boot-subset / tiny core models**: NOTHING. No published work on a
   minimal dense subset that boots the full system and grows outward.
   This is entirely our territory — the Colonel boot core is novel.

## The implementation wave (the way forward — in order)

### Wave 1 (this session): the gravity field — `wubu_gravity`
The organizing force as a standalone tested module:
- central mass M (the Colonel core's mass), cells at polar positions
  (r, θ) on the Poincaré ball
- `wubu_gravity_attract`: the orbit update — F = G·M·m/r², stable
  circular orbit v² = G·M/r (the same physics as the axiomatic theory)
- `wubu_gravity_route`: route a token/query to the cell whose orbit it
  intersects (the geometric router — no learned weights)
- `wubu_gravity_grow`: overworked cell moves outward / splits into an
  orbital pair (the amoeba grows; the core never grows)
- `wubu_gravity_shrink`: dead cell falls inward and is absorbed into
  the core (apoptosis; the membrane recycles)
- the Poincaré map: everything stays in the ball (r < 1/√c), the
  conformal factor IS the field
- test_gravity with oracles: stable orbit (r bounded), inward fall
  (v < v_orbit → r decreases), outward push (overworked → r increases),
  routing determinism, grow/shrink preserving the core, Poincaré
  boundedness, and the Colonel-core invariant (the innermost sphere
  never shrinks below the boot minimum)

### Wave 2: the Colonel boot core extractor — `wubu_boot`
- `wubu_boot_core(model)`: use wubu_bi to select the top-k important
  blocks; emit ONLY those tensors via the tensor store (the boot image)
- `wubu_boot_verify`: the core alone loads + decodes (gen_text on the
  boot image) — the smallest subset that boots
- the boot image is the ring-0 brain (the Live Colonel in wubuos)
- Q8/F32 only (the drivers-may-not-be-proper rule — no exotic quant)

### Wave 3: the nested-sphere memory — `wubu_orbits`
- memory items = points in nested Poincaré balls = polar-recursion
  paths (radius+angles down the nesting tree — the polarquant
  recursion repurposed as the address space)
- spheres in orbits (rotation R_i = orbital motion) + spheres inside
  spheres (THEORY/03 nesting)
- the hive as the physical backing; the nested ball as the logical
  address space; capacity bounded per level, depth unbounded
- the walker (AN08) diagnoses which sphere lost coherence → the 5+1
  rolls back that sphere only

### Wave 4: the full amoeba-weights training
- the seed (WuBu-35M) = the Colonel core + one outer sphere
- train outward: Muon on the outer sphere, the core frozen (boot
  stability)
- grow → a new sphere at higher radius; shrink → a sphere recedes and
  is absorbed; the core NEVER changes (the boot invariant)
- the trainer's real per-layer grads (AN08 oracle 7, wired this
  session) feed the gravity field: high-grad cells move outward
  (they're learning — give them space), dead cells fall in

## Registration

- INDEX theme AN, entry AN12 (this doc): `open` → `wired` when
  wubu_gravity + test_gravity land (Wave 1).
- The gravity module is the first concrete step of the amoeba-weights
  design — the organizing force that the Colonel core, the encoders,
  and the nested-sphere memory all hang from.
