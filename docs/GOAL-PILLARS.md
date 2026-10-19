# GOAL PILLARS — organising the 1,000 goals around the design

Companion to `docs/GOAL-LAYOUT-REASSESSMENT.md` (which established
cluster-first closure). This page answers a different question: **what order
should the clusters be worked in, and which ones are the design?**

The registry keeps its invariant — exactly 1,000 goals, 16 domains — but the
*reading order* and *tier* now follow the doctrine (`wubu-design-philosophy`),
not whichever domain happens to be next in rotation.

## The six pillars

| pillar | doctrine | open / total | what it is |
|---|---|---:|---|
| **GEO** | geometry IS the architecture | 18 / 29 | Möbius/Poincaré operations, nested spheres, curvatures, exp/log, geodesics, backward passes |
| **BODY** | the amoeba loop | 44 / 62 | grow/shrink/mutate, block importance, archives, 5+1 rollback, RSI operator |
| **KVFS** | the KV cache is a file system | 85 / 128 | namespace, mounts, 9P/Styx, KV tiering, paging, audit to KV-FS |
| **ROUTE** | everything is a routing problem | 19 / 30 | hash router, DSA indexer, hyper-connections, backend dispatch, expert selection |
| **REV** | static → revolver | 54 / 67 | runtime dims, slots/cartridges, probing, registries, hot-swap |
| **MANIFOLD** | pre-train / post-train / fine-tune on the manifold | 8 / 16 | manifold loss, Möbius LoRA, GRPO via transport, tangent-space gradient rule |

(Keyword mapping is in the session notes; it is a reading aid, not a schema
field. A goal can belong to more than one pillar — `wubu_kvfs` sits in both
KVFS and REV.)

## Why this order

GEO first: every other pillar's correctness is stated in geometric terms (Möbius
closure, exp∘log, gyration preserving the inner product). If the geometry is
wrong, the loss gates are measuring the wrong space — the same failure the
manifold-training doctrine records (ambient gradient vs tangent gradient:
loss 1.07 → 30, point pushed out of the ball).

BODY second: growth without a correct geometry mutates into a worse shape.

KVFS third: the namespace is where state lives; serving and persistence both
depend on it, and it is the largest open block (85 goals).

ROUTE and REV fourth and fifth: they are performance/adaptivity layers that
presuppose a working body and a working namespace.

MANIFOLD last only in ordering — it is small (8 open) and already has shipped
pieces (`wubu_manifold_train`, Möbius LoRA, hyperbolic output projection).

## What "reorganise" changed

1. **Precedence, not membership.** All 1,000 goals remain; the plan is read
   GEO → BODY → KVFS → ROUTE → REV → MANIFOLD.
2. **Tier corrections where the backlog contradicts the doctrine:**
   - **292** Möbius gyration-aware backward — P2 → **P0**. The geometry has 0 of
     1 backward goal closed while every curved block depends on it.
   - **931** HKDF-Expand per-realm KV-FS session keys — P2 → **P1**. KV-FS is the
     85-goal pillar and its security boundary was sitting below P2.
3. **Identity fixed at the top of the docs.** No pillar is "35M": the body is
   runtime-dimensioned, and 35,073,216 / 56,376,832 are two geometries it has
   worn. See `docs/MODEL-IDENTITY.md`.

## How a pillar is worked

Same rule as the cluster-first policy: pick a module inside the leading pillar,
close its whole cluster or ship nothing, and record the gate. Do not open the
next pillar while the current one has an unverified module.

## Current thin spots (design-critical, low coverage)

- `wubu_mobius` — 1 goal, 0 done (this is the geometry core)
- `wubu_bi` — 1 goal, 0 done (the skeleton's importance measure)
- `wubu_amoeba` — 4 goals, 2 done
- `wubu_kvfs` — 4 goals, 2 done

These are the modules where "following the design" is most directly at stake,
and they are the reason GEO and BODY lead.
