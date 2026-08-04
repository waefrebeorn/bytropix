# 051 — Triple-DA: the amoeba-on-nested-spheres methodology vs the lab convergence

> Status: `closed` (reviewed + philosophy archived as corpus knowledge).
> Date: 2026-08-04.
> The user's design philosophy (verbatim intent): "our model is like an
> amoeba that can start very small, grow very large, absorb information,
> and run for core model system needs on the very most dense layers —
> because we are running off of the sphere style system of nested
> spheres." This doc is the Devil's-Advocate audit of that methodology
> against research/050 (the 8-lab convergence).

## The philosophy being audited (the user's design)

1. **The amoeba body**: not a fixed-size network — a colony of cells
   (experts/blocks/neurons) watched by a diagnostic immune system that
   grows (pseudopod: split the overworked), shrinks (apoptosis: prune +
   recycle), or stays (stasis). Every mutation validated + archived,
   mistakes rolled back by the 5+1 recovery. (docs/wubu-amoeba-design.md)
2. **The nested spheres**: the body lives in a product of K Poincaré
   balls with K curvatures (wubu_nested_ssm, THEORY/03) — hyperbolic
   geometry IS the architecture, not a feature. Nested scales =
   Russian-doll levels, each its own adaptive hyperbolic space with
   learnable curvature. Dense layers handle the core system needs;
   the sphere nesting gives multi-scale abstraction.
3. **Start small, grow large**: 35M → 70M → back to 30M — the model
   adapts to the task, not to a fixed config. The most dense layers are
   the core; the spheres give the structure.

## DA-1 (Decision): where the lab convergence CONFIRMS the philosophy

| Lab rule (research/050) | The amoeba-sphere mapping | Verdict |
|---|---|---|
| Data mix 50/25/17/8 (Llama 3) | our cosmopedia/finemath/openmath mix | ✓ same diet feeds any body |
| Muon for matrices + AdamW for vectors (Moonlight) | the growth engine — cells learn only with real gradients | ✓ CONFIRMED — the amoeba's growth engine is the labs' optimizer |
| WSD / warmup→const→cosine→anneal (MiniCPM, DeepSeek, Llama) | the colony trains between mutations on this schedule | ✓ adopt the anneal phase |
| SFT cold-start at LOW lr (~1e-5) before RL (R1, GLM, Llama) | the colony's first alignment — delicate, low-lr | ✓ adopt |
| GRPO verifiable rewards (DeepSeek V3+R1) | the colony's natural selection — fitness measured not assumed | ✓ the amoeba's validate step IS GRPO-style fitness |
| Multi-token prediction λ=0.3 (DeepSeek) | denser learning per token | ✓ add when trainer supports |
| Stability gate, no rollbacks (DeepSeek) | the 5+1 recovery + wubu_priority ledger | ✓ CONFIRMED — DeepSeek's "no rollbacks" is our safety net |
| Small models overtrained (Llama 3: smaller trained longer than compute-optimal) | the amoeba starts small and trains long | ✓ the start-small philosophy is lab-endorsed |
| MoE fine-grained experts (DeepSeekMoE) | the colony cells ARE fine-grained experts | ✓ the amoeba body is the MoE evolution |

**DA-1 verdict: the lab recipe is the amoeba's METABOLISM — the rules
for how cells learn (optimizer, schedule, data, SFT, RL) all transfer
directly. The labs confirm the substrate; the amoeba adds the body that
grows on it.**

## DA-2 (Design): where the philosophy DIFFERS from (or stresses) the labs

1. **Fixed vs morphing architecture.** Every lab trains ONE fixed shape
   (DeepSeek 671B-MoE, Llama 405B dense). The amoeba changes shape
   mid-training. The labs' hyperparameter sweeps were done on fixed
   shapes; a morphing body invalidates the assumption that the LR /
   batch / schedule tuned for shape N applies at shape N+1.
   → Mitigation: the lab rules are scale-adaptive where it matters
   (WSD branches from stable checkpoints — MiniCPM's insight is
   EXACTLY the amoeba's branch-validate-archive loop). Adopt WSD as the
   schedule so every mutation branches from a stable checkpoint.
2. **Dense-core vs sparse-everywhere.** The user's design: the most
   dense layers are the CORE (system needs), the spheres nest around
   them. The labs run dense OR sparse-uniformly. A dense-core +
   sparse-shell topology is NOT covered by any lab recipe — it's the
   unique claim, and it must be validated on its own (does the dense
   core actually carry the "system" function? measure it with the
   block-importance oracle: wubu_bi).
3. **Nested-sphere SSM vs standard attention/MLP.** The labs' recipes
   (LR, batch, anneal) were derived on standard transformer blocks.
   Hyperbolic SSM blocks (Möbius addition, exp/log maps, per-ball
   curvature) have different loss-landscape geometry; the FD-verified
   backward (wubu_nested_ssm_backward) is what makes the standard
   recipe usable at all. Risk: the labs' LR range may not transfer to
   curved blocks — measure, don't assume (the HAKMEMQController /
   meta-controller idea from THEORY/01 is the honest answer: the model
   tunes its own LR from diagnostics).
4. **The prover gate.** No lab gates training on a Lean prover
   (Möbius closure, exp∘log, gyroassoc). The fitness gate is stricter
   than held-out loss — this is our own addition. Risk: over-strict
   invariants could block legitimate growth (the 5+1 rollback is the
   backstop). DA-2 rule: the prover guards GEOMETRIC invariants (the
   spheres must stay spheres), loss gates FITNESS — two gates, one per
   concern.

**DA-2 verdict: the philosophy's novel parts (dense-core+shell, curved
blocks, prover gate) are NOT lab-covered — they carry their own risk and
must be validated with our own oracles (wubu_bi for the dense core,
FD-verified backward for the curved blocks, prover for geometry).**

## DA-3 (Diagnostic): the honest risks of the amoeba-on-spheres method

1. **Morphing cost.** Every grow/shrink costs a validation pass + a
   checkpoint slot. At 35M that's cheap; at scale it isn't. The lab
   answer: mutations are RARE (plateau-triggered), not per-step — the
   plateau detector + wubu_priority ledger keep the mutation rate
   honest. Measure the cost per mutation and cap it.
2. **The dense core could be a bottleneck.** If the "core system needs"
   concentrate in a few dense layers, those layers become the
   overworked cells → the amoeba grows them → the core is no longer
   dense. The wubu_bi oracle + the growth operator (function-preserving
   split) keep this healthy: splitting a dense layer into two
   specialized daughters preserves function while adding capacity.
3. **Nested-sphere curvature drift.** If the per-ball curvatures drift
   to extreme values, the exp/log maps can blow up numerically. The
   prover (gyroassoc invariant) + the safety kernel (no NaN) catch it;
   the FD tolerance (-ffast-math-honest 5e-2) is the measurement bar.
4. **The lab recipes assume one epoch of data.** The amoeba re-trains
   between mutations — multiple passes over the same corpus. The labs
   warn about overfitting on repeated data (DeepSeek's dedup, Llama's
   filter). The amoeba's counter: the anneal phase up-samples quality
   data and the priority ledger archives what worked — re-exposure is
   targeted, not blind.
5. **Start-small means start-scared.** The labs' "small models train
   longer" (Llama 3) is endorsement, but their small models were still
   trained on trillions of tokens. At 6.7B tokens WuBu-35M is
   UNDER-trained by lab standards — the growth to 70M must come with
   MORE data, not just more layers. The corpus wave (ponds, SFT,
   agentic) is the food supply; the amoeba grows when food is plentiful.

**DA-3 verdict: the method is sound but data-hungry and mutation-costly.
The mitigations already exist (WSD branches, plateau-triggered
mutations, wubu_bi oracle, prover + safety kernel, priority ledger).
The single biggest risk is under-training: grow the corpus with the
body.**

## The synthesis (what the Triple-DA converges on)

**The lab recipe is the amoeba's metabolism; the nested spheres are its
skeleton; the dense core is its spine. They are not in conflict — the
labs answer HOW cells learn, the philosophy answers WHAT the body is.**

The integrated methodology (the operating doctrine):
1. **Metabolism (from the labs)**: Muon+AdamW split, WSD schedule with
   branch-from-stable, warmup→const→cosine→anneal, data mix
   50/25/17/8-shaped, SFT cold-start at lr ~1e-5, GRPO with verifiable
   rewards, MTP λ=0.3, stability gate.
2. **Skeleton (the spheres)**: K Poincaré balls with learnable
   curvatures; the prover guards the geometric invariants; the
   FD-verified backward keeps the standard recipe valid on curved
   blocks.
3. **Spine (the dense core)**: the most dense layers carry the system
   needs; wubu_bi tells us which they are; the function-preserving
   grow operator lets the core expand without losing what it learned.
4. **Body (the amoeba loop)**: train (metabolism) → diagnose (immune
   system: util/grad/loss-delta/entropy + wubu_bi) → mutate
   (grow/shrink/stasis, plateau-triggered) → validate (prover + loss +
   safety) → archive/rollback (5+1 + priority ledger) → repeat.
   The body grows when food (corpus) is plentiful and shrinks when
   the task is easy — it adapts to the work, not to a config.

## Philosophy archived as corpus knowledge (the recursive loop)

The user's directive: "you will need to learn all of the design
philosophies as we go along." This doc + THEORY/01 + THEORY/03 +
docs/wubu-amoeba-design.md are ingested as Tier-3 methodology corpus
text (build: tools/build_methodology_tier.py → wubu-methodology.tok),
so the model itself learns the geometry-is-the-architecture philosophy
that built it.
