# Symbolic Regression + Automated Theorem Proving + Invariant Discovery — 7-hop KB sweep
## EE axis: from data → equations → proofs → verified invariants (at home, C11)

> Each stone seeds the next hop. Target: map the "discover and prove" substrate
> WuBuOS lacks — turning observations into equations, equations into proofs,
> and proofs into certified invariants for the safety layer.

## Hop 1: Symbolic regression (equation discovery from data)
PySR / AI-Feynman / Genetic Programming: discover closed-form equations from
(x, y) data. Operators: +, *, sin, exp, log. Fitness = MSE + complexity penalty.
At home: our recursive_optimize produces tok/s observations as a function of
sweep config (15 dims). SymReg can discover the *governing law* relating config
→ tok/s, turning a black-box sweep into a white-box equation.

## Hop 2: SINDy (sparse identification of nonlinear dynamics)
SINDy: from trajectory data (dx/dt), build a library of candidate terms
(constants, x, x², sin(x), x·y, ...) and solve sparse regression (LASSO/STLSQ)
to find the few terms that matter. Recovers e.g. ẋ = -2x exactly.
At home: the AGI-OS operator's sweep produces a *trajectory* of (config_t, tok_s_t).
SINDy discovers the dynamical law: how tok_s evolves as configs change.

## Hop 3: Counterexample-guided inductive synthesis (CEGIS)
CEGIS: ∃f.∀x. φ(f,x). Loop: synthesize candidate f from grammar → verify
(∀x sound) → if fail, verifier returns counterexample → refine grammar.
At home: given a spec (e.g. "max throughput config"), CEGIS searches the config
grammar, tests candidates, and converges on the optimal config with proof of
optimality (verifier found no counterexample).

## Hop 4: Automated theorem proving (Lean-style proof search)
ATP: given a conjecture, search for a proof in a formal system. Lean Copilot
uses LLMs to suggest proof steps; the kernel verifies each step soundly.
At home: we need a lightweight *propositional* prover — natural deduction for
boolean/arithmetic facts. The causal SCM (AW) + symbolic rules (AW) produce
conjectures (e.g. "if KV=512K then tok_s > 25"); the prover checks them.

## Hop 5: Invariant discovery (loop invariants for the safety layer)
Loop invariants: a property P(x) that holds before/after every loop iteration.
Discovered by symbolically executing the loop + solving for P that is inductive.
At home: recursive_optimize's sweep loop needs an invariant (e.g. "tok_s is
monotonic non-decreasing on accepted configs"). The prover discovers + certifies
it, feeding the loopguard (AG-01) a *proof* not just a heuristic.

## Hop 6: Integration with causal + safety substrate
The discover-and-prove loop:
  1. recursive_optimize sweeps → trajectory data                  [recursive_optimize]
  2. SymReg/SINDy discovers law: tok_s = f(config)                [EE01, EE02]
  3. CEGIS searches config grammar for optimal f*               [EE03]
  4. Prover checks conjecture: "f*(config) ≥ 25 tok/s"           [EE04]
  5. Invariant discovery certifies loop invariant for sweep      [EE05]
  6. SafeKern/loopguard consume certified invariant              [safekern, loopguard]
  7. Causal SCM (AW) explains WHY the law holds                  [AW01-04]

## Hop 7: Closed-loop self-verification
The equation + proof become part of the agent's *knowledge* (symbolic KB +
vector store). Next sweep uses the predicted law to warm-start (not blind).
If the law is violated (world shifted), task-boundary detection (BB03) fires →
re-discover. This is continual learning (BB) of *laws*, not just params.

## Gap mapping
- EE01 Symbolic regression (genetic-programming equation discovery) `wired` (wubu_symreg.c)
- EE02 SINDy (sparse dynamics identification) `wired` (wubu_sindy.c)
- EE03 CEGIS (counterexample-guided config synthesis) `wired` (wubu_cegis.c)
- EE04 Automated theorem proving (natural-deduction prover) `wired` (wubu_prover.c)
- EE05 Invariant discovery (loop invariant synthesis) `wired` (wubu_invariant.c)
- EE06 Integration: discovered law → loopguard/safekern `wired` (test_ee.c)
- EE07 Closed-loop self-verification (re-discover on shift) `open` (research: needs world-model)
