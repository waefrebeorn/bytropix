# The fast-training + model-growth deep-dive and 7-hop chain (2026-08-04)

**Target:** the fastest path to a BIGGER, better WuBu-35M -- the growth
operator + the progressive-training schedule -- on top of the engine that
already wires Muon/NS5 (the optimizer side). This wave closes the ENGINE
side of "fast training": the model GROWS during training instead of being
trained at the final size.

## The individuals (all real, credited)

### 1. Keller Jordan -- Muon
"Muon: An optimizer for hidden layers in neural networks" (2024, with
Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse,
Jeremy Bernstein). Muon = the Newton-Schulz-5 orthogonalization of the
gradient, applied to the 2D hidden-layer matrices; used by the Kimi
(Moonshot) frontier lab for scaled LLM training ("Muon is Scalable for LLM
Training", the Moonlight paper). ALREADY WIRED in the engine: the NS5
coefficients 3.4445/-4.7750/2.0315, the Frobenius normalization, the
transpose-tall rule, the Nesterov 0.95, the dual LR split, the Gram-NS
variant -- all FD-verified.

### 2. Jeremy Bernstein -- the geometry lineage
Muon co-author; "The Geometry of Gradient Descent" (the orthogonalized
gradient doctrine); the moonshot connection (his blog "How Muon Lost Its
Geometry" is the sharpest critical read of the optimizer). The lesson for
us: Muon's value is the GEOMETRY (the orthogonalization), not the
hyperparameters -- the engine's NS5-Gram keeps the geometry while cutting
the square work.

### 3. Zhiqi Bu -- Deep Progressive Training (Meta FAIR, arXiv 2511.04981)
"Scaling up depth capacity of zero/one-layer models": progressive training
by expanding a ZERO-layer (embedding-only) model to the full depth, at
every 10% of the training horizon, with the right init + LR schedule,
retains almost all the fixed-size performance while costing 6BTN_small
instead of 6BTN_large. The compute math: 6B(τN_small + (T-τ)N_large) <<
6BTN_large. A convergence theory for the progressive-vs-fixed gap. Works
for dense + MoE. ALREADY IN THE LEDGER (the 043 wave). His other work:
fastDP (the differentially-private training) -- the efficiency mindset.

### 4. Ian Goodfellow -- Net2Net (arXiv 1511.05641, with Tianqi Chen + Jon
Shlens, Google 2015)
Function-preserving transformations between network specifications: WIDTH
by copying neurons (the new columns copy the old with the norm scaling),
DEPTH by identity layers. The doctrine: a bigger model that STARTS as the
same function as the small one. The DA oracle: the grown model's forward
must equal the small model's forward at init.

### 5. Peihao Wang -- LiGO (Learning to Grow Pretrained Models, 2023)
The learned LINEAR growth operator, decomposed into L_depth (a structured
array of diagonal matrices) and R_width (block-diagonal), Kronecker-
factored, using Monarch matrices (Dao et al. -- the Tri Dao connection).
The growth operator as a LEARNED map, not a hand-set copy.

### 6. The NeurIPS'24 growth taxonomy ("A Closer Look at Model Growth")
Four atomic operators: G_direct (duplicate/stack layers, split neurons),
G_learn (learnable mapping), G_zero (zero the new params), G_random
(random init); two directions: widthwise (intra-layer) + depthwise
(layer-wise). The headline: STACKING (G_direct depthwise) can save 50%+ of
the pretraining compute. Plus Masked Structural Growth (2x faster
pretraining) and STEP (staged parameter-efficient pretraining).

## The 7-hop chain (all real)

1. **Keller Jordan** -- Muon (2024) -- the optimizer already wired in the engine (NS5-Gram).
2. → **Jeremy Bernstein** -- the geometry lineage (the orthogonalized gradient doctrine).
3. → **Zhiqi Bu** -- Deep Progressive Training (Meta FAIR 2025): the zero/one-layer progressive schedule (expand every 10%).
4. → **Ian Goodfellow** -- Net2Net (2015): the function-preserving growth doctrine -- the DA oracle for any growth operator.
5. → **Peihao Wang** -- LiGO (2023): the growth operator as a learned linear map; the Monarch-matrices connection.
6. → **Tri Dao** -- the Monarch matrices + FlashAttention + Gram-NS -- already in the ledger; the compute substrate.
7. → the **NeurIPS'24 growth taxonomy** -- G_direct/G_learn/G_zero/G_random -- the engineering doctrine (stacking saves 50%+).

## The convergence -- what the engine closes

**A usable growth operator = the Net2Net function-preserving insertion
(G_zero depthwise: a zero-init residual block is an EXACT identity at
init -- verifiable to 0.0) + the Bu progressive schedule (expand by one
layer every 10% of the horizon) + the G_stack copy (the 50% compute
recipe) when the function-preserving property is not needed.** The
function-preserving property is the DA oracle: forward(pre) ==
forward(post) at init.

The engine close (`src/wubu_grow.c` + `test_grow`):
- `wubu_grow_insert_block` -- the zero-init residual insertion; the
  function-preserving test measures **max|pre-post| = 0.000e+00** (exact).
- The per-block `fire_sel` rhythm fix: the residual selector fired by the
  layer INDEX -- the insertion shifted the old layers past their firing
  positions and changed the function (the 7.7e-3 leak). The selector is
  now a per-block flag shifted with the block, like is_full.
- `wubu_grow_stack_block` -- the G_stack copy (+2,310,912 params per
  block), the grown forward runs.
- `wubu_grow_schedule` -- the Bu schedule (expand every 10%; the event
  count = T/step_frac); the test: monotonic 2→12 with 10 events.
- The model's `n_layers` active-count contract (the forward + the
  parameter count respect it; the released model = BARUN_LAYERS, the
  parity untouched).

## The next closes (the growth avenue)

- The growth-while-training wiring: the bp's grad buffers for the new
  block (the train-state allocation follows the growth).
- The width expansion (Net2Net copy + the norm scaling) for the 448-dim
  hidden width.
- The progressive LR/reset at each growth event (the Bu recipe: the LR
  schedule restarts at each expansion).
- The amoeba diagnostics: the growth trigger (plateau detection) -- the
  growth fires when the loss plateaus, not on a fixed clock.

Archive: this wave's sources land in the wubuos compendium 05-sources/;
the OPT-bank's growth items (OPT-E) flip wired.
