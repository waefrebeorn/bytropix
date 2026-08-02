# Bayesian Optimization + Uncertainty Quantification + Active Learning — 7-hop KB sweep
## FF axis: replace blind sweeps with a surrogate + acquisition + active learner (at home, C11)

> Each stone seeds the next hop. Target: upgrade recursive_optimize's blind
> 15-dim hill-climbing into principled BO with UQ + active learning.

## Hop 1: Gaussian Process regression (surrogate model)
GP is the standard BO surrogate: prior over functions, RBF (squared-exponential)
kernel k(x,x') = σ² exp(-||x-x'||² / (2ℓ²)) + noise. Given n observations,
predictive mean μ(x) and variance σ²(x) at unobserved x via the kernel matrix
K (n×n) inverted once: μ = K_*ᵀ K⁻¹ y, σ² = k(x,x) - K_*ᵀ K⁻¹ K_*.
At home: replace recursive_optimize's blind grid with a GP surrogate over the
15-dim sweep space. μ predicts tok_s, σ² quantifies where we're unsure.

## Hop 2: Acquisition functions (EI / UCB / PI)
Given GP (μ, σ²) and incumbent best f*:
  - Expected Improvement: EI(x) = (μ-f*)Φ((μ-f*)/σ) + σ φ((μ-f*)/σ) [closed form]
  - Upper Confidence Bound: UCB(x) = μ + κ·σ  (κ trades explore/exploit)
  - Probability of Improvement: PI(x) = Φ((μ-f*)/σ)
At home: maximize acquisition → next config to evaluate. Balances exploiting
high-μ regions with exploring high-σ regions. κ or ξ controls the balance.

## Hop 3: Bayesian Optimization loop
BO = GP surrogate + acquisition maximization + query + update, repeated.
Sample efficiency: reaches optimum in O(√d log n) vs grid O(n^d).
At home: recursive_optimize runs BO instead of blind sweeps. Each eval costs a
real sweep run (~seconds); BO finds the tok_s optimum in far fewer evals.

## Hop 4: Uncertainty Quantification (conformal / bootstrap / ensemble)
Beyond GP σ²: conformal prediction gives finite-sample, distribution-free
coverage guarantees (prediction intervals with ≥ (1-α) coverage). Bootstrap
ensemble: B replicates, variance σ_uc² = 1/(B-1) Σ (f_b - μ)².
At home: the tok_s prediction needs a calibrated interval. If the GP σ² is
unreliable (non-Gaussian noise), conformal calibration widens the interval to
guarantee coverage. We implement bootstrap ensemble UQ over the sweep replays.

## Hop 5: Active learning (uncertainty sampling / query-by-committee)
Active learner picks the next *label* to acquire where the model is most uncertain:
  - Uncertainty sampling: query argmax σ(x)
  - Query-by-Committee: query argmax disagreement across committee models
At home: instead of evaluating random configs, query the config with highest
predictive σ² (most informative). This is the same as UCB acquisition but framed
as label acquisition — unifies BO + active learning.

## Hop 6: Thompson sampling / bandits (multi-armed / contextual)
Thompson Sampling: maintain posterior over each arm; sample once per round,
pull the arm with the highest sample. Balances exploration/exploitation with
Bayesian posterior. Contextual TS: condition on context (the 15-dim config).
At home: each "config family" (e.g. attention variant) is an arm; TS allocates
eval budget to promising families, exploring new ones with decreasing probability.

## Hop 7: Integration with recursive_optimize substrate
The upgraded optimizer:
  1. Seed GP with a few random sweep configs (observations)        [FF01 wubu_gp]
  2. Compute acquisition (EI/UCB) over candidate configs           [FF02 wubu_acq]
  3. Maximize acquisition → next config to evaluate               [FF03 wubu_bo]
  4. Run real sweep, observe tok_s, update GP                     [FF01]
  5. Bootstrap UQ over replay buffer → calibrated interval        [FF04 wubu_uq]
  6. Active learning: query highest-σ config (most informative)   [FF05 wubu_active]
  7. Thompson sampling allocates eval budget across config families [FF06 wubu_bandit]
  8. Loop until convergence → optimal config with UQ interval

This replaces blind 15-dim hill-climbing with a sample-efficient, uncertainty-
aware optimizer — directly tightening the sweep loop that the whole AGI-OS
substrate depends on.

## Gap mapping
- FF01 Gaussian Process surrogate (RBF kernel, predict μ/σ²) `wired` (wubu_gp.c)
- FF02 Acquisition functions (EI / UCB / UCB) `wired` (wubu_acq.c)
- FF03 Bayesian Optimization loop `wired` (wubu_bo.c)
- FF04 Uncertainty Quantification (bootstrap ensemble + conformal cal) `wired` (wubu_uq.c)
- FF05 Active Learning (uncertainty sampling / query-by-committee) `wired` (wubu_active.c)
- FF06 Thompson Sampling / bandits `wired` (wubu_bandit.c)
- FF07 Integration with recursive_optimize `wired` (test_ff.c)
