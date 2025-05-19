Alright, Agent. The directive is clear: **Maximum Enhancement. Total Strategy. All In.** We're not just updating; we're evolving `WuBuSpecTrans_v0.1.1.py` into a flagship implementation of the advanced concepts outlined in "WuBu Nesting & Spatio-Temporal Dynamics... v5.19.25." This will be a significant undertaking.

**Overarching Philosophy:** Every change should be justifiable by the paper's principles (geometric adaptivity, anisotropic/resonant processing, adaptive strain engineering) or contribute to demonstrable robustness and performance. We are building a research platform.

---

**Total Strategy & Implementation Roadmap for `WuBuSpecTrans_v0.1.1.py`**

**Phase 0: Foundation & Setup (Agent: Execute these first)**

1.  **Version Control:** Ensure the current state is committed. Create a new branch for this major evolution (e.g., `wubu_evolution_v0.2`).
2.  **Configuration Overhaul:**
    *   **TODO:** Create dedicated `dataclass` or `OmegaConf`/`Hydra`-style configuration objects for each major WuBu stack (`WuBuSConfig`, `WuBuGConfig`, `WuBuDConfigPrimary`, `WuBuDConfigAlt`). These will replace the `_configure_wubu_stack` function and direct argparse parsing for these sections.
        *   Each config class should hold all relevant parameters from `DEFAULT_CONFIG_WUBU` plus new ones introduced below (e.g., `log_g_complexity_influence`, `anisotropy_type`, `resonance_gate_type`).
        *   Populate these from `args` in `main()`, but encapsulate the logic within the config classes or helper factory functions. This makes managing the burgeoning number of parameters cleaner.
        *   Include validation methods within these config classes (e.g., `validate_list_lengths_against_num_levels`).
    *   **TODO:** Update `parse_arguments` to reflect this. Arguments will now set fields in these config objects.
3.  **Enhanced Logging & Debugging:**
    *   **TODO:** Implement a more granular logging system. Allow different log levels for different modules (e.g., `WuBuLevel.L0` verbose, `QController` info).
    *   **TODO:** Integrate `torch.utils.tensorboard.SummaryWriter` alongside WandB for local, detailed inspection, especially for gradient norms, parameter distributions, and custom visualizations if WandB becomes too slow or cumbersome for rapid debugging.
4.  **Rigorous Unit Testing Framework:**
    *   **TODO:** Set up `pytest`. For every *new* significant geometric or adaptive component implemented below, write unit tests.
        *   Test forward passes with expected input/output shapes.
        *   Test numerical stability with edge-case inputs (zeros, large values, NaNs if handled).
        *   Test DDP compatibility for any new `nn.Module` that will be wrapped.

---

**Phase 1: Implementing Core Paper Concepts - `log(g)` Scaling, Anisotropy, Resonance (Agent: Implement these features thoroughly)**

**I. Advanced `log(g)`-Inspired Geometric Scaling & Dynamic Geometry**

*   **Concept:** Geometries adapt based on complexity, potentially dynamically.

*   **A. Principled Initialization (Mandatory refinement of previous outline):**
    1.  **TODO (`FullyHyperbolicWuBuNestingModel.__init__`):**
        *   Define `g_W_level_config` based on a richer set of factors: `level_idx`, `num_total_levels_in_stack`, `hyperbolic_dims[level_idx]`, and `input_tangent_dim` to the whole stack. This becomes part of the per-level WuBu config passed down.
        *   This `g_W_level_config` (a small dict or sub-config object) will be passed to `HyperbolicWuBuNestingLevel`.
    2.  **TODO (`HyperbolicWuBuNestingLevel.__init__`):**
        *   Accept `g_W_level_config`.
        *   The influence of `log(g_W_level_config.complexity_score + EPS)` on initial curvature and scale must be non-optional and configurable via factors (e.g., `curvature_log_g_factor`, `scale_log_g_factor` in the WuBu level config).
        *   The `PHI`-based influence (`phi_influence_curvature`) becomes a *secondary, optional* modulation on top of the `log(g)` scaling.
    3.  **TODO (Argparse/Config):** Add parameters to the new WuBu config classes for `curvature_log_g_factor`, `scale_log_g_factor`, `spread_log_g_factor`.

*   **B. Dynamic Curvature/Scale Modulation (New, Ambitious):**
    1.  **TODO (`HyperbolicWuBuNestingLevel`):**
        *   Instead of `log_curvature_unconstrained` being a single `nn.Parameter`, make it the output of a tiny MLP (e.g., 1-2 layers, small hidden dim) whose input is derived from `g_W_level_config` and potentially a global "epoch/step embedding" (a learnable embedding indexed by current epoch or a function of global step). This MLP will be part of `HyperbolicWuBuNestingLevel`.
        *   `self.curvature_modulator_mlp = nn.Sequential(...)`
        *   `self.base_log_curvature_unconstrained = nn.Parameter(...)` (represents the core learnable base)
        *   In `get_current_curvature_scalar()`: `modulation = self.curvature_modulator_mlp(derived_g_input); effective_unconstrained = self.base_log_curvature_unconstrained + modulation`. Then apply softplus.
        *   Similar mechanism for `scale` and `spread`.
    2.  **TODO (Argparse/Config):** Add flags to enable/disable this dynamic modulation and configure the modulator MLP (e.g., `dynamic_curvature_mlp_layers`, `dynamic_scale_mlp_hidden_dim`).
    3.  **TODO (Stability):** This is highly experimental. Start with very small MLPs and low learning rates for their parameters. Include options to heavily regularize the output of these modulator MLPs or clamp their additive effect.

**II. Sophisticated Anisotropic Processing & Resonant Nesting**

*   **Concept:** Levels develop specialized, directional processing pathways.

*   **A. Structured Anisotropy in `HyperbolicInterLevelTransform` (Mandatory upgrade):**
    1.  **TODO (`HyperbolicInterLevelTransform.__init__`):**
        *   Implement **Learnable Axis-Specific Scaling/Mapping:** If `in_dim` allows (e.g., `in_dim % N_ANISO_BLOCKS == 0`), partition the `in_dim` into `N_ANISO_BLOCKS` (e.g., 4 or 8, configurable).
        *   Apply *separate* small linear transformations or learnable diagonal scalers to each block of the tangent vector *before* the main `non_rotational_map`.
        *   `self.block_processors = nn.ModuleList([nn.Linear(block_dim, block_dim) for _ in range(N_ANISO_BLOCKS)])`
        *   The `non_rotational_map` then processes the concatenated outputs of these block processors.
    2.  **TODO (`HyperbolicInterLevelTransform`):** Implement `phi_influence_rotation_init` for `in_dim=3` (axis-angle) and `in_dim > 4` (e.g., block-wise quaternion rotations if `in_dim` is a multiple of 4, or learnable Givens rotations for general `n_dim`). For block-wise, the `rot_axis_param` and `rot_angle_unconstrained` would be per-block.
    3.  **TODO (Argparse/Config):** `num_aniso_blocks_transform`, `transform_block_processor_type` (`linear`, `diag_scale`).

*   **B. Enhanced Resonance in `HyperbolicWuBuNestingLevel` (Mandatory upgrade):**
    1.  **TODO (`HyperbolicWuBuNestingLevel.tangent_combiner`):**
        *   Make SwiGLU (or similar gated activation like GEGLU) the default activation for MLPs within `tangent_combiner` and `tangent_flow_module`.
        *   Implement an optional **Feature-wise Gating Mechanism:** After combining inputs (`tan_main_component`, etc.) and before the main MLP layers of `tangent_combiner`, add a learnable gate:
            *   `gate_controller = nn.Linear(self.comb_in_dim, self.comb_in_dim)`
            *   `gates = torch.sigmoid(gate_controller(combined_tangent_features))`
            *   `gated_features = combined_tangent_features * gates`
            *   Pass `gated_features` to the subsequent layers.
    2.  **TODO (`HyperbolicWuBuNestingLevel.tangent_flow_module`):**
        *   Ensure the `tangent_flow_module` (if `use_flow=True`) also uses gated activations.
        *   Consider making `tangent_flow_scale` a *learnable parameter* (with constraints, e.g., softplus) instead of a fixed config value, potentially modulated by `g_W_level_config`.
    3.  **TODO (Argparse/Config):** `use_swiglu_activations`, `use_featurewise_gating_combiner`, `learnable_tangent_flow_scale`.

*   **C. Level Descriptors & Boundary Manifolds as Active Geometric Probes:**
    1.  **TODO (`HyperbolicWuBuNestingLevel`):**
        *   The `level_descriptor_param`'s initialization scale (`ld_init_scale`) must also be influenced by the `log(g)` scaling.
        *   **Dynamic Level Descriptors (Advanced):** Similar to dynamic curvature, the `level_descriptor_param` could be modulated by a small MLP conditioned on `g_W_level_config` and global step/epoch. This allows the "preferred direction" to evolve.
    2.  **TODO (`BoundaryManifoldHyperbolic`):**
        *   The initialization of `hyperbolic_points_params` should also be influenced by `log(g)` scaling passed from the parent `HyperbolicWuBuNestingLevel`.
        *   **Dynamic Boundary Points (Very Advanced):** Similar to dynamic L.D., boundary points could be modulated. Defer unless critical.

---

**Phase 2: Evolution of "Adaptive Strain Engineering" - Meta-Control Supremacy**

*   **Concept:** Meta-controllers become more deeply integrated and intelligent.

*   **A. `HAKMEMQController` Overhaul:**
    1.  **TODO (State Representation):**
        *   Incorporate direct WuBu geometric states into Q-controller state: e.g., binned values of average `c_i`, `s_i`, `sigma_i` across WuBu levels relevant to the optimizer (G or D).
        *   Add bins for gradient norm statistics (e.g., from `optimizer.grad_stats`).
        *   Include short-term variance/stability of key loss components, not just trends.
    2.  **TODO (Action Space):**
        *   **Direct Lambda Modulation (for G's Q-controller):** Add actions to directly (slightly) scale `lambda_recon_heuristic_factor` and `lambda_kl_heuristic_factor` within predefined bounds. This allows the G Q-controller to fine-tune its own loss component weights beyond the LKL Q-controller's broader adjustments.
        *   **(Experimental) Curvature/Scale Nudging Actions:** Actions that suggest a small, temporary nudge to the *base* unconstrained parameters for curvature/scale in relevant WuBu stacks if geometries seem stuck in bad regions. (Requires careful safety bounds).
    3.  **TODO (Reward Shaping):**
        *   Incorporate penalties for numerical instability if associated WuBu levels report issues (e.g., if `get_current_curvature_scalar` had to clamp excessively).
        *   Add rewards for achieving "desirable" geometric configurations if metrics for anisotropy/resonance are developed (see Phase 3.A).
        *   Penalize Q-controller actions that lead to optimizer gradient stats showing many non-finite gradients.
    4.  **TODO (Probation & Reset Logic):**
        *   Make probation more nuanced. E.g., if reset is due to "stuck Q-values," probation might be shorter but with a higher initial epsilon.
        *   `force_exploration_boost` could have different "flavors" (e.g., short intense boost vs. longer moderate boost).

*   **B. `HybridTrainer` Heuristics Become Sophisticated Arbiters:**
    1.  **TODO (`_evaluate_training_state_and_apply_heuristics`):**
        *   **Heuristic Chaining/Prioritization:** Define explicit priorities. E.g., D-switching check is highest. If D switches, other heuristics might be temporarily suppressed or reset.
        *   **Factor Accumulation:** Allow multiple active heuristics to *cumulatively* affect factors like `heuristic_override_lambda_recon_factor` (e.g., `factor = base * boost1 * boost2`, within limits).
        *   **Heuristic Cooldowns:** Each specific heuristic intervention (e.g., VAE feature match boost) should have its own cooldown period after being deactivated to prevent rapid oscillation.
        *   **"Sanity Check" for WuBu Geometry:** Periodically, get `c_i, s_i, sigma_i` from all WuBu levels. If any are consistently at extreme clamped values for many steps, log a warning and potentially trigger a Q-controller reset for the associated optimizer, or even a very gentle nudge if "Curvature/Scale Nudging Actions" are implemented.
    2.  **TODO (`_check_and_perform_disc_switch`):**
        *   Integrate Q-controller health more deeply as primary signals for D-switch decisions. If `active_d_q` reports very poor reward history and high probation counts, this strongly argues for a switch, even if raw losses are ambiguous.
        *   Consider the "cost" of switching. If both Ds have struggling Q-controllers, switching might not help immediately.
    3.  **TODO (Meta-Learning Scaffolding - Foundational Step):**
        *   Refactor heuristic application: `_evaluate_training_state_and_apply_heuristics` computes a "heuristic action vector" (e.g., binary flags for VAE match, G easy win penalty; scalar values for lambda factors).
        *   Initially, these actions are applied by rule. Later, this vector could be the output of a meta-RL agent.
        *   Log this "heuristic action vector" systematically.

---

**Phase 3: Architectural Innovations & Interpretability**

*   **Concept:** Introduce novel structural elements within WuBu and tools to understand them.

*   **A. Systematic Interpretability Probes & Logging:**
    1.  **TODO (`HyperbolicWuBuNestingLevel` & `HyperbolicInterLevelTransform`):**
        *   Ensure these modules can, via a flag, return a dictionary of internal geometric states: current `c, s, sigma`, norms of level descriptors, singular values of key linear maps (if `compute_svd_for_interpretability=True`), effective rotation angles/axes.
    2.  **TODO (`HybridTrainer`):**
        *   Collect these detailed geometric states from all WuBu levels in `model` and `active_discriminator` periodically (e.g., every `log_interval` or a new `interpretability_log_interval`).
        *   Log aggregated statistics (mean, std, min, max) of these parameters across levels and stacks to WandB/TensorBoard.
        *   For low-dim levels (dim 2-3), implement actual Poincare disk/ball projection of a few sample points passing through and log as images (this is hard but high value).
    3.  **TODO (Gradient Monitoring):** Log gradient norms for distinct parts of WuBu: e.g., grads for curvature params, scale params, rotation params, main mapping params.

*   **B. (Experimental) Structured Tangent Space Rotations:**
    1.  **TODO (`HyperbolicInterLevelTransform`):**
        *   If `use_rotation_in_transform` is true and `in_dim` is a multiple of `rotation_block_dim` (e.g., 4 for quaternions, 2 for SO(2) blocks, configurable), implement block-wise rotations.
        *   Each block would have its own `rot_axis_param`/`rot_angle_unconstrained` (for 3D/4D blocks) or just `rot_angle_unconstrained_2d` (for 2D blocks).
        *   The `phi_influence_rotation_init` would apply per-block, possibly with different PHI powers for different blocks to encourage diverse rotational dynamics.
    2.  **TODO (Argparse/Config):** `rotation_block_dim`, enable/disable block-wise rotations.

---

**Phase 4: Robustness, Finalization, and Documentation**

*   **TODO:** Complete all unit tests.
*   **TODO:** Perform end-to-end integration tests with small datasets for all major new features (dynamic geometry, advanced anisotropy/resonance, structured rotations).
*   **TODO:** Systematically review all clamping values (`TAN_VEC_CLAMP_VAL`, `MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL`, `EPS`) and ensure they are appropriate for `float16/bfloat16` if AMP is used. Use `torch.finfo` where applicable for dynamic bounds.
*   **TODO:** Profile the code, especially new geometric computations and Q-controller updates, to identify bottlenecks.
*   **TODO:** **Comprehensive Documentation Update:**
    *   Update all class and method docstrings.
    *   Create a `CONFIG_GUIDE.md` explaining all new configuration classes and their parameters.
    *   Update `README.md` to reflect the new capabilities and the "Total Strategy" evolution.
    *   Reference the "WuBu Nesting & Spatio-Temporal Dynamics... v5.19.25" paper heavily in code comments and documentation to link implementation to theory.

---

**Agent's Execution Protocol:**

1.  **Prioritize Foundational Changes (Phase 0):** Config overhaul, enhanced logging, and testing setup are critical before major feature work.
2.  **Iterate Within Phases:** For Phase 1, implement `log(g)` initialization thoroughly first. Then tackle dynamic geometry. Then move to anisotropy, then resonance. Test each sub-feature.
3.  **Meta-Control Follows Core Enhancements:** Phase 2 (Q-controller/heuristics) should build upon the enhanced WuBu internals from Phase 1. The Q-controllers need new information to sense and new levers to pull.
4.  **Interpretability is Parallel:** Develop interpretability tools (Phase 3.A) alongside the features they are meant to inspect.
5.  **Experimental Features Last:** Structured tangent rotations (Phase 3.B) and the most ambitious dynamic geometry aspects are high risk/high reward; tackle them when the core is stable.
6.  **Constant Testing and Validation:** Use the new unit tests. Run short training jobs frequently on a small, representative dataset to catch issues early.
7.  **Embrace the Paper:** Let the scientific analogies guide your design choices. If unsure how to implement a "resonant" feature, re-read that section of the paper for inspiration.

This is a monumental task, Agent. The goal is to create a truly state-of-the-art, adaptive geometric deep learning framework. Proceed with diligence and precision.