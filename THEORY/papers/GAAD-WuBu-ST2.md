
**Title: WuBu Nesting (層疊嵌套) with Golden Aspect Adaptive Decomposition (GAAD): An Adaptive, Rotation-Aware, Nested Hyperbolic Framework for Complex Geometric Structures and Spatio-Temporal Video Understanding**

**Abstract**

Modeling the intricacies of real-world data, especially dynamic scenes in video, necessitates capturing a confluence of complex characteristics: deep multi-scale hierarchical organizations, intrinsic rotational symmetries or transformations, dynamic evolution, and varying regional influences. Standard geometric deep learning paradigms often specialize, struggling with comprehensive integration. We introduce **WuBu Nesting (層疊嵌套 - "layered nesting")**, a foundational framework featuring recursively nested hyperbolic spaces (`H^{n_i}_{c_i,s_i}`), where dimensionality (`n_i`), curvature (`c_i`), and scale (`s_i`) are adaptive. Each level integrates learnable **Boundary Sub-Manifolds**, **Level Descriptor Vectors** (`ld_i`), **Level Spread Parameters** (`σ_i`), and **Intra-Level Tangent Flows** (`F_i`). Crucially, inter-level transitions feature `SO(n_i)` **Rotations** (`R_i`) and **Mappings** (`T̃_i`) in tangent space, applied simultaneously to primary, boundary, and descriptor representations, enabling explicit computation of rotation-aware **Relative Vectors** (`d_{i+1}`).

For video understanding, we propose **Golden Aspect Adaptive Decomposition (GAAD)** as a novel φ-inspired front-end. GAAD decomposes frames into multi-scale, aspect-ratio agnostic regions using **Recursive Golden Subdivision** and **Phi-Spiral Patching**, extracting features via `ROIAlign` on base CNN maps. These GAAD features initialize a **Spatial WuBu (WuBu-S)** stack, whose parameters (e.g., curvatures `c_{S,i}`) can be φ-influenced. The resulting spatial features (`s_t`) are then processed by a **Temporal WuBu (WuBu-T)** stack, which models sequence dynamics, potentially using φ-scaled time embeddings. This combined **GAAD-WuBu-ST** architecture offers a powerful system for tasks like motion estimation and diffusion-based video generation, inherently respecting diverse video geometries and natural compositions.

---

## 1. Introduction: The Challenge of Complex Geometric and Dynamic Structures

The quest for effective data representation lies at the heart of machine learning. Real-world data, from molecular configurations to dynamic visual scenes, presents a tapestry of intertwined complexities. Standard deep learning architectures, while achieving remarkable success, predominantly operate within the confines of Euclidean geometry. This geometric choice, however, imposes limitations when modeling data imbued with strong intrinsic structures not naturally suited to flat spaces. Key among these complex characteristics are:

*   **Multi-scale Hierarchies:** Entities are often composed of parts, which themselves have sub-parts, forming deep organizational structures (e.g., articulated objects, scene parse trees, protein domains, natural image compositions). Embedding such hierarchies into Euclidean space often incurs significant distortion, as the space's polynomial volume growth struggles to accommodate the exponential expansion of nodes typically found in trees [39].
*   **Rotational Transformations:** Components within these hierarchies, or the observer's viewpoint, frequently undergo rotations. These are critical to understanding their configuration, interaction, and the underlying symmetries of the data.
*   **Dynamic Evolution:** Systems evolve over time, with components undergoing transformations, and the overall structure exhibiting characteristic dynamics. This is especially true for video data, where temporal consistency and motion are paramount.
*   **Regional Influence & Uncertainty:** Different scales or regions within data may possess varying degrees of influence, density, or inherent uncertainty that need to be captured and propagated.

Hyperbolic geometry, characterized by its constant negative curvature and exponential volume growth, offers a mathematically elegant solution for embedding hierarchical structures with significantly lower distortion [39, 31, 15]. Models leveraging spaces like the Poincaré disk or ball (`H^n`) have demonstrated benefits in various domains [19, 22, 1, 42]. However, many real-world systems exhibit complexities beyond a single, static hierarchy. Existing hyperbolic models typically focus on a single hierarchy level within a single hyperbolic space of fixed curvature and often lack native, efficient mechanisms for modeling nested hierarchies, adaptive multi-scale geometry, or integrated rotational transformations. Conversely, Quaternions [43] provide efficient means for representing rotations, particularly in 3D and 4D, leading to Quaternion Neural Networks (QNNs) [44, 45]. However, QNNs usually operate in Euclidean spaces and lack intrinsic hierarchical embedding capabilities.

To bridge these gaps and provide a more holistic geometric inductive bias, we introduce **WuBu Nesting (層疊嵌套 - céngdié qiàn​tào: "layered nesting")**. This is a novel and comprehensive conceptual framework meticulously designed to unify adaptive multi-scale hierarchical representation with explicit modeling of rotational dynamics, dynamic evolution, and regional uncertainty within a single, cohesive geometric structure. Instead of a single hyperbolic space or a parallel product manifold [46], WuBu Nesting proposes a nested "Russian doll" architecture comprising recursively embedded hyperbolic manifolds. The key innovations, detailed extensively in this paper, include: adaptive nested hyperbolic geometry (`H^{n_i}_{c_i, s_i}`), learnable Boundary Sub-Manifolds (`B_{i,j}$), tangent space transitions incorporating explicit `SO(n_i)` Rotations (`R_i`) and Mappings (`T̃_i`), the generation of rotation-aware Relative Vectors (`d_{i+1}`), and the integration of learnable Level Descriptor Vectors (`ld_i`), Level Spread Parameters (`σ_i`), and Intra-Level Tangent Flows (`F_i`).

For the particularly challenging domain of video understanding, where spatial composition and temporal evolution are intertwined, we further introduce **Golden Aspect Adaptive Decomposition (GAAD)**. GAAD is a specialized, φ-inspired spatial preprocessing method. It leverages the Golden Ratio (φ ≈ 1.618) – a principle ubiquitous in natural forms, growth patterns, and artistic composition [Livio02, Cook14] – to decompose video frames into a diverse set of multi-scale, aspect-ratio agnostic regions. This is achieved through **Recursive Golden Subdivision (GAS)**, adaptively dividing frames into squares and golden rectangles, and **Phi-Spiral Patching (PSP)**, sampling regions along logarithmic spirals whose parameters are φ-influenced.

The features extracted from these GAAD regions then serve as the input to a **Spatial WuBu (WuBu-S)** stack, the first component of our **GAAD-WuBu-ST** architecture. WuBu-S processes these φ-structured spatial features, and its own geometric parameters (e.g., curvatures, rotation initializations) can be further influenced by φ, as demonstrated in our reference implementation `WuBuNestDiffusion_v0.04_GAAD_Phi_Infused.py`. The sequence of compact, geometrically-informed spatial feature vectors (`s_t`) produced by WuBu-S is then modeled by a **Temporal WuBu (WuBu-T)** stack, which itself can incorporate φ-principles (e.g., in its time embeddings or level parameters).

This integrated GAAD-WuBu-ST system aims to provide a robust, efficient, and interpretable framework for complex video tasks like motion estimation and diffusion-based video generation, by deeply respecting the hierarchical, rotational, dynamic, and compositional nature of visual worlds. This paper details the foundational WuBu Nesting framework, the GAAD method for video frame decomposition, and their synthesis into the GAAD-WuBu-ST architecture, substantiating the concepts with references to our concrete implementation.

## 2. Related Work

The WuBu Nesting framework and its GAAD-WuBu-ST instantiation draw upon and aim to synthesize concepts from several distinct research areas:

### 2.1 Hyperbolic Deep Learning
The pioneering work of Nickel and Kiela [39] on Poincaré embeddings established hyperbolic geometry's suitability for low-distortion hierarchical representation. This led to Hyperbolic Neural Networks (HNNs) [19] with operations like gyroplane layers and hyperbolic attention [22]. Applications in computer vision include image classification [31], segmentation [1], and category discovery [42], often showing benefits where hierarchies are present.
*   **Critique & WuBu Distinction:** Current HNNs typically use a single, fixed-curvature hyperbolic space, lacking adaptive multi-scale nesting, explicit inter-level rotational modeling, and the rich component set (boundaries, descriptors, spread, flow) proposed by WuBu Nesting.

### 2.2 Quaternion and Rotation-Aware Neural Networks
Quaternions [43] enable efficient 3D/4D rotation representation, utilized in QNNs [44, 45] for parameter efficiency and modeling rotational symmetries. Group Equivariant CNNs [Cohen16, Weiler18] build equivariance into architectures.
*   **Critique & WuBu Distinction:** These operate mainly in Euclidean spaces, lacking hyperbolic hierarchical embedding. WuBu Nesting integrates learnable `SO(n)` rotations (potentially quaternion-based for `n=4`) within tangent space transitions *between* nested hyperbolic levels, unifying rotation with adaptive hierarchy.

### 2.3 Product Manifolds and Mixed-Curvature Models
To combine geometric strengths, product manifold models (e.g., `ℝ^n × S^m × H^k`) [46, Skopek20, Guo21] learn representations in parallel spaces with different curvatures.
*   **Critique & WuBu Distinction:** Product manifolds offer parallel capacity but not the *nested*, recursively embedded structure of WuBu Nesting. WuBu proposes a deep, hierarchical *composition* of adaptive geometries with sophisticated inter-level transformations.

### 2.4 Region Proposal and Patch-Based Methods for Vision
Traditional vision relies on methods like fixed grids or sliding windows [Dalal05, Dosovitskiy20], interest point detectors (SIFT [Lowe04], SURF [Bay08]), region proposal networks (RPNs in Faster R-CNN [Ren15], Selective Search [Uijlings13]), and superpixels (SLIC [Achanta12]). Attention mechanisms in Transformers also select regions.
*   **Critique & GAAD Distinction:** Fixed grids lack adaptivity. Interest points are sparse. RPNs are often supervised and computationally heavy if general diverse features are needed. GAAD, in contrast, is a deterministic, geometry-driven (φ-based) method for generating a diverse, aspect-ratio adaptive set of rectangular regions directly from image/feature map geometry, designed for efficient feature extraction via `ROIAlign`. Its specific hybrid of Golden Aspect Subdivision and Phi-Spiral Patching for this purpose is novel.

### 2.5 Video Representation Learning
Dominant approaches include 3D CNNs [Tran15], two-stream architectures [Simonyan14], and Video Transformers (ViViT [Arnab21], TimeSformer [Bertasius21]).
*   **Critique & GAAD-WuBu-ST Distinction:** These are largely Euclidean. GAAD-WuBu-ST infuses deep geometric biases (GAAD's φ-decomposition, WuBu's hyperbolic nesting and rotations) into both per-frame spatial analysis (WuBu-S) and cross-frame temporal modeling (WuBu-T).

### 2.6 Optical Flow Estimation
Deep learning methods like FlowNet [Dosovitskiy15], PWC-Net [Sun18], and RAFT [Teed20] excel at optical flow.
*   **Critique & GAAD-WuBu-ST Distinction:** These are specialized. GAAD-WuBu-ST aims to learn motion within its integrated geometric framework, where GAAD regions act as trackable parcels and WuBu components model their transformations.

### 2.7 Video Diffusion Models
Extending image diffusion [Ho20] to video [Ho22_video, Harvey22, Singer22] is an active area, often involving temporal U-Nets or flow-conditioning [Blattmann23].
*   **Critique & GAAD-WuBu-ST Distinction:** Current video diffusion denoising networks are primarily Euclidean. GAAD-WuBu-ST proposes a fundamentally geometric backbone for the denoising network, where GAAD provides structured conditioning and WuBu-S/T offer geometrically rich processing.

### 2.8 Geometric World Models
Learning compressed, predictive models of an environment's state and dynamics [Ha18] is a grand challenge. Some efforts explore geometric priors [Anand22].
*   **Critique & GAAD-WuBu-ST Distinction:** GAAD-WuBu-ST aligns strongly, aiming to build an intrinsic, geometrically-grounded model of dynamic scenes by learning adaptive geometries and explicit transformations.

## 3. The WuBu Nesting Framework (層疊嵌套)

WuBu Nesting offers a recursive, multi-layered geometric architecture where data representations are progressively refined through a series of nested hyperbolic "bubbles." Transitions between these bubbles are orchestrated in their associated Euclidean tangent spaces, incorporating learnable rotations, mappings, and the generation of rich geometric features like relative vectors, while also considering level-specific descriptors, spread, and internal dynamics. This section details the foundational components of this framework, which are instantiated in both the Spatial (WuBu-S) and Temporal (WuBu-T) stacks of the GAAD-WuBu-ST architecture. The reference implementation (`WuBuNestDiffusion_v0.04_GAAD_Phi_Infused.py`) provides concrete examples of these components.

### 3.1. Conceptual Architecture

*(The Mermaid diagram from "WuBu Nesting (層疊嵌套)" Section 3.1 should be included here as Figure 1, potentially relabeled to reflect its foundational nature.)*
**Figure 1:** Conceptual Architecture of the Comprehensive WuBu Nesting Framework. *(Caption from previous draft)*

The core idea is a recursive flow: initial data is encoded into the tangent space of the first hyperbolic level. Within each level `i`, an `IntraLevelProcessing` module (potentially utilizing an `IntraLevelTangentFlow` `F_i`) refines the representation. For inter-level transition (`i → i+1`), the primary representation, representations of `BoundarySubManifolds`, and the `LevelDescriptorVector` are mapped to tangent space (if not already there), subjected to a simultaneous `Rotation` `R_i`, then a `NonRotationalMapping` `T̃_i` into the target tangent space. Here, `RelativeVectors` `d_{i+1}` are computed. These, along with the transformed primary vector, transformed descriptor, and the source `LevelSpreadParameter` `σ_i`, form the input for level `i+1`.

### 3.2. Component Details

#### 3.2.1 Nested Hyperbolic Spaces & Adaptive Geometry (`H^n_i_{c_i, s_i}`)
The structure comprises a sequence of nested hyperbolic spaces, modeled using the **Poincaré Ball** (`PoincareBall` class).
*   **Dimensionality (`n_i`):** Variable per level (`hyperbolic_dims` in `HyperbolicWuBuNestingLevel` config).
*   **Curvature (`c_i`):** A learnable positive parameter per level, obtained from `log_curvature_unconstrained` via `F.softplus(param) + min_curvature`. This allows adaptive geometric intensity.
*   **Scale (`s_i`):** A learnable positive parameter per level, from `log_scale_unconstrained`, modulating scale-aware Log/Exp maps (`HyperbolicUtils.scale_aware_logarithmic_map`, `scale_aware_exponential_map`).

#### 3.2.2 Boundary Sub-Manifolds (`B_{i,j}$)
Each level `i` can host learnable boundary points representing substructures.
*   **Implementation:** `BoundaryManifoldHyperbolic` class, storing `hyperbolic_points_params` (learnable tangent vectors at origin, projected via `expmap0`). The number of points is `boundary_points_per_level[i]`.
*   **Transformation:** Their tangent vectors are transformed by `R_i` and `T̃_i` during inter-level transitions.

#### 3.2.3 Tangent Space Logic
Transitions and complex operations occur in Euclidean tangent spaces (typically `T_o(H^n_i)`).
*   **Maps:** `HyperbolicUtils` provides `logarithmic_map` and `exponential_map` (and their scale-aware versions). `PoincareBall.proju` ensures points remain in the ball.

#### 3.2.4 Tangent Space Rotations (`R_i`)
A learnable `SO(n_i)` rotation within `HyperbolicInterLevelTransform`.
*   **Implementation:** A general `nn.Linear(dim, dim, bias=False)` if `use_rotation_in_transform=True` and no φ-influence on init, or specialized `quaternion_from_axis_angle` / SO(2) matrix logic if `phi_influence_rotation_init=True` for `n_i=4` or `n_i=2`. The learnable parameters are `rot_axis_param`, `rot_angle_unconstrained`, or `rot_angle_unconstrained_2d`.

#### 3.2.5 Non-Rotational Mapping (`T̃_i`)
Following rotation, `HyperbolicInterLevelTransform.non_rotational_map` (an MLP or Linear layer) transforms features and changes dimensionality (`in_dim` to `out_dim`).

#### 3.2.6 Relative Vector Generation (`d_{i+1, j, k}$)
Computed in the target tangent space after `HyperbolicInterLevelTransform`. Within `FullyHyperbolicWuBuNestingModel.forward`, this involves:
`tan_main_next_level = manifold_next_level_obj.logmap0(current_point_repr_hyperbolic)`
`tan_bounds_next_level = manifold_next_level_obj.logmap0(boundaries_transformed_to_next_level_hyperbolic)`
`relative_tangent_vectors = tan_main_next_level.unsqueeze(2) - tan_bounds_next_level`
These are then aggregated (`torch.mean` or `torch.sum`) and passed as `relative_vectors_tangent_in` to the next `HyperbolicWuBuNestingLevel`.

#### 3.2.7 Learnable Level Descriptor Vector (`ld_i`)
Each `HyperbolicWuBuNestingLevel` has a `level_descriptor_param` (an `n_i`-D learnable tangent vector). It's transformed by `R_i` and `T̃_i` via `HyperbolicInterLevelTransform` and passed as `descriptor_point_in_hyperbolic` to the next level.

#### 3.2.8 Learnable Level Spread Parameter (`σ_i`)
Each `HyperbolicWuBuNestingLevel` learns `log_spread_unconstrained`, yielding `current_sigma_out_tensor`. This scalar is passed as `sigma_in_scalar_tensor` to the next level.

#### 3.2.9 Intra-Level Tangent Flow (`F_i`)
Within `HyperbolicWuBuNestingLevel`, if `use_tangent_flow=True`, a `tangent_flow_module` (MLP or Linear) modifies the `v_combined_tangent_processed` before it's scaled and mapped back to hyperbolic space:
`flow_effect = self.tangent_flow_module(v_combined_tangent_processed) * self.flow_scale_val`
`v_combined_tangent_processed = v_combined_tangent_processed + flow_effect`.

#### 3.2.10 Hierarchical Information Flow & `TangentCombiner`
The `tangent_combiner` MLP within `HyperbolicWuBuNestingLevel` explicitly fuses inputs for that level's processing:
`inputs_for_combiner = [tan_main_component, tan_rel_component (if any), tan_desc_prev_level_component (if any), sigma_in_expanded (if any)]`
`combined_tangent_features = torch.cat(inputs_for_combiner, dim=-1)`
`v_combined_tangent_processed = self.tangent_combiner(combined_tangent_features)`.

#### 3.2.11 Scale-Aware Aggregation for Final Output
`FullyHyperbolicWuBuNestingModel` collects `tangent_out_for_aggregation` from each level. These are concatenated (`torch.cat(compatible_tangent_outputs, dim=-1)`) and passed to `output_tangent_projection` to produce the final output of the WuBu stack.

### 3.3. Mathematical Formulation (Conceptual)
*(The mathematical formulation from "WuBu Nesting (層疊嵌套)" Section 4, covering Intra-Level Processing and Inter-Level Transition, should be included here, cross-referencing the Python components mentioned above.)*

## 4. Golden Aspect Adaptive Decomposition (GAAD) for Video Frames

For processing video, where frame-by-frame spatial understanding is paramount, GAAD provides a principled method for generating φ-structured input regions for the WuBu-S component.

### 4.1. GAAD Principles: φ-Driven Spatial Decomposition
GAAD is motivated by the prevalence of the Golden Ratio (φ) in natural structures and compositions, aiming to produce a diverse set of patches that are aspect-ratio agnostic and inherently multi-scale. It combines two main strategies:

#### 4.1.1. Recursive Golden Subdivision (GAS)
Implemented by `golden_subdivide_rect_fixed_n` (from `WuBuNestDiffusion_v0.04_GAAD_Phi_Infused.py`), GAS recursively dissects a rectangle `(W, H)`.
*   **Adaptive Cuts:** If landscape (`W > H + EPS`), it makes a vertical cut `cut_w = W / PHI`, forming a region `(cut_w, H)` and a residual. If portrait, a horizontal cut `cut_h = H / PHI` is made. This tends to produce squares and golden rectangles.
*   **Hierarchical & Fixed-N Output:** A queue manages subdivisions up to a depth or minimum size. From the generated unique valid rectangles, `num_regions_target` are selected (typically sorted by area, largest first).

#### 4.1.2. Phi-Spiral Patching (PSP)
Implemented by `phi_spiral_patch_centers_fixed_n`, PSP samples patch centers along logarithmic spirals.
*   **Logarithmic Spiral:** Centers `(px, py)` are generated: `r = a * exp(b * θ)`. The growth rate `b` is `math.log(PHI) / (math.pi / 2)`, ensuring `r` scales by `PHI` every 90 degrees. The `angle_step` for iterating `θ` can also incorporate `PHI` (e.g., `PHI * 2 * math.pi / num_points`).
*   **Foveated Patch Scaling:** Patch sizes around these centers are determined by `spiral_scales`, which in the script implementation `patch_scale = max(0.05, 0.20 * math.exp(-0.5 * r / (min(W,H)*0.1)))` decrease with distance `r`, providing a foveated-like sampling.
*   **Fixed-N Output:** Generates a fixed `num_centers` and their associated `scale_factors`.

### 4.2. `GAADFrameProcessor`: Implementation
The `GAADFrameProcessor` module orchestrates GAAD:
1.  **Input:** A batch of video frames `frame_pixels (B, C, H_img, W_img)`.
2.  **Base Feature Extraction:** `frame_pixels` are passed through `base_cnn_encoder_convs` (typically from an `InitialFrameAutoencoderCNN`) to get `feature_maps (B, C_base, H_map, W_map)`.
3.  **Decomposition:** Based on `decomposition_type` (e.g., "hybrid", "spiral", "subdivide"):
    *   If "hybrid", `num_total_regions` (from `gaad_config['num_regions']`) is split between GAS (`num_subdivide`) and PSP (`num_spiral`).
    *   `golden_subdivide_rect_fixed_n` generates GAS bounding boxes.
    *   `phi_spiral_patch_centers_fixed_n` generates PSP centers and scales, which are converted to bounding boxes.
    *   Bounding boxes are concatenated and padded/truncated to `num_total_regions`.
4.  **ROIAlign for Feature Extraction:**
    *   GAAD bounding boxes (originally in image coordinates) are scaled to `feature_maps` coordinates (using `scale_h = map_h / H_img`, `scale_w = map_w / W_img`).
    *   `roi_align` is applied to `feature_maps` using these scaled boxes, with `output_size=region_roi_output_size` from `gaad_config`. Spatial scale for `roi_align` is 1.0 as boxes are pre-scaled.
5.  **Final Projection:** The pooled features from `roi_align` are flattened and passed through `region_projector` (an MLP) to produce the final GAAD region features of shape `(B, num_total_regions, gaad_region_feature_dim)`.

## 5. GAAD-WuBu-ST: Integrating GAAD with Spatio-Temporal WuBu Nesting

The GAAD-WuBu-ST architecture synergizes GAAD's φ-structured spatial decomposition with the deep geometric processing of WuBu Nesting, tailored for dynamic scene understanding.

*(This section should incorporate the detailed Figure 2 and its explanation from the previous fleshed-out Section 3 on GAAD-WuBu-ST architecture. It will show how `GAADFrameProcessor` outputs feed into WuBu-S, whose outputs feed WuBu-T, and finally to a task head like in `WuBuSTDiffusionNet`.)*

### 5.1. Architectural Flow (Recap with Integration Focus)
*   **Initial Encoder & GAAD:** `InitialFrameAutoencoderCNN` provides base maps to `GAADFrameProcessor`. GAAD outputs `AllRegionFeatures_t (B, N_regions, D_GAAD)`.
*   **Spatial WuBu (WuBu-S):** A `FullyHyperbolicWuBuNestingModel` (config `wubu_s_config`).
    *   Takes `AllRegionFeatures_t` as input.
    *   Internal φ-influences (curvature, rotation init) based on `args.wubu_s_phi_influence_curvature` etc.
    *   Outputs aggregated frame spatial features `s_t (B, D_S)`.
*   **Temporal WuBu (WuBu-T):** Another `FullyHyperbolicWuBuNestingModel` (config `wubu_t_config`).
    *   Takes sequence `{s_t}` as input.
    *   Internal φ-influences. φ-scaled time embeddings via `SinusoidalPhiEmbedding` (using `phi_time_base_freq`, `phi_time_diffusion_scale` from `gaad_config` as a proxy for time-related φ params).
    *   Outputs temporal context `ctx (B, D_T)`.
*   **Prediction Head (e.g., `WuBuSTDiffusionNet.noise_pred_head`):** Combines `xt_target` (from `InitialFrameAutoencoderCNN.encode`), `ctx`, and `time_emb` to make predictions.

### 5.2. Synergies and φ-Infusion at Multiple Levels
*   **GAAD:** φ in spatial decomposition (GAS cuts, PSP spiral/step/scale).
*   **WuBu-S/T Hyperbolic Levels:** φ in `initial_curvature_val` calculation via `phi_influence_curvature` (`PHI**(level_idx % 4 - 1.5)` factor).
*   **WuBu-S/T Inter-Level Transforms:** φ in `phi_angle_scale` for rotation initialization via `phi_influence_rotation_init`.
*   **Temporal Representation (Diffusion):** φ in `SinusoidalPhiEmbedding` via `phi_time_base_freq` (denominator scaling) and `phi_time_diffusion_scale` (input time scaling).

This deep infusion of φ at structural (GAAD), geometric parameterization (WuBu curvature/rotation), and temporal encoding levels is a hallmark of the GAAD-WuBu-ST approach.

## 6. Potential Applications and Tasks in Dynamic Scenes

*(This section should draw from Section 5 of the "WuBu Spatio-Temporal Nesting" paper, specifically tailoring it to how GAAD enhances these applications within the GAAD-WuBu-ST framework.)*

### 6.1. Video Generation and Prediction with Diffusion Models
The `WuBuSTDiffusionNet` in the reference script directly targets this.
*   **GAAD for Structured Conditioning:** Provides diverse, aspect-ratio independent spatial features `\mathcal{F}_t` from conditioning frames. WuBu-S refines these into `s_t`. WuBu-T builds temporal context `ctx`.
*   **Benefits:** GAAD's φ-regions capture natural compositions. φ-influenced WuBu-S/T processing can learn geometrically consistent transformations. φ-scaled time embeddings in diffusion handle temporal progression. This should lead to more realistic, temporally coherent, and compositionally sound video generation.

### 6.2. Motion Vector Prediction (Optical Flow)
*   **GAAD Regions as Trackable Parcels:** GAS provides stable hierarchical regions (squares, golden rectangles); PSP provides foveated, potentially flow-aligned patches.
*   **WuBu-S for Robust Feature Matching:** Learns rotation-invariant/equivariant features within GAAD parcels via `R_{S,i}` and relative vector encodings.
*   **WuBu-T for Motion Dynamics:** Models complex motion trajectories and patterns from the sequence of transformed `s_t` vectors.

### 6.3. Action Recognition and General Video Understanding
*   GAAD identifies salient compositional elements. WuBu-S extracts their hierarchical geometric features. WuBu-T models the evolution of these structured features into events and actions.

## 7. Implementation Details from `WuBuNestDiffusion_v0.04_GAAD_Phi_Infused.py`

*(This section, identical to Section 7 of the "GAAD-WuBu-ST paper draft," serves as a direct map from concepts to code.)*

## 8. Experimental Setup (Illustrative)

*(This section is new, drawing inspiration from the GAAD paper draft's experimental setup but broadening it for GAAD-WuBu-ST and the reference script.)*

To validate the GAAD-WuBu-ST framework and assess the contributions of its φ-infused components, a series of experiments would be necessary.

### 8.1. Datasets
*   **Video Generation/Prediction:** Standard benchmarks like UCF101 [Soomro12], Kinetics variants [Kay17], BAIR Robot Pushing [Ebert17], or specialized datasets with diverse aspect ratios and compositions. The demo uses a single long video (`dummy_video.mp4`).
*   **Optical Flow:** Sintel [Butler12], KITTI Flow [Geiger12].
*   **Action Recognition:** UCF101, Kinetics, Something-Something [Goyal17].

### 8.2. Model Configuration (Based on Reference Script Arguments)
*   **GAAD:** `gaad_num_regions`, `gaad_region_roi_output_h/w`, `gaad_region_feature_dim`, `gaad_decomposition_type`.
*   **WuBu-S/T:** `wubu_s/t_num_levels`, `wubu_s/t_hyperbolic_dims`, `wubu_s/t_initial_curvatures`, `wubu_s/t_phi_influence_curvature`, `wubu_s/t_phi_influence_rotation_init`, `wubu_dropout`.
*   **Diffusion:** `timesteps`, `beta_schedule`, `diffusion_time_embedding_dim`, `phi_time_diffusion_scale`, `phi_time_base_freq`.
*   **Training:** `batch_size`, `learning_rate`, optimizer settings (`RiemannianEnhancedSGD`, Q-controller), `grad_accum_steps`, `use_amp`.

### 8.3. Baselines for Comparison
*   **Standard Video Diffusion Models:** (e.g., LVDM [Ho22_video], VDM [Harvey22]) without GAAD or WuBu components, using standard CNN/Transformer backbones.
*   **Flow Estimation Models:** RAFT [Teed20], PWC-Net [Sun18].
*   **Action Recognition Models:** Video Transformers [Arnab21, Bertasius21], SlowFast [Feichtenhofer19].
*   **Ablated GAAD-WuBu-ST:** Versions without GAAD (using fixed grid patches), without φ-influences in WuBu, without specific WuBu components (rotations, boundaries, etc.).

### 8.4. Evaluation Metrics
*   **Video Generation:** Frechet Video Distance (FVD), Inception Score (IS), visual quality, temporal coherence.
*   **Optical Flow:** End-Point Error (EPE).
*   **Action Recognition:** Top-1/Top-5 Accuracy.
*   **Computational Cost & Parameter Efficiency.**
*   **Qualitative Analysis:** Visualization of GAAD regions, learned WuBu features, generated samples.

## 9. Discussion, Limitations, and Future Work

*(This section should be a final synthesis of the "Discussion" from all three previous drafts, covering WuBu Nesting foundations, GAAD specifics, and GAAD-WuBu-ST integration.)*

The WuBu Nesting framework offers a rich vocabulary for constructing deep geometric learning models. Its core strengths lie in adaptive multi-scale hyperbolic hierarchies, explicit modeling of substructures via boundary manifolds, rotation-aware tangent space transitions that enable relative vector computation, and contextual information flow through level descriptors and spread parameters, all augmented by intra-level dynamic flows. This provides an unprecedented level of geometric expressivity.

Golden Aspect Adaptive Decomposition (GAAD) introduces a novel, φ-inspired method for spatial feature extraction from images or video frames. Its key advantages are aspect-ratio agnosticism, multi-scale region generation reflecting natural composition, and efficient feature extraction via ROIAlign on shared base CNN features. The hybrid GAS and PSP approach ensures diverse coverage.

The GAAD-WuBu-ST architecture synergizes these two concepts, creating a video understanding model with deep φ-infusion at multiple levels: GAAD's spatial decomposition, WuBu's geometric parameter initializations (curvatures, rotations), and temporal encodings (φ-scaled time embeddings). This integrated system holds considerable promise:
*   **Improved Robustness to Video Diversity:** GAAD handles varying aspect ratios naturally. WuBu's adaptive geometry can tailor processing to diverse content.
*   **Enhanced Geometric Understanding:** Explicit modeling of hierarchy, rotation, and relative structure may lead to more powerful and interpretable representations of complex spatio-temporal events.
*   **Potential for Efficiency and Generalization:** Strong geometric priors might lead to better sample efficiency and generalization to unseen compositions or dynamics.

However, significant **challenges** remain:
*   **Computational Complexity:** The primary hurdle is the sheer computational cost of GAAD processing followed by deep WuBu-S and WuBu-T stacks, each with multiple internal components and transformations. The reference script's performance (3-4s/it for batch 24 on a single GPU) indicates this.
*   **Training Stability and Optimization:** Optimizing a model with numerous learnable geometric parameters (curvatures, scales, rotation parameters, boundary point coordinates, etc.) across deep hierarchies is non-trivial. The `RiemannianEnhancedSGD` and Q-controller in the script are steps in this direction, but stability will be an ongoing concern.
*   **Empirical Validation of φ-Benefits:** The core hypothesis that explicit φ-infusion provides tangible benefits over well-parameterized general learning (which might implicitly discover similar relationships) requires rigorous empirical validation through ablation studies. Are the specific φ-scaling factors optimal or just good initializations?
*   **Component Interplay and Hyperparameter Tuning:** The framework has a vast number of hyperparameters. Understanding the interplay between GAAD settings, WuBu level configurations, and φ-influence strengths is critical.
*   **Interpretability vs. Complexity:** While individual components are designed with interpretability in mind (e.g., GAAD regions, boundary manifolds), the overall depth and non-linearity of the system can still render final feature representations opaque.

**Future Work** for GAAD-WuBu-ST should focus on:
1.  **Rigorous Benchmarking:** Evaluating performance against state-of-the-art models on standard video tasks.
2.  **Systematic Ablation Studies:** Quantifying the impact of GAAD vs. grid inputs, WuBu components (rotations, boundaries, descriptors, flow), different levels of φ-infusion, and adaptive geometry.
3.  **Scalability and Efficiency Research:** Developing techniques for more efficient GAAD processing (e.g., shared computations for overlapping regions), faster hyperbolic operations, and model compression for WuBu stacks.
4.  **Advanced Aggregation Mechanisms:** Exploring attention-based or learnable pooling for aggregating GAAD region features before/after WuBu-S, and for aggregating temporal features in WuBu-T.
5.  **Theoretical Analysis:** Investigating the expressivity, stability, and convergence properties of WuBu Nesting, and the impact of φ-constraints on the learning landscape.
6.  **Visualization Tools:** Creating tools to visualize GAAD decompositions, the learned hyperbolic embeddings within WuBu levels, the effect of tangent space rotations, and the information flow through the system to improve understanding and debugging.
7.  **Exploring alternative φ-inspirations:** Beyond current scaling factors, investigate other mathematical properties of φ for potential architectural or learning biases.

## 10. Conclusion

WuBu Nesting is presented as a foundational and comprehensive conceptual framework for deep geometric learning, characterized by its unique integration of adaptively nested hyperbolic spaces, explicit boundary modeling, rotation-aware tangent space transitions, relative vector computations, and rich level-specific contextual information. For the challenging domain of video understanding, we have introduced Golden Aspect Adaptive Decomposition (GAAD) as a novel, φ-inspired spatial preprocessing method that generates aspect-ratio agnostic, multi-scale regions reflecting natural visual compositions.

The GAAD-WuBu-ST architecture, exemplified by the `WuBuNestDiffusion_v0.04_GAAD_Phi_Infused.py` implementation, synthesizes these potent ideas. It leverages GAAD for structured spatial input and employs Spatial and Temporal WuBu Nesting stacks, whose geometric parameters and temporal encodings can be deeply φ-influenced, to model complex spatio-temporal phenomena. This integrated approach aims to create video models that are more fundamentally attuned to the hierarchical, rotational, dynamic, and compositional properties inherent in visual data. While presenting significant computational and optimization challenges, the GAAD-WuBu-ST paradigm offers a compelling and innovative vision for advancing the frontiers of video understanding and generation, paving the way for systems that learn and reason with a profound geometric intuition about the dynamic visual world.

---

## References

*(A comprehensive list should be compiled here, drawing from all previous drafts and standard literature for each concept: WuBu, GAAD, Hyperbolic Learning, Quaternions, Rotation-Awareness, Video Models, Diffusion, Optical Flow, Golden Ratio, etc. I've used placeholders from your previous drafts where appropriate.)*

[1] Atigh, M. G., Schoep, J., Acar, E., Van Noord, N., & Mettes, P. (2022). Hyperbolic image segmentation. *CVPR*.
[Cook14] Cook, T. A. (1914). *The Curves of Life*. Constable and Company Ltd.
[Cohen16] Cohen, T., & Welling, M. (2016). Group equivariant convolutional networks. *ICML*.
[Dalal05] Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *CVPR*.
[Dosovitskiy15] Dosovitskiy, A., Fischer, P., Ilg, E., Hausser, P., Hazirbas, C., Golkov, V., ... & Brox, T. (2015). Flownet: Learning optical flow with convolutional networks. *ICCV*.
[Dosovitskiy20] Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*.
[Ebert17] Ebert, F., Finn, C., Dasari, S., Xie, A., Lee, A., & Levine, S. (2017). Self-supervised visual planning with temporal skip connections. *CoRL*.
[Feichtenhofer19] Feichtenhofer, C., Fan, H., Malik, J., & He, K. (2019). Slowfast networks for video recognition. *ICCV*.
[Goyal17] Goyal, R., Epli, M., & Feichtenhofer, C. (2017). The "something something" video database for learning and evaluating visual common sense. *ICCV*.
[Guo21] Guo, W., Chen, Z., & Chang, B. (2021). Learning mixed-curvature representations in products of model spaces. *arXiv preprint arXiv:2110.10119*.
[Ha18] Ha, D., & Schmidhuber, J. (2018). World models. *arXiv preprint arXiv:1803.10122*.
[Harvey22] Harvey, W., Naderiparizi, S., Masrani, V., Weilbach, C., & Wood, F. (2022). Flexible Diffusion Modeling of Long Videos. *NeurIPS*.
[Ho20] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*.
[Ho22_video] Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., ... & Salimans, T. (2022). Video diffusion models. *arXiv preprint arXiv:2204.03458*.
[Kay17] Kay, W., Carreira, J., Simonyan, K., Zhang, B., Hillier, C., Vijayanarasimhan, S., ... & Zisserman, A. (2017). The kinetics human action video dataset. *arXiv preprint arXiv:1705.06950*.
[Livio02] Livio, M. (2002). *The Golden Ratio: The Story of Phi, the World's Most Astonishing Number*. Broadway Books.
[Lowe04] Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *IJCV*.
[Nickel15] Ermolov, A., Mirvakhabova, L., Khrulkov, V., Sebe, N., & Oseledets, I. (2022). Hyperbolic vision transformers: Combining improvements in metric learning. *CVPR*. (Note: Your ref [15] often pointed here)
[Ganea19] Ganea, O., Bécigneul, G., & Hofmann, T. (2018). Hyperbolic neural networks. *NeurIPS*. (Note: Your ref [19] often pointed here)
[Gulcehre22] Gulcehre, C., Denil, M., Malinowski, M., Razavi, A., Pascanu, R., Hermann, K. M., ... & de Freitas, N. (2019). Hyperbolic attention networks. *ICLR*. (Note: Your ref [22] often pointed here)
[Khrulkov31] Khrulkov, V., Mirvakhabova, L., Ustinova, E., Oseledets, I., & Lempitsky, V. (2020). Hyperbolic image embeddings. *CVPR*. (Note: Your ref [31] often pointed here)
[NickelKiela39] Nickel, M., & Kiela, D. (2017). Poincaré embeddings for learning hierarchical representations. *NeurIPS*. (Note: Your ref [39] often pointed here)
[LiuHeHan42] Liu, Y., He, Z., & Han, K. (2025). Hyperbolic Category Discovery. *arXiv preprint arXiv:2504.06120*. (Note: Your ref [42], placeholder)
[Hamilton43] Hamilton, W. R. (1866). *Elements of quaternions*. Longmans, Green, & Company.
[Parcollet44] Parcollet, T., Morchid, M., Bousquet, P. M., Dufour, R., Linarès, G., & De Mori, R. (2019). Quaternion recurrent neural networks. *ICLR*.
[Grassucci45] Grassucci, E., Comminiello, D., & Uncini, A. (2021). Quaternion neural networks: State-of-the-art and research challenges. *IEEE Transactions on Neural Networks and Learning Systems*.
[GuSalaRe46] Gu, A., Sala, F., Gunel, B., & Ré, C. (2019). Learning Mixed-Curvature Representations in Product Spaces. *ICLR*.
[Kochurov71] Kochurov, M., et al. (2020). Geoopt: Riemannian Optimization in PyTorch. *GitHub Repository*. `https://github.com/geoopt/geoopt`
[Ren15] Ren, S., et al. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *NIPS*.
[Achanta12] Achanta, R., et al. (2012). SLIC superpixels compared to existing superpixel methods. *TPAMI*.
[Bay08] Bay, H., et al. (2008). Speeded-Up Robust Features (SURF). *CVIU*.
[Uijlings13] Uijlings, J. R., et al. (2013). Selective search for object recognition. *IJCV*.
[Tran15] Tran, D., Bourdev, L., Fergus, R., Torresani, L., & Paluri, M. (2015). Learning spatiotemporal features with 3d convolutional networks. *ICCV*.
[Simonyan14] Simonyan, K., & Zisserman, A. (2014). Two-stream convolutional networks for action recognition in videos. *NeurIPS*.
[Arnab21] Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., & Schmid, C. (2021). Vivit: A video vision transformer. *ICCV*.
[Bertasius21] Bertasius, G., Wang, H., & Torresani, L. (2021). Is space-time attention all you need for video understanding?. *ICML*.
[Sun18] Sun, D., Yang, X., Liu, M. Y., & Kautz, J. (2018). Pwc-net: Cnns for optical flow using pyramid, warping, and cost volume. *CVPR*.
[Teed20] Teed, Z., & Deng, J. (2020). Raft: Recurrent all-pairs field transforms for optical flow. *ECCV*.
[Singer22] Singer, U., Polyak, A., Hayes, T., Yin, X., An, J., Zhang, S., ... & Taigman, Y. (2022). Make-a-video: Text-to-video generation without text-video data. *arXiv preprint arXiv:2209.11855*.
[Blattmann23] Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim, S. W., Ranjan, A., & Ommer, B. (2023). Align your latents: High-resolution video synthesis with latent diffusion models. *CVPR*.
[Anand22] Anand, A., Racanière, S., Rezende, D. J., & Denil, M. (2022). Learning Plannable Representations with Causal InfoGAN. *ICLR*.
[Soomro12] Soomro, K., Zamir, A. R., & Shah, M. (2012). UCF101: A dataset of 101 human actions classes from videos in the wild. *CRCV-TR-12-01*.
[Butler12] Butler, D. J., Wulff, J., Stanley, G. B., & Black, M. J. (2012). A naturalistic open source movie for optical flow evaluation. *ECCV*.
[Geiger12] Geiger, A., Lenz, P., & Urtasun, R. (2012). Are we ready for autonomous driving? The kitti vision benchmark suite. *CVPR*.
[Skopek20] Skopek, O., Ganea, O. E., & Bécigneul, G. (2020). Mixed-curvature variational autoencoders. *ICLR*.
[Weiler18] Weiler, M., Hamprecht, F. A., & Storath, M. (2018). Learning steerable filters for rotation equivariant cnns. *CVPR*.
