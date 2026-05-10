

## WuBu Nesting (層疊嵌套 - céngdié qiàn​tào): An Adaptive, Rotation-Aware, Nested Hyperbolic Framework for Complex Geometric Structures

**Abstract**

Modeling the intricacies of real-world data often necessitates capturing a confluence of complex characteristics: deep multi-scale hierarchical organizations, intrinsic rotational symmetries or transformations between structural levels, dynamic evolution within scales, and varying degrees of uncertainty or regional influence. Existing geometric deep learning paradigms, while powerful, often specialize. Standard Euclidean models struggle with hierarchical data, traditional hyperbolic models excel at single-level hierarchies but typically lack integrated rotational mechanics and multi-scale adaptivity, and quaternion-based models handle rotations efficiently but usually lack inherent hierarchical or multi-scale geometric structure. To bridge these gaps and provide a more holistic geometric inductive bias, we introduce **WuBu Nesting (層疊嵌套 - céngdié qiàn​tào: "layered nesting")**, a novel and comprehensive conceptual framework. WuBu Nesting features a recursively nested structure of hyperbolic spaces (conceptually, `H^n1_{c1,s1} ⊃ H^n2_{c2,s2} ⊃ ...`), where the dimensionality (`n_i`), curvature (`c_i > 0`), and relative scale (`s_i > 0`) of each hyperbolic "bubble" can be learned, allowing the geometry to adapt dynamically to data complexity and distribution. Within each level `i`, the framework accommodates learnable **Boundary Sub-Manifolds** (e.g., parameterized point sets representing "circles" or landmarks) symbolizing distinct substructures or feature clusters relevant at that scale. Crucially, transitions between adjacent levels (`i → i+1`) are orchestrated via sophisticated, learnable transformations operating within the flat **Euclidean tangent spaces** (`T_p(H^n_i) ≅ R^n_i`) associated with the hyperbolic manifolds. These inter-level transformations are deliberately decomposed into a learned **Rotation** component (`R_i`, implemented efficiently via Quaternions for 4D or general `SO(n_i)` matrices) applied *simultaneously* to the primary data representation, the representations of boundary manifolds (mapped to tangent vectors), and a novel **learnable Level Descriptor Vector** (represented as `ld_i`) intrinsic to the source level. This rotation is followed by a learnable **non-rotational Mapping** (`T̃_i`) that adjusts features and potentially dimensionality, producing transformed vectors in the target tangent space (`T_o(H^n_{i+1})`). From these transformed vectors, we compute **Relative Vectors** (represented as `d_{i+1, j, k}`) between the primary representation and the boundary representations, explicitly encoding rotation-aware spatial relationships in the target tangent space. Furthermore, each level `i` possesses a learnable **Level Spread Parameter** (`σ_i`), representing characteristic uncertainty or density, which is passed as context to the next level. The framework also allows for **Intra-Level Tangent Flow** (`F_i`), a learnable dynamic transformation applied within the tangent space during a level's internal processing step, modeling localized evolution or adjustment. The inputs informing the processing at level `i+1` thus include the primary representation mapped back into `H^n_{i+1}`, the computed relative vectors `{d_{i+1, j, k}}`, the transformed level descriptor `ld_{i+1}`, and the contextual spread parameter `σ_i`. This rich, multi-faceted information stream allows WuBu Nesting to capture scale-aware, rotation-informed, dynamic, and density-sensitive relationships, offering an exceptionally flexible and powerful geometric framework adaptable to the profound complexity of real-world data exhibiting intertwined hierarchical, rotational, dynamic, and uncertain characteristics.

## 1. Introduction

The quest for effective data representation lies at the heart of machine learning. Standard deep learning architectures, while achieving remarkable success, predominantly operate within the confines of Euclidean geometry. This geometric choice, however, imposes limitations when modeling data imbued with strong intrinsic structures not naturally suited to flat spaces. A prominent example is hierarchical data, such as taxonomies, phylogenetic trees, complex molecules, articulated objects, or parse trees, where relationships exhibit a natural parent-child structure. Embedding such hierarchies into Euclidean space often incurs significant distortion, as the space's polynomial volume growth struggles to accommodate the exponential expansion of nodes typically found in trees [39].

Hyperbolic geometry, characterized by its constant negative curvature and exponential volume growth relative to radius, offers a mathematically elegant and practically effective solution for embedding hierarchical structures with significantly lower distortion [39, 31, 15]. Models leveraging spaces like the Poincaré disk or ball (`H^n`) have demonstrated substantial benefits in tasks ranging from graph embedding and natural language processing [19, 22] to computer vision [31, 15, 1] and category discovery [42]. These successes underscore the power of aligning the model's geometric inductive bias with the data's underlying structure.

However, many real-world systems exhibit complexities beyond a single, static hierarchy. Firstly, hierarchies themselves can be **nested**: structures contain sub-structures which themselves possess internal hierarchies (e.g., a molecule composed of domains, composed of secondary structures, composed of residues). Secondly, components within these structures often possess **intrinsic orientations**, and transformations between different levels or viewpoints frequently involve **rotations**. For instance, analyzing articulated objects requires understanding part hierarchies alongside their relative orientations and movements, while modeling protein interactions involves recognizing hierarchical domains and their rotational alignment during docking. Existing hyperbolic models typically focus on embedding a single hierarchy level within a single hyperbolic space of fixed curvature and lack native, efficient mechanisms for modeling rotations or adaptively handling multiple scales of hierarchy.

Conversely, **Quaternions** [43] provide an exceptionally compact and computationally efficient algebra for representing and manipulating rotations, particularly in 3D and 4D. Quaternion Neural Networks (QNNs) [44, 45] have leveraged this power for tasks involving orientation and 3D data, demonstrating parameter efficiency and improved performance. However, QNNs typically operate within Euclidean spaces and lack the intrinsic capacity for hierarchical embedding offered by hyperbolic geometry. Combining different geometries via product spaces (e.g., `R^n × S^m × H^k`) [46] offers increased capacity by arranging spaces in parallel, but does not directly address nested hierarchies or integrated rotational transformations *between* hierarchical levels.

This paper introduces **WuBu Nesting (層疊嵌套)**, a comprehensive conceptual framework meticulously designed to bridge these gaps. WuBu Nesting aims to unify adaptive multi-scale hierarchical representation with explicit modeling of rotational dynamics, dynamic evolution, and regional uncertainty within a single, cohesive geometric structure. Instead of a single hyperbolic space or a parallel product manifold, WuBu Nesting proposes a nested "Russian doll" architecture comprising recursively embedded hyperbolic manifolds. The key innovations, detailed extensively in this paper, are:

1.  **Adaptive Nested Hyperbolic Geometry:** The core structure is a sequence of nested hyperbolic spaces, conceptually `H^n1_{c1, s1} ⊃ H^n2_{c2, s2} ⊃ ...`. Critically, the dimensionality (`n_i`), curvature (`c_i > 0`), and a relative scale parameter (`s_i > 0`, influencing the zoom/density) of each hyperbolic "bubble" can be learnable parameters, allowing the overall geometry to dynamically adapt its capacity and structure to the specific data distribution and complexity.
2.  **Boundary Sub-Manifolds:** Each hyperbolic level `H^n_i` can host learnable, lower-dimensional **Boundary Sub-Manifolds** (`B_{i,j}`), such as sets of points representing hyperbolic disks ("circles") or other landmark configurations. These symbolize distinct substructures, components, or feature clusters pertinent to the scale represented by level `i`.
3.  **Tangent Space Transitions:** Transitions between levels (`i → i+1`) are mediated not directly in the curved hyperbolic spaces, but within their associated **Euclidean Tangent Spaces** (`T_p(H^n_i) ≅ R^n_i`). This allows leveraging the well-understood properties and operations of Euclidean vector spaces for complex transformations.
4.  **Explicit Tangent Space Rotations (`R_i`):** A core component of the inter-level transition is a learnable **Rotation** `R_i`. This rotation operates within the tangent space `T_o(H^n_i)` and can be implemented using efficient quaternion multiplication (if `n_i=4`) or general `SO(n_i)` rotation matrices (parameterized appropriately).
5.  **Simultaneous Transformation:** The learned rotation `R_i` is applied *consistently and simultaneously* to the tangent vector representing the main data point (`v_i`), the tangent vectors representing the boundary manifolds (`v_{b_{i,j,k}}`), and a learnable Level Descriptor Vector (`ld_i`). This ensures that the relative orientations of all relevant features are preserved and correctly transformed into the rotated frame.
6.  **Non-Rotational Mapping (`T̃_i`):** Following the rotation, a learnable **non-rotational mapping** `T̃_i: T_o(H^n_i) → T_o(H^n_{i+1})` is applied. This mapping handles potential dimension changes (`n_i → n_{i+1}`), applies non-linear feature transformations (e.g., using MLPs), and prepares the vectors for the target tangent space of the next level. The full inter-level tangent transformation is thus:
    ```math
    T_{i \rightarrow i+1} = \tilde{T}_i \circ R_i
    ```
7.  **Relative Vector Generation (`d_{i+1}`):** After the full tangent space transformation (`T_{i → i+1}`), **Relative Vectors** are computed in the target tangent space `T_o(H^n_{i+1})`:
    ```math
    d_{i+1, j, k} = v_{i+1} - v''_{b_{i,j,k}}
    ```
    These vectors explicitly encode the spatial relationship between the transformed primary data representation (`v_{i+1}$) and the transformed boundary representations (`v''_{b_{i,j,k}}$), capturing geometry informed by the inter-level rotation.
8.  **Learnable Level Descriptor Vector (`ld_i`):** Each level `i` possesses an intrinsic, learnable **Level Descriptor Vector** `ld_i ∈ T_o(H^n_i)`. This vector, transformed alongside other features to `ld_{i+1}` for the next level, potentially captures scale-specific anisotropy, a dominant feature direction, or other characteristic geometric properties of the level itself.
9.  **Learnable Level Spread Parameter (`σ_i`):** Each level `i` is associated with a learnable scalar **Level Spread Parameter** `σ_i > 0`, representing the characteristic "atmosphere," uncertainty radius, or density falloff at that scale. This parameter is passed as contextual information to the subsequent level (`i+1`).
10. **Intra-Level Tangent Flow (`F_i`):** The framework allows for a learnable **Intra-Level Tangent Flow** field `F_i: T_o(H^n_i) → T_o(H^n_i)`, applied during the internal processing stage of level `i`. This mechanism can model scale-specific dynamics, adjustments, or "orbital" transformations within the level's tangent space representation before output.
11. **Rich Hierarchical Information Flow:** The processing module within level `i+1` receives a comprehensive set of inputs: the primary representation mapped into `H^n_{i+1}`, the set of relative vectors `{d_{i+1, j, k}}`, the transformed level descriptor `ld_{i+1}`, and the contextual spread parameter `σ_i` from level `i`. This rich input stream allows the model to make decisions based on position, relative structure, orientation, level characteristics, and source level uncertainty.

We hypothesize that this deeply integrated, multi-faceted geometric structure – encompassing nested adaptivity, explicit boundaries, tangent space rotations, relative geometry, level descriptors, spread context, and intra-level dynamics – provides an exceptionally powerful and flexible inductive bias. WuBu Nesting is proposed as a foundational framework capable of modeling complex real-world systems where hierarchical, rotational, dynamic, and uncertainty characteristics are inextricably intertwined.

## 2. Related Work

The WuBu Nesting framework draws inspiration from, and aims to synthesize concepts across, several distinct areas of geometric deep learning and representation learning.

### 2.1 Hyperbolic Deep Learning

The seminal work of Nickel and Kiela [39] demonstrated the aptitude of hyperbolic geometry, specifically the Poincaré ball model, for embedding hierarchical data structures like taxonomies with significantly lower distortion compared to Euclidean counterparts. This spurred a wave of research exploring hyperbolic geometry for various machine learning tasks. Key developments include:
*   **Hierarchical Embeddings:** Further applications in embedding trees [31], graphs [15], and ontologies [1].
*   **Hyperbolic Neural Networks:** Defining analogues of standard neural network operations within hyperbolic space, such as fully connected layers (Gyroplane layers) [19], attention mechanisms [22], and convolutions [55]. Ganea et al. [19] provided foundational work on hyperbolic neural networks using the gyrovector space formalism.
*   **Computer Vision:** Applying hyperbolic embeddings to image classification [31], image retrieval [15], object detection, semantic segmentation [1], and category discovery (e.g., HypCD [42]), often showing benefits where hierarchical part relationships or semantic hierarchies are relevant.

**Critique & WuBu Distinction:** While foundational, these methods predominantly utilize a *single* hyperbolic space with a *fixed* curvature. They typically lack mechanisms for handling *nested* hierarchies adaptively and do not incorporate explicit modeling of *rotational* transformations or the other novel components introduced in WuBu Nesting (boundaries, relative vectors, descriptors, spread, flow). HypCD [42], for example, shows the benefit of a single hyperbolic space for GCD but doesn't employ nesting or explicit rotations.

### 2.2 Quaternion Neural Networks (QNNs)

Quaternions [43], a four-dimensional normed division algebra extending complex numbers, offer a highly efficient representation for 3D rotations. QNNs [44, 45] leverage this property:
*   **Parameter Efficiency:** Quaternion-valued weights and operations can significantly reduce the number of parameters compared to equivalent real-valued networks for tasks involving 3D/4D structure or rotations.
*   **Rotational Equivariance/Invariance:** QNNs can be designed to better respect rotational symmetries.
*   **Applications:** Primarily successful in areas like 3D computer vision, robotics, signal processing, and physics simulations where orientation and rotation are critical.

**Critique & WuBu Distinction:** QNNs operate primarily in Euclidean space (or spaces easily representable with quaternions). They lack the intrinsic geometric bias for hierarchical embedding provided by hyperbolic spaces. WuBu Nesting incorporates rotational modeling (potentially using quaternions when `n_i = 4`) but does so within a *tangent space transition* mechanism *between nested hyperbolic levels*, thus integrating rotation with adaptive multi-scale hierarchy.

### 2.3 Product Manifolds and Multi-Scale Approaches

To combine the strengths of different geometries, some approaches utilize **Product Manifolds** [46, 56, 57], creating spaces like `R^n × S^m × H^k`.
*   **Increased Capacity:** Allows simultaneous representation in spaces with different inductive biases (e.g., Euclidean for attributes, Spherical for directions, Hyperbolic for hierarchy).
*   **Parallel Structure:** Geometries are typically arranged in parallel; information is processed within each component space and then aggregated.

Traditional **Multi-Scale Methods** in deep learning (e.g., feature pyramids in vision, wavelet transforms) typically operate in Euclidean space, extracting features at different spatial resolutions or frequency bands.

**Critique & WuBu Distinction:** Product manifolds offer parallel capacity but do not inherently model the *nested*, "Russian doll" structure proposed by WuBu Nesting. Transitions and interactions between the different geometric components in product spaces are often handled via simple concatenation or aggregation, lacking the sophisticated, rotation-aware tangent space transformations of WuBu. Standard multi-scale methods lack the specific geometric biases of hyperbolic spaces and the integrated rotational modeling. WuBu Nesting proposes a fundamentally different architecture based on *recursive embedding* and *geometrically meaningful transitions*, integrating hierarchy, scale, rotation, dynamics, and uncertainty in a deeply unified manner.

In summary, WuBu Nesting distinguishes itself by proposing a novel synthesis: an *adaptive, nested hyperbolic* structure providing multi-scale hierarchy, combined with *explicit tangent space rotations and mappings* for handling orientation changes between levels, further enriched by *learnable boundary manifolds, relative vectors, level descriptors, spread parameters, and intra-level flows* to capture unprecedented geometric detail and dynamics.

## 3. The WuBu Nesting Framework

WuBu Nesting offers a recursive, multi-layered geometric architecture where data representations are progressively refined through a series of nested hyperbolic "bubbles." Transitions between these bubbles are orchestrated in their associated Euclidean tangent spaces, incorporating learnable rotations, mappings, and the generation of rich geometric features like relative vectors, while also considering level-specific descriptors, spread, and internal dynamics.

### 3.1. Conceptual Architecture

The core concept envisions data flowing through a hierarchy of processing stages, each associated with a hyperbolic space `H^n_i_{c_i, s_i}`. An initial encoding maps the input data into the tangent space of the outermost hyperbolic level. Within each level `i`, the representation undergoes processing which may involve an intra-level tangent flow `F_i`. To transition to the next, deeper level `i+1`, the representation (along with boundary manifold representations and the level descriptor vector) is mapped to the tangent space `T_o(H^n_i)` via the logarithmic map. Here, a learned rotation `R_i` is applied simultaneously to all these vectors. Subsequently, a learnable non-rotational mapping `T̃_i` transforms these rotated vectors into the tangent space `T_o(H^n_{i+1})` of the next level, potentially changing dimensionality. In this target tangent space, relative vectors (`d_{i+1, j, k}`) are computed between the main transformed vector and the transformed boundary vectors. The main transformed vector `v_{i+1}`, the relative vectors `{d_{i+1, j, k}}`, the transformed level descriptor `ld_{i+1}`, and the spread parameter `σ_i` from the source level collectively form the input for processing within level `i+1`. The main vector `v_{i+1}$ is typically mapped into the hyperbolic ball `H^n_{i+1}` using the exponential map for hyperbolic operations within that level. This process repeats recursively through the nested levels. Finally, information aggregated across relevant levels and tangent spaces is used for the final task prediction.

```mermaid
graph TD
    %% == 1. Define ALL Nodes Globally ==
    A(InputData)
    B(InitialEuclideanEncoding)
    C(MapToTangentSpaceH1)
    Proc1(IntraLevelProcessingL1)
    LD1(L1DescLd1_Param)
    Sigma1(L1SpreadSigma1_Param)
    Flow1(L1FlowF1_Module)
    BM1(L1BoundaryManifoldModule)
    BM1P(GetL1BoundaryPoints_b1jk)
    D(L1StateOut_x1)
    F(LogMap_o_s1_c1(x1))
    FBT1(LogMap_o_s1_c1(b1jk))
    R1(RotationModule_R1)
    R1_V(Rotate v1_out)
    R1_B(Rotate v_b1jk)
    R1_LD(Rotate ld1_param)
    T1Map(MapModule_Ttilde1)
    V2(TargetTangentH2_v2)
    V2B(TargetBoundaryTangentH2_v''_b1jk)
    Ld2In(TargetDescTangentH2_ld2_in)
    VectorsD2(ComputeRelativeVectors_d2jk)
    Ctx2(ContextL2_PassSigma1)
    Proc2Input(GatherInputsL2)
    Proc2(IntraLevelProcessingL2)
    LD2(L2DescLd2_Param)
    Sigma2(L2SpreadSigma2_Param)
    Flow2(L2FlowF2_Module)
    BM2(L2BoundaryManifoldModule)
    BM2P(GetL2BoundaryPoints_b2jk)
    J1(L2StateOut_x2)
    L1Out(LogMap_o_s2_c2(x2))
    FBT2(LogMap_o_s2_c2(b2jk))
    R2(RotationModule_R2)
    R2_V(Rotate v2_out)
    R2_B(Rotate v_b2jk)
    R2_LD(Rotate ld2_param)
    T2Map(MapModule_Ttilde2)
    V3(TargetTangentH3_v3)
    V3B(TargetBoundaryTangentH3_v''_b2jk)
    Ld3In(TargetDescTangentH3_ld3_in)
    VectorsD3(ComputeRelativeVectors_d3jk)
    Ctx3(ContextL3_PassSigma2)
    Proc3Input(GatherInputsL3)
    N(IntraLevelProcessingL3)
    O(AggregateTangentInfo)
    P(FinalProjectionTaskHead)
    Q(Output)

    %% == 2. Define ALL Edges Globally ==
    A --> B; B --> C;
    C --> Proc1;
    BM1 --> BM1P; BM1P --> FBT1;
    LD1 --> R1_LD; Sigma1 --> Ctx2;
    Flow1 -- Used_by --> Proc1;
    Proc1 --> D; D --> F;
    F --> R1_V; FBT1 --> R1_B;
    R1_V -- Input --> R1; R1_B -- Input --> R1; R1_LD -- Input --> R1;
    R1 -- Output --> R1_V; R1 -- Output --> R1_B; R1 -- Output --> R1_LD;
    R1_V -- Input --> T1Map; R1_B -- Input --> T1Map; R1_LD -- Input --> T1Map;
    T1Map -- Output --> V2; T1Map -- Output --> V2B; T1Map -- Output --> Ld2In;
    V2 --> VectorsD2; V2B --> VectorsD2;
    V2 --> Proc2Input; VectorsD2 --> Proc2Input; Ld2In --> Proc2Input; Ctx2 --> Proc2Input;
    Proc2Input --> Proc2;
    BM2 --> BM2P; BM2P --> FBT2;
    LD2 --> R2_LD; Sigma2 --> Ctx3;
    Flow2 -- Used_by --> Proc2;
    Proc2 --> J1; J1 --> L1Out;
    L1Out --> R2_V; FBT2 --> R2_B;
    R2_V -- Input --> R2; R2_B -- Input --> R2; R2_LD -- Input --> R2;
    R2 -- Output --> R2_V; R2 -- Output --> R2_B; R2 -- Output --> R2_LD;
    R2_V -- Input --> T2Map; R2_B -- Input --> T2Map; R2_LD -- Input --> T2Map;
    T2Map -- Output --> V3; T2Map -- Output --> V3B; T2Map -- Output --> Ld3In;
    V3 --> VectorsD3; V3B --> VectorsD3;
    V3 --> Proc3Input; VectorsD3 --> Proc3Input; Ld3In --> Proc3Input; Ctx3 --> Proc3Input;
    Proc3Input --> N;
    N --> O; % Example: Output tangent of L3
    Proc2 --> O; % Example: Output tangent of L2
    Proc1 --> O; % Example: Output tangent of L1
    O --> P; P --> Q;


    %% == 3. Define Subgraphs for Grouping ==
    subgraph Level1 ["Level 1: (H_n1, c1, s1)"]
        Proc1; D; F; BM1P; FBT1; LD1; Sigma1; Flow1; BM1; C;
    end
    subgraph InterLevelTransformationT12 ["Inter-Level Transformation T(1->2)"]
         R1; R1_V; R1_B; R1_LD; T1Map; V2; V2B; Ld2In; VectorsD2; Ctx2;
    end
     subgraph Level2 ["Level 2: (H_n2, c2, s2)"]
        Proc2Input; Proc2; J1; L1Out; BM2P; FBT2; LD2; Sigma2; Flow2; BM2;
    end
     subgraph InterLevelTransformationT23 ["Inter-Level Transformation T(2->3)"]
        R2; R2_V; R2_B; R2_LD; T2Map; V3; V3B; Ld3In; VectorsD3; Ctx3;
    end
    subgraph Level3 ["Level 3: (H_n3, c3, s3)"]
        Proc3Input; N;
    end
    subgraph AggregationOutput ["Aggregation & Output"]
        O; P; Q;
    end


    %% == 4. Styling ==
    classDef baseStyle color:#000000,font-size:14px
    classDef level1 fill:#B2DFDB,stroke:#00796B,stroke-width:1.5px
    classDef level2 fill:#E0F2F1,stroke:#00796B,stroke-width:1px
    classDef level3 fill:#E0F7FA,stroke:#006064,stroke-width:1px
    classDef processing fill:#C8E6C9,stroke:#388E3C,stroke-width:1.5px
    classDef rotation fill:#FFF176,stroke:#FBC02D,stroke-width:2px
    classDef transform fill:#FFCCBC,stroke:#E64A19,stroke-width:1.5px
    classDef tangent fill:#FFFFFF,stroke:#BDBDBD,stroke-dasharray:2 2,stroke-width:1px
    classDef ball fill:#F5F5F5,stroke:#BDBDBD,stroke-width:1px
    classDef boundary fill:#EEEEEE,stroke:#757575,stroke-dasharray:5 5,stroke-width:1px
    classDef vectorGen fill:#BBDEFB,stroke:#1976D2,stroke-width:1.5px
    classDef levelDesc fill:#E0E0E0,stroke:#616161,stroke-dasharray:3 3,stroke-width:1px
    classDef spread fill:#CFD8DC,stroke:#546E7A,stroke-dasharray:1 1,stroke-width:1px
    classDef flow fill:#B3E5FC,stroke:#0277BD,stroke-dasharray:4 4,stroke-width:1px
    classDef module stroke-dasharray: 4 1, stroke-width: 1px, color:#333333

    %% Apply base text style to ALL nodes first
    class A,B,C,Proc1,LD1,Sigma1,Flow1,BM1,BM1P,D,F,FBT1,R1,R1_V,R1_B,R1_LD,T1Map,V2,V2B,Ld2In,VectorsD2,Ctx2,Proc2Input,Proc2,LD2,Sigma2,Flow2,BM2,BM2P,J1,L1Out,FBT2,R2,R2_V,R2_B,R2_LD,T2Map,V3,V3B,Ld3In,VectorsD3,Ctx3,Proc3Input,N,O,P,Q baseStyle

    %% Apply specific styles
    class Proc1,Proc2,N,Proc2Input,Proc3Input processing
    class BM1,BM2,BM1P,BM2P boundary
    class R1,R2 rotation; class R1_V,R1_B,R1_LD,R2_V,R2_B,R2_LD tangent
    class T1Map,T2Map transform
    class C,F,FBT1,V2,V2B,L1Out,FBT2,V3,V3B tangent
    class D,J1 ball
    class VectorsD2,VectorsD3 vectorGen
    class LD1,LD2,Ld2In,Ld3In levelDesc
    class Sigma1,Sigma2,Ctx2,Ctx3 spread
    class Flow1,Flow2 flow
    class R1,T1Map,BM1,BM2,Flow1,Flow2 module

    %% Apply Level Backgrounds
    class C,Proc1,D,F,BM1P,FBT1,LD1,Sigma1,Flow1,BM1 level1
    class Proc2Input,Proc2,J1,L1Out,BM2P,FBT2,LD2,Sigma2,Flow2,BM2 level2
    class Proc3Input,N level3
```
**Figure 1:** Conceptual Architecture of the Comprehensive WuBu Nesting Framework. This diagram illustrates the flow through nested hyperbolic levels (`H^n_i_{c_i, s_i}`) with adaptive parameters. It highlights key components: learnable Boundary Manifolds (`B_{i,j,k}`), Level Descriptors (`ld_i`), Level Spreads (`σ_i`), and Intra-Level Tangent Flows (`F_i`). Inter-level transitions involve tangent space mapping (LogMap), simultaneous Rotation (`R_i`) of primary (`v_i`), boundary (`v_{b_ijk}`), and descriptor (`ld_i`) vectors, followed by a Mapping (`T̃_i`). Relative Vectors (`d_{i+1}`) are computed in the target tangent space. The next level's processing utilizes the transformed primary tangent vector (`v_{i+1}$), relative vectors (`d_{i+1}`), transformed descriptor (`ld_{i+1}`), and contextual spread (`σ_i`).

### 3.2. Component Details

We now elaborate on each distinct component of the WuBu Nesting framework.

#### 3.2.1 Nested Hyperbolic Spaces & Adaptive Geometry
The foundational structure is a sequence of **nested hyperbolic spaces**. We typically employ the **Poincaré Ball model** for each level `i`, denoted `H^n_i_{c_i, s_i}`.
*   **Nesting:** The embedding conceptually proceeds from an outer, potentially lower-curvature space `H^n_1` to progressively deeper, potentially higher-curvature or differently scaled spaces `H^n_2, H^n_3, ...`. This nesting allows the model to capture hierarchical structure across multiple scales.
*   **Dimensionality (`n_i`):** The dimension `n_i` of the hyperbolic space at level `i$ can vary between levels. This allows the model to allocate representational capacity differently across the hierarchy. `n_i` could be a hyperparameter or potentially learned/selected via architecture search.
*   **Curvature (`c_i`):** The curvature parameter `c_i > 0` (where the manifold curvature is typically `-c_i^2`) determines the "steepness" of the geometry at level `i`. Higher curvature leads to faster volume growth and potentially better embedding of deep hierarchies within that level. `c_i` can be a fixed hyperparameter per level or, more powerfully, a **learnable parameter**, allowing the model to adapt the geometry's intensity at each scale. Learning requires careful optimization (e.g., using parameterizations like `c_i = softplus(raw_c_i) + min_c` or Riemannian optimization) to keep `c_i` positive and stable.
*   **Scale (`s_i`):** We introduce a **learnable positive scale parameter** `s_i > 0` for each level `i$. This parameter acts as a "zoom factor" modulating the relationship between the tangent space and the hyperbolic ball, typically incorporated into the scale-aware exponential and logarithmic maps. Conceptually, a scale-aware exponential map might resemble:
    ```math
    \text{exp}_{o,s_i}^{c_i}(v) = \tanh\left(s_i \cdot \frac{\sqrt{c_i}\|v\|}{2}\right) \frac{v}{\sqrt{c_i}\|v\|} \quad \text{(Needs careful derivation)}
    ```
    The scale-aware log map would be its inverse. Learning `s_i` allows the model to control the effective density or spatial extent represented within each level's tangent space mapping. Ensuring `s_i > 0` requires parameterization like `s_i = softplus(raw_s_i) + min_s`.

#### 3.2.2 Boundary Sub-Manifolds (`B_{i,j}$)
To explicitly model substructures or landmark features within a given scale, each level `H^n_i` can host a set of learnable **Boundary Sub-Manifolds** `B_{i,j}`.
*   **Representation:** These are conceptually lower-dimensional manifolds embedded within `H^n_i`. A practical implementation involves parameterizing them using a set of characteristic **learnable points** `{b_{i,j,k}} ⊂ H^n_i`. These points `b_{i,j,k}` are model parameters, learned via backpropagation. More simply, they can be represented as learnable *tangent vectors* `{v_{b_{i,j,k}}^{param}} \in T_o(H^n_i)$ at the origin, which are then mapped into the ball `b_{i,j,k} = \text{exp}_{o,s_i}^{c_i}(v_{b_{i,j,k}}^{param})$ when needed for hyperbolic computations, but primarily manipulated in tangent space for transitions. This avoids optimizing points directly in `H^n_i`.
*   **Purpose:** They represent distinct components, parts, feature clusters, or reference frames relevant at the scale defined by level `i$. Their relative positions to the main data representation `x_i` become important features.
*   **Transformation:** For inter-level transitions, their tangent vectors `v_{b_{i,j,k}}^{param}` are rotated by `R_i` and mapped by `T̃_i` alongside the primary representation.

#### 3.2.3 Tangent Space Logic
A cornerstone of WuBu Nesting is that complex transformations, particularly rotations and mappings between potentially different dimensions, occur within the **Euclidean tangent spaces** associated with the hyperbolic levels (typically centered at the origin `o`).
*   **Mapping To/From:** The **Logarithmic Map** (`Log^{c_i}_{o,s_i}: H^n_i → T_o(H^n_i)`) projects points from the hyperbolic ball to the tangent space at the origin, incorporating the scale `s_i` and curvature `c_i$. The **Exponential Map** (`exp^{c_i}_{o,s_i}: T_o(H^n_i) → H^n_i`) performs the inverse projection. Robust implementations (e.g., from libraries like `geoopt` [71] or custom implementations as shown in the Python code) are crucial.
*   **Operations:** Within the tangent space `T_o(H^n_i) ≅ R^n_i`, standard Euclidean vector operations (addition, subtraction, scalar multiplication, linear transformations, rotations, MLPs) can be applied.

#### 3.2.4 Tangent Space Rotations (`R_i`)
To explicitly model orientational changes between hierarchical levels, a learnable **Rotation** `R_i` is applied in the tangent space `T_o(H^n_i)` during the `i → i+1` transition.
*   **Implementation:**
    *   If `n_i = 4`, `R_i` can be efficiently implemented using **unit quaternion multiplication**. A general `SO(4)` rotation can be parameterized by two unit quaternions `p, q` acting as:
      ```math
      v' = p \cdot v_{quat} \cdot q
      ```
      where `v_{quat}` is the 4D tangent vector represented as a quaternion (e.g., with zero scalar part if representing 3D vectors). These unit quaternions (8 parameters constrained to `S^3 × S^3`) are learned.
    *   If `n_i ≠ 4`, `R_i` is implemented using **rotation matrices** from the Special Orthogonal group `SO(n_i)`. These matrices `R_i ∈ R^{n_i × n_i}` satisfy `R_i^T R_i = I` and `det(R_i)=1`. They are learned parameters, typically parameterized using techniques that ensure they remain on the `SO(n_i)` manifold during optimization (e.g., using the matrix exponential map from the Lie algebra `so(n_i)`, or using orthogonal parameterizations like Cayley maps or projections [58, 62]).
*   **Simultaneous Application:** `R_i` is applied to the main tangent vector `v_i^{out}`, all boundary tangent vectors `v_{b_{i,j,k}}`, and the level descriptor vector `ld_i` originating from level `i`.

#### 3.2.5 Non-Rotational Mapping (`T̃_i`)
Following the rotation `R_i`, a learnable **non-rotational mapping** `T̃_i` is applied to the rotated tangent vectors.
*   **Purpose:** This component handles feature transformation, non-linear interactions, and dimensionality changes between levels (`n_i → n_{i+1}`).
*   **Implementation:** `T̃_i: T_o(H^n_i) → T_o(H^n_{i+1})` can be implemented using standard neural network layers operating on vectors, such as:
    *   Multi-Layer Perceptrons (MLPs).
    *   Linear projections (if only dimension change is needed).
*   **Output:** Produces the final tangent vectors `v_{i+1}`, `v''_{b_{i,j,k}}`, and `ld_{i+1}$ in the target tangent space `T_o(H^n_{i+1})`.

#### 3.2.6 Relative Vector Generation (`d_{i+1, j, k}$)
After the full tangent space transformation `T_{i → i+1} = T̃_i ∘ R_i`, **Relative Vectors** are computed directly in the target Euclidean tangent space `T_o(H^n_{i+1})`.
*   **Computation:**
    ```math
    d_{i+1, j, k} = v_{i+1} - v''_{b_{i,j,k}}
    ```
*   **Purpose:** These vectors explicitly encode the geometric relationship (displacement and direction) between the primary data representation and the boundary substructures *after* accounting for the learned rotation and mapping between levels. They provide rich, orientation-aware structural information.
*   **Usage:** The set of relative vectors `{d_{i+1, j, k}}` (potentially aggregated, e.g., via mean or sum pooling, or used in an attention-like mechanism) is passed as input to the processing stage of the next level, `H^n_{i+1}`.

#### 3.2.7 Learnable Level Descriptor Vector (`ld_i`)
Each level `i$ possesses an intrinsic **Learnable Level Descriptor Vector** `ld_i`.
*   **Representation:** `ld_i ∈ T_o(H^n_i) ≅ R^n_i` is a learnable parameter vector, initialized (e.g., randomly near zero) and optimized alongside other model parameters.
*   **Purpose:** This vector aims to capture characteristic geometric properties of level `i` itself, independent of the specific input data instance. It might learn to represent a preferred orientation, a direction of maximum variance within the level, an axis of symmetry, or some other scale-specific anisotropic feature.
*   **Transformation:** `ld_i` is treated similarly to other feature vectors during the inter-level transition: it is rotated by `R_i` (`ld'_i = R_i(ld_i)`) and then mapped by `T̃_i` (`ld_{i+1} = T̃_i(ld'_i)`).
*   **Usage:** The transformed vector `ld_{i+1}$ is passed as input to the processing stage of level `i+1`, providing context about the learned geometric characteristics of the source level `i`.

#### 3.2.8 Learnable Level Spread Parameter (`σ_i`)
Each level `i$ is associated with a **Learnable Level Spread Parameter** `σ_i`.
*   **Representation:** A learnable positive scalar parameter `σ_i > 0`. Learning requires ensuring positivity (e.g., parameterizing `σ_i = softplus(raw_σ_i) + min_σ`).
*   **Purpose:** Represents the characteristic "atmosphere," radius of influence, uncertainty measure, or density falloff associated with representations at scale `i$. A large `σ_i` might indicate broader clusters or higher uncertainty at that level.
*   **Transformation & Usage:** `σ_i` is typically passed directly as a scalar **contextual input** to the processing stage of the next level `i+1`. It does not usually undergo the rotation/mapping transform itself. The processing module at level `i+1` can use `σ_i` to modulate its computations, for example, by adjusting attention weights, scaling features, or simply using it as an additional input feature.

#### 3.2.9 Intra-Level Tangent Flow (`F_i`)
To model dynamics or adjustments *within* a scale, each level `i$ can incorporate a learnable **Intra-Level Tangent Flow** field `F_i`.
*   **Representation:** A learnable function `F_i: T_o(H^n_i) → T_o(H^n_i)` operating within the tangent space of level `i$. It could be parameterized as:
    *   An MLP predicting a displacement: `F_i(v) = \text{MLP}_i(v)`. The flowed vector is then: `v_{flowed} = v + F_i(v)`.
    *   A linear transformation: `F_i(v) = M_i v`. The flowed vector is then `v_{flowed} = F_i(v)`.
    *   More complex flows like Neural ODEs [63] could also be considered.
*   **Purpose:** Models characteristic evolution, refinement, or "orbital" adjustment of the representation pertinent to the scale `i$. It allows the representation to shift within its local geometric context before potentially being passed to the next level.
*   **Placement:** `F_i` is applied as part of the `IntraLevelProcessing` module within level `i$. It typically acts on a tangent space representation derived from the hyperbolic state or other tangent vectors within that level.

#### 3.2.10 Hierarchical Information Flow
The design ensures a rich flow of information between levels. The input to the `IntraLevelProcessing` module of level `i+1` comprises:
*   The primary tangent vector `v_{i+1}$.
*   Aggregated relative tangent vectors derived from `{d_{i+1, j, k}}`.
*   The transformed Level Descriptor tangent vector `ld_{i+1}$.
*   The scalar Level Spread parameter `σ_i` from the source level.
The `IntraLevelProcessing` module (which may itself apply the flow `F_{i+1}$) can then utilize this comprehensive set of inputs (e.g., via concatenation followed by projection, or attention-like mechanisms) to compute the refined representation `v_{i+1}^{out}$ (tangent space output) for that level.

#### 3.2.11 Scale-Aware Aggregation
To produce a final output for a downstream task, information from multiple levels of the WuBu Nesting hierarchy often needs to be aggregated.
*   **Mechanism:** Representations from different levels (e.g., the output tangent vectors `v_i^{out}$ from each level) are collected.
*   **Strategies:** Since tangent space operations are central, aggregation often happens in tangent space:
    *   **Concatenation:** Concatenate the output tangent vectors `v_1^{out}, v_2^{out}, ..., v_L^{out}$ from all `L` levels.
    *   **Pooling:** Apply max or mean pooling across the tangent vectors from different levels (assuming compatible dimensions or after projection).
    *   **Attention:** Use an attention mechanism where vectors from different levels attend to each other.
*   **Final Projection:** The aggregated tangent space representation is then projected to the final output dimension using a standard layer (e.g., MLP or Linear).

## 4. Mathematical Formulation (Conceptual)

Let's outline the conceptual mathematical flow for a single step from level `i$ to level `i+1`.

**Inputs to Level `i` Processing:**
*   Primary tangent vector from previous transition: `v_i^{in} ∈ T_o(H^n_i)`.
*   Aggregated relative tangent vectors from previous transition: `d_i^{agg} ∈ T_o(H^n_i)`.
*   Transformed level descriptor from previous level: `ld_i^{in} ∈ T_o(H^n_i)`.
*   Contextual spread parameter from previous level: `σ_{i-1} ∈ R^+`.

**Parameters specific to Level `i$:**
*   Learnable curvature `c_i > 0`, Scale `s_i > 0`.
*   Learnable boundary tangent vectors `{v_{b_{i,j,k}}^{param}} ⊂ T_o(H^n_i)`.
*   Learnable Level Descriptor Vector `ld_i^{param} ∈ T_o(H^n_i)`.
*   Learnable Level Spread Parameter `σ_i > 0`.
*   Intra-Level Tangent Flow function `F_i: T_o(H^n_i) → T_o(H^n_i)`.
*   Intra-Level Processing Module `Proc_i` (e.g., MLP combiner).

**A. Intra-Level Processing within Level `i$:**
1.  **Combine Inputs (Tangent Space):** Combine the input tangent vectors using `Proc_i`:
    ```math
    v_{combined} = \text{Proc}_i(v_i^{in}, d_i^{agg}, ld_i^{in}, \sigma_{i-1})
    ```
2.  **Apply Intra-Level Flow (Tangent Space):** Apply the flow `F_i` to the combined vector:
    ```math
    v_{flowed} = v_{combined} + F_i(v_{combined}) \quad \text{(e.g., additive flow)}
    ```
3.  **Determine Output Tangent Vector:** The flowed vector is the tangent space output for this level:
    ```math
    v_i^{out} = v_{flowed} ∈ T_o(H^n_i)
    ```
    *(Note: For operations requiring hyperbolic space, intermediate Exp/Log maps might be used within Proc_i or F_i, but the final output here is tangent)*

**B. Inter-Level Transition (`i → i+1`):**
1.  **Retrieve Tangent Vectors from Level `i`:**
    *   Primary Output: `v_i^{out}` (from step A.3).
    *   Boundary Points (Parameters): `v_{b_{i,j,k}} = v_{b_{i,j,k}}^{param}`.
    *   Level Descriptor (Parameter): `ld_i = ld_i^{param}`.

2.  **Apply Learned Rotation `R_i: T_o(H^n_i) → T_o(H^n_i)`:**
    *   `v'^{out}_i = R_i(v_i^{out})`
    *   `v'_{b_{i,j,k}} = R_i(v_{b_{i,j,k}})`
    *   `ld'_i = R_i(ld_i)`

3.  **Apply Mapping Transform `T̃_i: T_o(H^n_i) → T_o(H^n_{i+1})`:**
    *   `v_{i+1} = T̃_i(v'^{out}_i)`
    *   `v''_{b_{i,j,k}} = T̃_i(v'_{b_{i,j,k}})`
    *   `ld_{i+1} = T̃_i(ld'_i)`
    (These vectors are now in the target tangent space `T_o(H^n_{i+1})`).

4.  **Generate and Aggregate Relative Vectors in `T_o(H^n_{i+1})`:**
    *   Compute individual vectors: `d_{i+1, j, k} = v_{i+1} - v''_{b_{i,j,k}}`
    *   Aggregate (e.g., mean): `d_{i+1}^{agg} = \text{Aggregate}(\{d_{i+1, j, k}\})`

5.  **Gather Inputs for Level `i+1` Processing:** The inputs passed to the next level's processing module `Proc_{i+1}$ are:
    *   `v_{i+1}` (Primary tangent vector for level `i+1`).
    *   `d_{i+1}^{agg}` (Aggregated relative vectors).
    *   `ld_{i+1}` (Transformed level descriptor).
    *   `σ_i` (Spread parameter *from level `i$*).

This process repeats for the transition from level `i+1` to `i+2`, and so on. The final output is generated by aggregating the tangent outputs `v_1^{out}, v_2^{out}, ...` from relevant levels.

*(Note: This formulation emphasizes tangent space operations for transitions and internal processing, minimizing hyperbolic mapping steps to potentially address the user's goal of reducing conversions, though Exp/Log maps are still essential for boundary point interpretation and potential internal hyperbolic ops.)*

## 5. Potential Applications

The comprehensive nature of the WuBu Nesting framework, integrating multi-scale hierarchy, rotation, relative geometry, level-specific characteristics, and dynamics, makes it potentially suitable for a wide range of complex modeling tasks:

*   **Computer Vision:**
    *   **Articulated Object Understanding:** Modeling complex objects like humans or animals, where nested part hierarchies (limbs, digits) combine with rotational joint movements (handled by `R_i`) and potentially part-specific dynamics (modeled by `F_i`). Boundary manifolds (`B_{i,j}$) could represent keypoints or parts, relative vectors (`d`) their configuration. Level descriptors (`ld_i`) could capture part symmetry or orientation bias.
    *   **Scene Analysis with Viewpoint Changes:** Representing scenes with nested object structures where viewpoint transformations involve rotations (`R_i`) applied across scales. Spread (`σ_i`) could model ambiguity or scale uncertainty.
    *   **Robotic Vision & Interaction:** Representing robot configurations (hierarchy of links/joints) and their interaction with complex, structured environments, involving both physical rotations and potential dynamic adjustments (`F_i`). Level descriptors might represent tool orientation.

*   **Molecular Biology & Cheminformatics:**
    *   **Protein Structure & Function:** Modeling proteins with hierarchical structures (domains, secondary structures, residues). Rotations (`R_i`) are crucial for conformational changes and docking. Boundary manifolds could represent active sites or key residues. Relative vectors can capture precise spatial arrangements. Level descriptors might encode chirality or domain orientation. Spread could model flexibility or ensemble variation. Tangent flow could model local folding dynamics.
    *   **Drug Discovery & Docking:** Representing molecules and protein pockets hierarchically, using rotations for alignment scoring. Spread (`σ_i`) could model docking pose uncertainty.

*   **Robotics & Control:**
    *   **Hierarchical Planning & Control:** Representing complex tasks decomposed into sub-tasks at different scales (nesting). Physical robot movements involve rotations (`R_i`). Intra-level flows (`F_i`) could model local trajectory refinements or impedance control adaptations. Level descriptors might represent tool orientation.
    *   **State Representation:** Encoding complex robot states (e.g., manipulators with complex grippers) and their interaction with the environment.

*   **Knowledge Graph Representation:**
    *   **Complex Ontologies:** Embedding knowledge graphs with deep hierarchical category structures (nesting) and potentially relational orientations or types (captured partly by `R_i` or `ld_i`). Boundary manifolds could represent salient entity types within a hierarchy level.

*   **Generative Models:**
    *   **Structured Data Generation:** Creating complex, structured data like 3D shapes, molecules, or scenes with inherent hierarchical consistency, controlled orientations, and potentially learned dynamic variations (`F_i`). The adaptive geometry (`c_i, s_i$) could allow generation of structures with varying complexity.

*   **Time Series Analysis:**
    *   **Hierarchical Processes:** Modeling time series with multi-scale temporal patterns where dynamics (`F_i`) and state transitions (`T_{i → i+1}`) are key. Rotation might model phase shifts or periodic components.

## 6. Implementation Considerations & Strategy

Implementing the full WuBu Nesting framework presents significant technical challenges, demanding careful attention to mathematical rigor, numerical stability, and computational efficiency.

*   **Mathematical Rigor:**
    *   **Scale-Aware Maps/Metrics:** Formal derivation and implementation of consistent, differentiable scale-aware exponential maps, logarithmic maps, and associated hyperbolic metrics are required. (As attempted in the provided Python code).
    *   **Tangent Space Consistency:** Ensuring reference points for tangent spaces are handled consistently (using the origin `o` simplifies this).
    *   **Rotation Parameterization:** Choosing and implementing stable, differentiable parameterizations for `SO(n_i)` matrices (e.g., using matrix exponential from skew-symmetric matrices, Cayley maps) or unit quaternions.
    *   **Flow Parameterization:** Defining suitable parameterizations for the intra-level tangent flows (`F_i`).

*   **Numerical Stability:**
    *   **Hyperbolic Operations:** Even if minimized, any use of LogMap/ExpMap requires robust handling near boundaries or for large norms (clipping, precision, stable gradients).
    *   **Tangent Space Operations:** While Euclidean, deep stacks of transformations (`R_i`, `T̃_i`, `F_i`) can still cause gradient issues. LayerNorm, residual connections, and gradient clipping are essential.
    *   **Curvature/Scale/Spread Learning:** Keeping `c_i, s_i, σ_i` positive and bounded requires careful parameterization (e.g., `softplus(param) + min_val`).

*   **Computational Cost:**
    *   **Multiple Levels & Components:** Each level adds layers (Rotation, Mapping, Processing, Flow). Boundary points increase vector processing load.
    *   **Complex Transformations:** `SO(n_i)` matrix exponentiation or complex MLPs for `T̃_i`/`F_i` can be costly.

*   **Component Design & Interaction:**
    *   **Boundary Representation/Aggregation:** Deciding how many boundary points and how to aggregate their relative vectors is crucial.
    *   **Flow Field Design:** Balancing expressivity and stability/cost of `F_i`.
    *   **Information Fusion:** Designing `Proc_i` to effectively use the diverse tangent space inputs.

*   **Optimization:**
    *   **Complex Loss Landscape:** Requires careful initialization, learning rate scheduling, and potentially advanced optimizers (though standard Adam/SGD with correct parameterization might work).
    *   **Regularization:** Needed to prevent extreme geometries or exploding parameters (weight decay, dropout, potential geometric regularizers).

**Incremental Implementation Strategy:** (As reflected in the Python code versions)
1.  **Foundation (2 Levels, Basic Transition):** Implement two levels with fixed geometry. Stable Log/Exp Maps. Tangent transition with basic Rotation (`R_1`) and Mapping (`T̃_1`). (Similar to initial code structure).
2.  **Add Boundaries & Relative Vectors:** Introduce learnable boundary tangent vectors and compute/aggregate relative vectors. Pass `d_2` to `Proc_2`.
3.  **Add Descriptors & Spread:** Implement learnable `ld_i` and `σ_i`. Pass transformed `ld_{i+1}$ and `σ_i` to `Proc_{i+1}$.
4.  **Adaptive Geometry:** Make `c_i`, `s_i$ learnable using stable parameterizations. Implement scale-aware maps.
5.  **Intra-Level Flow:** Add the tangent flow module `F_i` within `Proc_i`.
6.  **Multi-Level & Aggregation:** Extend to more levels and implement final aggregation.
7.  **Refinement & Optimization:** Tune all components.

## 7. Discussion & Future Work

WuBu Nesting, as presented, is a highly ambitious conceptual framework aiming to unify multiple desirable geometric properties within a single deep learning architecture. Its potential lies in providing a much richer and more flexible inductive bias than currently available methods, potentially leading to breakthroughs in modeling complex systems where hierarchy, orientation, scale, dynamics, and uncertainty are all crucial aspects. By emphasizing tangent space operations for transitions and internal processing, it attempts to reduce reliance on direct hyperbolic computations, potentially aligning with the goal of fewer conversions while retaining geometric structure.

**Potential Impact:**
*   **Unified Geometric Modeling:** Offers a path towards a single model architecture capable of handling diverse geometric complexities simultaneously.
*   **Improved Representation Learning:** The explicit modeling of these geometric features could lead to more robust, interpretable, and efficient representations, particularly for structured data.
*   **New Modeling Capabilities:** Enables tackling problems previously difficult due to the lack of suitable geometric biases (e.g., complex protein dynamics, fine-grained articulated object interaction).
*   **Interpretability:** Components like boundary manifolds, relative vectors, level descriptors, and spread parameters might offer more interpretable insights.

**Limitations & Challenges:**
*   **Complexity & Cost:** The framework remains significantly complex and computationally demanding.
*   **Training Stability:** Ensuring stable training across all components (especially learnable geometries and rotations) is paramount.
*   **Data Requirements:** May require large, rich datasets to learn meaningful geometric parameters.
*   **Theoretical Analysis:** Requires further investigation into stability, expressivity, and convergence.

**Future Work:**
1.  **Formal Mathematical Development:** Rigorous derivation and analysis of scale-aware maps, rotation parameterizations, and flow stability.
2.  **Robust Implementation & Benchmarking:** Continued development of stable code (like the provided Python example) and rigorous evaluation on suitable benchmarks.
3.  **Component Ablation & Variations:** Systematically study the impact of each component (boundaries, descriptors, flow, etc.). Explore alternative implementations (e.g., attention for relative vectors).
4.  **Optimization Strategies:** Investigate specialized optimizers or learning rate schedules.
5.  **Theoretical Analysis:** Deeper investigation into the framework's properties.
6.  **Visualization Tools:** Develop tools to visualize the learned nested structures and transformations.

## 8. Conclusion

WuBu Nesting is presented as a novel, comprehensive conceptual framework for deep geometric learning. By uniquely integrating **adaptively nested hyperbolic spaces**, **explicit boundary sub-manifolds**, **tangent space rotations and mappings**, **relative vector computations**, **learnable level descriptors**, **contextual level spread parameters**, and **intra-level tangent flows**, it aims to provide an unprecedentedly rich geometric inductive bias. The framework emphasizes tangent space operations for complex transformations, potentially reducing Euclidean/non-Euclidean conversions while capturing the interplay of multi-scale hierarchy, orientation, relative structure, scale-specific characteristics, density/uncertainty, and local dynamics. While significant challenges remain, WuBu Nesting offers a promising direction for developing next-generation deep learning models capable of understanding and generating data with profound geometric complexity.

## References

*(References are based on the initial list and standard GDL/Hyperbolic/Quaternion literature)*

[1] Atigh, M. G., Schoep, J., Acar, E., Van Noord, N., & Mettes, P. (2022). Hyperbolic image segmentation. *CVPR*.
[4] Becigneul, G., & Ganea, O. E. (2019). Riemannian adaptive optimization methods. *ICLR*.
[15] Ermolov, A., Mirvakhabova, L., Khrulkov, V., Sebe, N., & Oseledets, I. (2022). Hyperbolic vision transformers: Combining improvements in metric learning. *CVPR*.
[19] Ganea, O., Bécigneul, G., & Hofmann, T. (2018). Hyperbolic neural networks. *NeurIPS*.
[22] Gulcehre, C., Denil, M., Malinowski, M., Razavi, A., Pascanu, R., Hermann, K. M., ... & de Freitas, N. (2019). Hyperbolic attention networks. *ICLR*.
[31] Khrulkov, V., Mirvakhabova, L., Ustinova, E., Oseledets, I., & Lempitsky, V. (2020). Hyperbolic image embeddings. *CVPR*.
[39] Nickel, M., & Kiela, D. (2017). Poincaré embeddings for learning hierarchical representations. *NeurIPS*.
[42] Liu, Y., He, Z., & Han, K. (2025). Hyperbolic Category Discovery. *arXiv preprint arXiv:2504.06120*. *(Note: Placeholder date/ID)*
[43] Hamilton, W. R. (1866). *Elements of quaternions*. Longmans, Green, & Company.
[44] Parcollet, T., Morchid, M., Bousquet, P. M., Dufour, R., Linarès, G., & De Mori, R. (2019). Quaternion recurrent neural networks. *ICLR*.
[45] Grassucci, E., Comminiello, D., & Uncini, A. (2021). Quaternion neural networks: State-of-the-art and research challenges. *IEEE Transactions on Neural Networks and Learning Systems*.
[46] Gu, A., Sala, F., Gunel, B., & Ré, C. (2019). Learning Mixed-Curvature Representations in Product Spaces. *ICLR 2019*. (Corrected Conf/Ref)
[50] Ungar, A. A. (2008). *Gyrovector spaces and gyrovector space theory*. Springer.
[51] Nickel, M., & Kiela, D. (2018). Learning continuous hierarchies in the Lorentz model of hyperbolic geometry. *ICML*.
[52] Tifrea, A., Bécigneul, G., & Ganea, O. E. (2019). Poincaré Glove: Hyperbolic word embeddings. *ICLR*.
[53] Chami, I., Ying, R., Ré, C., & Leskovec, J. (2019). Hyperbolic graph convolutional neural networks. *NeurIPS*.
[54] Zhang, Y., Wang, X., Shi, C., & Ye, Y. (2021). Hyperbolic graph attention network. *WWW*.
[55] Liu, Y., Wang, M., Long, M., & Yu, F. (2022). Fully Hyperbolic Neural Networks. *CVPR*. (Updated Ref)
[56] Skopek, O., Ganea, O. E., & Bécigneul, G. (2020). Mixed-curvature variational autoencoders. *ICLR*.
[57] Guo, W., Chen, Z., & Chang, B. (2021). Learning mixed-curvature representations in products of model spaces. *arXiv preprint arXiv:2110.10119*.
[58] Zhou, Y., et al. (2019). On the continuity of rotation representations in neural networks. *CVPR*.
[59] Weiler, M., Hamprecht, F. A., & Storath, M. (2018). Learning steerable filters for rotation equivariant CNNs. *CVPR*.
[60] Cohen, T., & Welling, M. (2016). Group equivariant convolutional networks. *ICML*.
[61] Falorsi, L., et al. (2020). Explorations in geometric deep learning. *arXiv preprint arXiv:2007.07203*.
[62] Lezama, J., Qiu, Q., & Sapiro, G. (2017). Riemannian stochastic optimization methods for non-convex matrix completion. *arXiv preprint arXiv:1701.00306*.
[63] Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. *NeurIPS*.
[64] Grathwohl, W., et al. (2018). FFJORD: Free-form continuous dynamics for scalable reversible generative models. *ICLR*.
[65] Dupont, E., Doucet, A., & Teh, Y. W. (2019). Augmented neural ODEs. *NeurIPS*.
[66] Mathieu, E., Le Lan, C., Maddison, C. J., Tomioka, R., & Teh, Y. W. (2020). Riemannian continuous normalizing flows. *NeurIPS*.
[67] Davidson, T. R., Falorsi, L., De Cao, N., Kipf, T., & Tomczak, J. M. (2018). Hyperspherical variational auto-encoders. *NeurIPS*.
[68] Skopek, O., et al. (2021). Geometric bayesian deep learning: A review. *arXiv preprint arXiv:2107.08957*.
[69] Bronstein, M. M., Bruna, J., LeCun, Y., Szlam, A., & Vandergheynst, P. (2017). Geometric deep learning: going beyond euclidean data. *IEEE Signal Processing Magazine*.
[70] Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. *arXiv preprint arXiv:2104.13470*.
[71] Kochurov, M., et al. (2020). Geoopt: Riemannian Optimization in PyTorch. *GitHub Repository*. `https://github.com/geoopt/geoopt`

