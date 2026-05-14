# WuBu Nesting: A Comprehensive Geometric Framework for Adaptive Multi-Scale Hierarchical Representation with Integrated Rotational Dynamics

- **Author:** WaefreBeorn
- **Subject:** Geometric Deep Learning, Hyperbolic Geometry, Machine Learning
- **Keywords:** Hyperbolic Geometry, Geometric Deep Learning, Nested Models, Rotation, Tangent Space, Hierarchy, Representation Learning
- **Pages:** 10

---

## Page 1

## WuBu Nesting: A Comprehensive Geometric Framework for  
## Adaptive Multi-Scale Hierarchical Representation with Integrated  
## Rotational Dynamics

Wubu WaefreBeorn∗

**Abstract**

Real-world data frequently exhibits complex characteristics including multi-scale hierarchi-  
cal organization, rotational symmetries, dynamic evolution, and regional uncertainty. While  
geometric deep learning offers powerful tools, existing paradigms often specialize: Euclidean  
models struggle with hierarchies, hyperbolic models typically handle single static hierarchies  
without rotational mechanics, and quaternion networks excel at rotation but lack hierarchical  
structure. To address these limitations, we introduce WuBu Nesting ("layered nesting"), a  
novel conceptual framework unifying these geometric properties. WuBu Nesting employs a re-  
cursively nested structure of hyperbolic spaces (Hn1  
c1,s1 ⊃Hn2  
c2,s2 ⊃. . . ) where dimensionality (ni),  
curvature (ci > 0), and scale (si > 0) are learnable, allowing dynamic adaptation to data com-  
plexity. Each level i incorporates learnable Boundary Sub-Manifolds representing scale-specific  
structures. Inter-level transitions (i →i + 1) occur in Euclidean tangent spaces (Tp(Hni  
, ) ∼= Rni),  
featuring a learned Rotation (Ri, e.g., SO(ni) or Quaternions) applied simultaneously to primary  
data, boundary representations, and a learnable Level Descriptor (ldi). This is followed by a  
non-rotational Mapping (T ˜i) adjusting features and dimension. Relative Vectors (di+1,j,k) are  
computed in the target tangent space, encoding rotation-aware structure. Each level also has a  
learnable Spread Parameter (σi) for uncertainty context and allows Intra-Level Tangent Flow  
(Fi) for local dynamics. The resulting rich information flow enables WuBu Nesting to capture  
intertwined hierarchical, rotational, dynamic, and uncertain characteristics, offering a highly  
flexible geometric inductive bias for complex data modeling.

### 1  
### Introduction

Effective data representation is fundamental to machine learning. While standard deep learning  
models achieve remarkable success, their predominant reliance on Euclidean geometry limits their  
ability to capture intrinsic data structures not naturally suited to flat spaces. Hierarchical data—  
taxonomies, phylogenetic trees, molecular structures, articulated objects, parse trees—presents a  
key challenge. Embedding hierarchies in Euclidean space incurs significant distortion due to the  
mismatch between Euclidean polynomial volume growth and the typically exponential expansion of  
hierarchical nodes [15].  
Hyperbolic geometry, with its constant negative curvature and exponential volume growth,  
provides a more natural embedding space for such structures [5, 11, 15]. Models utilizing the Poincaré  
ball (Hn) or other hyperbolic models have shown substantial advantages in graph embedding, NLP  
[6, 9], computer vision [1, 5, 11], and category discovery [13], demonstrating the benefit of aligning  
geometric inductive bias with data structure.

∗Wubu WaefreBeorn, X: @WaefreBeorn Twitch: @WaefreBeorn Youtube: @WuBuStreams @WaefreBeorn

1

---

## Page 2

However, real-world systems often exhibit complexities beyond single, static hierarchies. Data  
frequently possesses nested hierarchies (structures within structures) and involves components  
with intrinsic orientations where transformations between levels or viewpoints include rotations.  
For instance, modeling articulated objects requires understanding both part hierarchies and their  
relative rotational movements. Existing hyperbolic models typically focus on a single hierarchy level  
with fixed geometry, lacking mechanisms for adaptive multi-scale nesting or integrated rotational  
modeling.  
Conversely, Quaternions [10] offer efficient rotation representation, leveraged by Quaternion  
Neural Networks (QNNs) [7, 16] for tasks involving orientation. However, QNNs operate in Euclidean  
spaces, lacking intrinsic hierarchical embedding capabilities. Product manifolds (Rn × Sm × Hk) [8]  
combine geometries in parallel but do not directly model nested structures or integrated inter-level  
rotational transformations.  
This paper introduces WuBu Nesting, a comprehensive conceptual framework designed to  
bridge these gaps by unifying adaptive multi-scale hierarchical representation with explicit modeling  
of rotational dynamics, dynamic evolution, and regional uncertainty. Instead of a single space  
or parallel product, WuBu Nesting proposes a nested "Russian doll" architecture of recursively  
embedded hyperbolic manifolds. Key innovations include:

1) Adaptive Nested Hyperbolic Geometry: A sequence of nested hyperbolic spaces Hn1  
c1,s1 ⊃  
Hn2  
c2,s2 ⊃. . . , where dimensionality ni, curvature ci > 0, and scale si > 0 are learnable, allowing  
dynamic geometric adaptation.

2) Boundary Sub-Manifolds: Learnable manifolds Bi,j within each level Hni  
,  
(e.g., parameterized  
by points {bi,j,k}) representing scale-specific substructures or landmarks.

3) Tangent Space Transitions: Inter-level transitions (i →i + 1) mediated within Euclidean  
tangent spaces Tp(Hni  
, ) ∼= Rni, enabling complex yet tractable transformations.

4) Explicit Tangent Space Rotations (Ri): A learnable rotation Ri (e.g., SO(ni) or Quaternions)  
applied within To(Hni  
, ).

5) Simultaneous Transformation: Rotation Ri applied consistently to the main tangent vector  
vi, boundary tangent vectors vbi,j,k, and a learnable Level Descriptor Vector ldi.

6) Non-Rotational Mapping (T ˜i): A learnable mapping T˜i : To(Hni  
, ) →To(Hni+1  
,  
) following  
rotation, handling feature transformation and dimension changes. Full tangent transform: Ti→i+1 =  
T˜i ◦Ri.

7) Relative Vector Generation (di+1): Computed in the target tangent space To(Hni+1  
,  
) as  
di+1,j,k = vi+1 −v′′  
bi,j,k, encoding rotation-aware structure.

8) Learnable Level Descriptor Vector (ldi): An intrinsic vector ldi ∈To(Hni  
, ) capturing  
level-specific geometric properties, transformed to ldi+1 for the next level.

9) Learnable Level Spread Parameter (σi): A scalar σi > 0 representing scale-specific  
uncertainty or density, passed as context to level i + 1.

10) Intra-Level Tangent Flow (Fi): A learnable field Fi : To(Hni  
, ) →To(Hni  
, ) modeling dynamics  
or adjustments within level i.

11) Rich Hierarchical Information Flow: Level i+1 processing uses the primary representation,  
relative vectors {di+1,j,k}, transformed descriptor ldi+1, and contextual spread σi.

2

---

## Page 3

We posit that this integrated geometric structure provides a powerful and flexible inductive bias  
for modeling complex real-world systems exhibiting intertwined hierarchical, rotational, dynamic,  
and uncertain characteristics.

### 2  
### Related Work

The WuBu Nesting framework builds upon and synthesizes concepts from several areas of geometric  
deep learning.

2.1  
**Hyperbolic Deep Learning**

Pioneered by Nickel and Kiela [15], hyperbolic geometry offers superior embedding for hierarchical  
data compared to Euclidean spaces due to its negative curvature and exponential volume growth.  
Subsequent work extended this to tree embeddings [11], graph embeddings [3, 5, 18], ontologies [1],  
hyperbolic neural network operations [6, 9], and applications in computer vision [1, 5, 11, 13].  
Critique & WuBu Distinction: Existing methods typically use a single, fixed-curvature  
hyperbolic space. They lack mechanisms for adaptive nested hierarchies and explicit rotational  
modeling. WuBu Nesting introduces learnable, nested geometries (ni, ci, si), boundary manifolds,  
level descriptors, spread parameters, tangent flows, and integrated rotation-aware tangent space  
transitions.

2.2  
**Quaternion Neural Networks (QNNs)**

QNNs [7, 16] leverage the efficiency of quaternions [10] for representing 3D/4D rotations, achieving  
parameter efficiency and respecting rotational symmetries in tasks like 3D vision and robotics.  
Critique & WuBu Distinction: QNNs operate in Euclidean spaces and lack intrinsic  
hierarchical embedding capabilities. WuBu Nesting integrates rotational modeling (potentially via  
quaternions) within tangent space transitions between nested hyperbolic levels, combining rotational  
awareness with adaptive hierarchy.

2.3  
**Product Manifolds and Multi-Scale Approaches**

Product manifolds [8] (e.g., Rn × Sm × Hk) combine different geometries in parallel, increasing  
capacity but lacking nested structure and sophisticated inter-geometry transformations. Traditional  
multi-scale methods (e.g., feature pyramids) operate in Euclidean space without specific geometric  
biases for hierarchy or rotation.  
Critique & WuBu Distinction: WuBu Nesting proposes a fundamentally different re-  
cursive embedding architecture, not parallel composition. Its inter-level transitions are designed  
to be geometrically meaningful, incorporating learned rotations and mappings, unlike the simpler  
aggregation often used in product spaces. It integrates hierarchy, scale, rotation, dynamics, and  
uncertainty in a unified framework distinct from standard multi-scale techniques.

### 3  
### The WuBu Nesting Framework

WuBu Nesting provides a recursive, multi-level geometric architecture. Data flows through nested  
hyperbolic spaces, with inter-level transitions orchestrated in tangent spaces, incorporating rotations,  
mappings, relative vector generation, and propagation of level-specific context (descriptors, spread).

3

---

## Page 4

3.1  
**Conceptual Architecture**

Input data is initially encoded and mapped to the tangent space of the outermost level (Hn1  
c1,s1).  
Within level i, processing occurs, potentially including intra-level tangent flow Fi. For transition  
i →i + 1, the primary representation, boundary manifold representations {bi,j,k}, and level  
descriptor ldi are mapped to tangent vectors (vout  
i  
, {vbi,j,k}, ldparam  
i  
) via Logco,sii. A rotation Ri is  
applied simultaneously to these vectors in To(Hni  
, ). A mapping T˜i then transforms the rotated  
vectors into the target tangent space To(Hni+1  
,  
), yielding vi+1, {v′′  
bi,j,k}, ldi+1. Relative vectors  
di+1,j,k = vi+1 −v′′  
bi,j,k are computed. Level i + 1 processing uses vi+1 (often mapped to xi+1  
via expci+1  
o,si+1), relative vectors {di+1,j,k}, descriptor ldi+1, and spread σi from level i. Aggregated  
information from relevant levels forms the final output. (See Figure 1).

3.2  
**Component Details**

We elaborate on the framework’s components.

**3.2.1**  
Nested Hyperbolic Spaces & Adaptive Geometry (Hni  
ci,si, ni, ci, si)

The structure comprises nested Poincaré Balls Hni  
ci,si.

• Nesting: Allows multi-scale hierarchical modeling (Hn1  
,  
⊃Hn2  
,  
⊃. . . ).

• Dimensionality (ni): Learnable or fixed dimension per level, adapting capacity.

• Curvature (ci > 0): Learnable parameter controlling geometry steepness. Requires constrained  
optimization.

• Scale (si > 0): Learnable parameter modulating tangent space mapping (Eq. 1), controlling  
density/zoom. Requires constrained optimization.

expco,sii(v) = tanh  
�  
si ·  
√ci∥v∥

�  
v  
√ci∥v∥  
(1)

2

**3.2.2**  
Boundary Sub-Manifolds (Bi,j)

Learnable manifolds within Hni  
, , often parameterized by points {bi,j,k}, representing scale-specific  
landmarks or substructures. Mapped to tangent space for transformations.

**3.2.3**  
Tangent Space Logic (Tp(Hni  
, ) ∼= Rni)

Complex transformations occur in Euclidean tangent spaces Tp(Hni  
, ) ∼= Rni, using Logarithmic  
(Logcp,sii) and Exponential (expcp,sii) maps [12] for transitions between hyperbolic and tangent spaces.

**3.2.4**  
Tangent Space Rotations (Ri)

Learnable rotation Ri applied in To(Hni  
, ) during transition i →i + 1.  
Implemented via unit  
quaternions (if ni = 4) or SO(ni) matrices (parameterized via exponentiation, Cayley maps, etc.  
[14]). Applied simultaneously to main vector vi, boundary vectors {vbi,j,k}, and descriptor ldi.

**3.2.5**  
Non-Rotational Mapping (T ˜i)

Learnable mapping T˜i : To(Hni  
, ) →To(Hni+1  
,  
) applied after rotation Ri. Handles feature transfor-  
mation and dimension change using MLPs, linear layers, etc.

4

---

## Page 5

Input Data

Initial  
Euclidean  
Encoding

ld1  
σ1  
B1

Level 1  
Processing  
(incl. F1)

Map to  
To(Hn1)

LogMap  
to To(Hn1)  
Rotation R1

(vout  
1  
, vb1 , ldparam  
1  
)

Mapping T˜1

ld2  
σ2  
B2

(v2, v′′  
b1 , ld2)

Level 2  
Processing  
(incl. F2)

Target  
To(Hn2)

{d2}

RelVecs d2

LogMap  
to To(Hn2)  
Rotation R2

(vout  
2  
, vb2 , ldparam  
2  
)

Mapping T˜2

(v3, v′′  
b2 , ld3)

Level 3  
Processing  
(incl. F3)

Target  
To(Hn3)

{d3}

RelVecs d3

LogMap  
to To(Hn3)

Aggregate  
Information

Final Pro-  
jection

Output

Figure 1: Conceptual Architecture of the Comprehensive WuBu Nesting Framework (TikZ Diagram).  
Illustrates data flow through nested hyperbolic levels (Hni  
ci,si) with adaptive parameters (ni, ci, si).  
Key components: Boundary Manifolds (Bi), Level Descriptors (ldi), Level Spreads (σi), Intra-Level  
Tangent Flows (Fi). Inter-level transitions use tangent space mapping (LogMap), simultaneous  
Rotation (Ri), and Mapping (T ˜i). Relative Vectors (di+1) are computed. Level i + 1 uses vi+1,  
di+1, ldi+1, and σi. Aggregated information yields the final output.

5

---

## Page 6

**3.2.6**  
Relative Vector Generation (di+1,j,k)

Computed in target tangent space To(Hni+1  
,  
) after full transformation Ti→i+1 = T˜i ◦Ri:

di+1,j,k = vi+1 −v′′  
bi,j,k  
(2)

Encodes rotation-aware structure relative to boundaries. Used as input for level i + 1.

**3.2.7**  
Learnable Level Descriptor Vector (ldi)

Learnable vector ldi ∈To(Hni  
, ) capturing intrinsic level properties (e.g., anisotropy). Transformed  
via Ri and T˜i to ldi+1 and passed as input to level i + 1.

**3.2.8**  
Learnable Level Spread Parameter (σi)

Learnable positive scalar σi > 0 representing scale-specific uncertainty or density. Passed as context  
to level i + 1. Requires constrained optimization.

**3.2.9**  
Intra-Level Tangent Flow (Fi)

Learnable function Fi : To(Hni  
, ) →To(Hni  
, ) applied during level i processing (e.g., vflowed =  
v + MLPi(v) or vflowed = Miv). Models local dynamics or adjustments. Can use Neural ODEs [4].

**3.2.10**  
**Hierarchical Information Flow**

Level i + 1 processing module receives primary vector vi+1 (or xi+1), relative vectors {di+1,j,k},  
descriptor ldi+1, and spread σi. Enables decisions based on position, relative structure, orientation,  
level characteristics, and source uncertainty.

**3.2.11**  
**Scale-Aware Aggregation**

Information from multiple levels (vout  
i  
, di,j,k, ldi, etc.) is aggregated for final prediction. Requires  
mapping representations to a common space (e.g., outermost tangent space via inverse transforms,  
or a dedicated output space) followed by concatenation, attention, or pooling.

### 4  
### Mathematical Formulation (Conceptual)

We outline the conceptual flow from level i to i + 1.  
Inputs to Level i Processing: Primary tangent vector vin  
i ; Relative vectors {di,j,k}; Trans-  
formed descriptor ldin  
i ; Contextual spread σi−1.  
Parameters for Level i: Geometry ci, si; Boundary points {bi,j,k}; Descriptor ldparam  
i  
; Spread  
σi; Flow function Fi.  
A. Intra-Level Processing (Level i):

1. Map vin  
i to xin  
i = expco,sii(vin  
i ) (optional).

2. Internal module computes intermediate state using xin  
i (or vin  
i ), {di,j,k}, ldin  
i , σi−1.

3. Apply flow Fi to intermediate tangent representation vintermediate to get vflowed.

4. Generate level output state xout  
i  
∈Hni  
ci,si.

6

---

## Page 7

B. Inter-Level Transition (i →i + 1):

1. Map to Tangent Space To(Hni  
, ): vout  
i  
= Logco,sii(xout  
i  
); vbi,j,k = Logco,sii(bi,j,k); ldi = ldparam  
i  
.

2. Apply Rotation Ri: v′out  
i  
= Ri(vout  
i  
); v′  
bi,j,k = Ri(vbi,j,k); ld′  
i = Ri(ldi).

3. Apply Mapping T˜i: vi+1 = T˜i(v′out  
i  
); v′′  
bi,j,k = T˜i(v′  
bi,j,k); ldi+1 = T˜i(ld′  
i). (Vectors now in  
To(Hni+1  
,  
)).

4. Generate Relative Vectors: di+1,j,k = vi+1 −v′′  
bi,j,k.

5. Gather Inputs for Level i + 1: vi+1, {di+1,j,k}, ldi+1, σi.

This process repeats recursively.

### 5  
### Potential Experiments and Evaluation

While this paper introduces the WuBu Nesting framework conceptually, rigorous empirical  
validation is crucial future work. Evaluation should target datasets exhibiting the complex geometric  
properties the framework is designed to capture. Potential experimental directions include:

• Hierarchical Classification/Regression: Tasks involving natural hierarchies (e.g., WordNet  
noun hierarchy embedding [15], biological taxonomies, molecular property prediction based on  
functional group hierarchies). Evaluate against standard Euclidean, single-level hyperbolic  
models (e.g., [6]), and product manifold models [8]. Metrics: Accuracy, Mean Average Precision  
(MAP), distortion measures, parameter efficiency.

• Articulated Object Pose Estimation/Reconstruction: Datasets like Human3.6M or  
animal pose datasets. Evaluate ability to model joint rotations (Ri) and part hierarchies.  
Compare against QNNs [7] and standard pose estimation networks. Metrics: Mean Per Joint  
Position Error (MPJPE), angular errors. Assess the contribution of boundary manifolds and  
relative vectors.

• Graph Representation Learning on Hierarchical Graphs: Node classification and link  
prediction on graphs with clear hierarchical or multi-scale structure (e.g., citation networks,  
social networks, biological networks). Compare against Euclidean GNNs, Hyperbolic GNNs  
(HGCN [3], HGAT [18]), and potentially product space GNNs. Evaluate the impact of adaptive  
geometry (ci, si) and rotational components.

• Generative Modeling of Structured Data: Generating realistic 3D shapes, molecules, or  
other data with inherent hierarchical and orientational properties. Evaluate using standard  
generative metrics (e.g., FID, Inception Score adapted for the domain) and measures of  
structural validity/diversity. Compare against geometric VAEs/GANs/Flows operating in  
Euclidean, hyperbolic [17], or product spaces.

• Ablation Studies: Systematically remove or simplify components (e.g., disable rotation  
Ri, remove boundary manifolds Bi,j, fix geometry ci, si, remove flow Fi) to quantify the  
contribution of each element to overall performance on specific tasks.

7

---

## Page 8

• Visualization and Interpretability: Use visualization tools (like those in wubu_nesting_  
visualization.py) to analyze the learned geometries (ci, si), boundary positions (bi,j,k),  
level descriptors (ldi), and flows (Fi) to gain insights into how the model represents the data  
structure.

Success in these experiments would validate the hypothesis that integrating adaptive nested  
hyperbolic geometry with explicit modeling of rotations, boundaries, relative geometry, and level-  
specific context provides a significant advantage for modeling complex, structured real-world data.

### 6  
### Implementation Considerations & Strategy

Implementing WuBu Nesting involves significant challenges in mathematical rigor, numerical  
stability, and computational efficiency.

• Mathematical Rigor: Requires consistent scale-aware hyperbolic maps/metrics, stable tangent  
space handling, and differentiable parameterizations for SO(ni) rotations [14] and flows Fi.

• Numerical Stability: Hyperbolic operations near boundaries require robust implementations  
(e.g., norm clipping [13], precision).  
Tangent space operations need normalization.  
Learn-  
ing positive parameters (ci, si, σi) requires constraints (e.g., parameterizing logarithms, using  
softplus).

• Computational Cost: Multiple levels, boundary points, and complex transformations (Ri, T˜i, Fi)  
increase computational load. Efficiency is key.

• Component Design: Effective parameterization of boundaries and flows, and designing modules  
to fuse the rich inter-level information stream are critical.

• Optimization: The complex loss landscape may necessitate Riemannian optimization methods  
[2, 12], careful initialization, and tailored regularization.

Incremental Implementation Strategy: A staged approach is advised: 1) Foundational  
2-level fixed geometry with basic rotation/mapping. 2) Add adaptive geometry (ci, si). 3) Add  
boundaries and relative vectors. 4) Add level descriptors and spread. 5) Add intra-level flow. 6)  
Expand levels. 7) Refine aggregation and optimize.

### 7  
### Discussion & Future Work

WuBu Nesting is presented as an ambitious conceptual framework unifying multi-scale hierarchy,  
rotation, relative geometry, dynamics, and uncertainty within a nested hyperbolic structure.  
Potential Impact: Offers a path towards unified geometric modeling, potentially leading to  
improved representation learning, new modeling capabilities for complex systems (e.g., protein  
dynamics, articulated objects), and enhanced interpretability through its structured components.  
Limitations & Challenges: Include significant implementation complexity, high compu-  
tational cost, potential need for large structured datasets, lack of current theoretical analysis  
(stability, convergence, expressivity), and the challenge of interpreting complex interactions between  
components.  
Future Work: Key directions include: 1) Formal mathematical development. 2) Robust and  
efficient implementation (e.g., using geoopt [12]). 3) Exploring component variations (boundaries,

8

---

## Page 9

flows [4], mappings). 4) Developing tailored optimization strategies (Riemannian methods [2]).  
5) Rigorous empirical validation across diverse tasks. 6) Theoretical analysis of the framework’s  
properties. 7) Architecture search for optimal configurations. 8) Developing visualization tools for  
learned structures.

### 8  
### Conclusion

WuBu Nesting introduces a novel conceptual framework for deep geometric learning, uniquely  
integrating adaptively nested hyperbolic spaces, explicit boundary sub-manifolds, tangent space  
rotations and mappings, relative vector computations, learnable level descriptors, contextual level  
spread parameters, and intra-level tangent flows. This aims to provide an unprecedentedly rich  
geometric inductive bias suitable for capturing the complex interplay of multi-scale hierarchy,  
orientation, relative structure, scale-specific characteristics, density/uncertainty, and local dynamics  
inherent in many real-world datasets. While substantial theoretical and implementation challenges  
remain, WuBu Nesting represents a promising direction for developing next-generation deep  
learning models capable of handling profound geometric complexity, with potential applications  
across numerous scientific and engineering domains.  
Future work will focus on formalization,  
implementation, and empirical validation.

### References

[1] Mina Ghadimi Atigh, Julian Schoep, Elmira Acar, Nanne Van Noord, and Pascal Mettes.  
Hyperbolic image segmentation. In Proceedings of the IEEE/CVF Conference on Computer  
Vision and Pattern Recognition (CVPR), 2022.

[2] Gary Bécigneul and Octavian-Eugen Ganea. Riemannian adaptive optimization methods. In  
International Conference on Learning Representations (ICLR), 2019.

[3] Ines Chami, Rex Ying, Christopher Ré, and Jure Leskovec. Hyperbolic graph convolutional  
neural networks. In Advances in Neural Information Processing Systems (NeurIPS), 2019.

[4] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K. Duvenaud. Neural ordinary  
differential equations. In Advances in Neural Information Processing Systems (NeurIPS), 2018.

[5] Aleksandr Ermolov, Leyla Mirvakhabova, Valentin Khrulkov, Nicu Sebe, and Ivan Oseledets.  
Hyperbolic vision transformers: Combining improvements in metric learning. In Proceedings of  
the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[6] Octavian-Eugen Ganea, Gary Bécigneul, and Thomas Hofmann. Hyperbolic neural networks.  
In Advances in Neural Information Processing Systems (NeurIPS), 2018.

[7] Eleonora Grassucci, Danilo Comminiello, and Aurelio Uncini. Quaternion neural networks:  
State-of-the-art and research challenges. IEEE Transactions on Neural Networks and Learning  
Systems, 2021.

[8] Albert Gu, Frederic Sala, Beliz Gunel, and Christopher Ré. Learning semantic representations  
using diffusion kernels. In Advances in Neural Information Processing Systems (NeurIPS),  
2019.

9

---

## Page 10

[9] Caglar Gulcehre, Misha Denil, Mateusz Malinowski, Ali Razavi, Razvan Pascanu, Karl Moritz  
Hermann, Peter Battaglia, Victor Bapst, David Raposo, Adam Santoro, and Nando de Freitas.  
Hyperbolic attention networks. In International Conference on Learning Representations  
(ICLR), 2019.

[10] William Rowan Hamilton. Elements of quaternions. Longmans, Green, & Company, 1866.

[11] Valentin Khrulkov, Leyla Mirvakhabova, Evgeniya Ustinova, Ivan Oseledets, and Victor  
Lempitsky. Hyperbolic image embeddings. In Proceedings of the IEEE/CVF Conference on  
Computer Vision and Pattern Recognition (CVPR), 2020.

[12] Max Kochurov, Rasul Karimov, and Serge Kozlukov. Geoopt: Riemannian optimization  
in PyTorch. arXiv preprint arXiv:2005.02819, 2020. Code: https://github.com/geoopt/  
geoopt.

[13] Yuzhen Liu, Zelin He, and Keke Han.  
Hyperbolic category discovery.  
arXiv preprint  
arXiv:2504.06120, 2025. (Note: Placeholder date/ID - Update when official).

[14] Zakaria Mhammedi, Andrew Hellicar, Ashfaqur Rahman, and James Bailey. Efficient orthogonal  
parametrisation of recurrent neural networks using householder reflections. arXiv preprint  
arXiv:1705.04107, 2017.

[15] Maximilian Nickel and Douwe Kiela. Poincaré embeddings for learning hierarchical representa-  
tions. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

[16] Titouan Parcollet, Mohamed Morchid, Pierre-Michel Bousquet, Renaud Dufour, Georges Linarès,  
and Renato De Mori. Quaternion recurrent neural networks. In International Conference on  
Learning Representations (ICLR), 2019.

[17] Ondrej Skopek, Octavian-Eugen Ganea, and Gary Bécigneul. Mixed-curvature variational  
autoencoders. In International Conference on Learning Representations (ICLR), 2020.

[18] Yiding Zhang, Xiao Wang, Chuan Shi, and Yanfang Ye. Hyperbolic graph attention network.  
In Proceedings of The Web Conference (WWW), 2021.

10

---
