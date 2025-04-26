# Bytropix - Byte-Level Modeling with WuBu Nesting

An advanced byte-level language model utilizing **WuBu Nesting** for adaptive multi-scale hyperbolic geometry, integrated with Babylon Index patching and reinforcement learning optimization for next-generation language understanding and generation.

## Overview

**Bytropix** is a cutting-edge language model architecture designed to operate directly at the byte level, bypassing traditional tokenizers. It incorporates the novel **WuBu Nesting (層疊嵌套 - "layered nesting")** framework, a comprehensive geometric approach designed to capture complex multi-scale hierarchical structures, rotational dynamics, dynamic evolution, and regional uncertainty often found in real-world data[cite: 1, 3]. This implementation specifically explores a "fully hyperbolic" variant, representing data points and performing key operations within Poincaré Ball manifolds, aiming to leverage the geometric inductive biases of hyperbolic space for hierarchical data[cite: 13, 39].

Key features include:

* **Tokenizer-Free Architecture**: Processes raw UTF-8 bytes directly, removing vocabulary limitations and working natively with any language or format.
* **Babylon Index Patching**: Dynamically identifies semantically meaningful patches (like words or delimiters) in byte streams using entropy-based analysis and UTF-8 decoding attempts.
* **WuBu Nesting**: Leverages a hierarchy of nested, adaptive hyperbolic spaces ($H^{n_i}_{c_i, s_i}$) with learnable geometry (curvature $c_i > 0$, scale $s_i > 0$), boundary sub-manifolds, tangent space transformations, and relative vector computations to model complex, multi-scale structures[cite: 4, 25, 60]. Visualizations like the "Nested Spheres" plot illustrate this layered concept[cite: 103].
* **Hyperbolic Implementation (Experimental)**: This version uses custom hyperbolic layers (embeddings, linear projections via tangent space, layer normalization, attention based on hyperbolic distance) and performs operations within the Poincaré Ball model[cite: 130]. This is considered experimental and may involve numerical stability challenges.
* **Q-Learning Enhanced Optimization**: Employs a Q-learning agent (`HAKMEMQController`) within the `RiemannianEnhancedSGD` optimizer to dynamically adjust learning rate and momentum scales based on training state (loss trends, gradient norms, oscillation detection).
* **Gradient Monitoring**: Includes sophisticated gradient statistics tracking (`GradientStats`) for monitoring training stability, including norm calculation and clipping counts.

This model aims to provide a powerful and flexible inductive bias for modeling complex systems exhibiting intertwined hierarchical, rotational, dynamic, and uncertain characteristics, making it  suitable for complex language understanding, multi-lingual text processing, and other structured data domains[cite: 36, 91, 93, 96, 98].

## Architecture

The Bytropix model integrates WuBu Nesting into a byte-level sequence-to-sequence pipeline:

1.  **Input Bytes** are processed by the **Babylon Index Patching** module (`HAKMEMBabylonIndex`), segmenting the raw byte stream into variable-length patches based on information density (entropy) and text structure (words/delimiters).
2.  Patches are fed into the **Hyperbolic Local Encoder** (`HyperbolicLocalEncoder`), which maps bytes to hyperbolic embeddings within the shared Poincaré manifold, adds optional Euclidean N-gram features in the tangent space, processes them with a standard Transformer operating on *tangent vectors*, and pools the results using cross-attention into patch representations (also as tangent vectors).
3.  The resulting patch embeddings (as tangent vectors) are the input to the **Fully Hyperbolic WuBu Nesting Model** (`FullyHyperbolicWuBuNestingModel`), the core geometric processing engine.
4.  Representations pass through the **WuBu Nesting Stack**:
    * Data flows sequentially through nested levels, each associated with a Poincaré Ball manifold ($H^{n_i}$) whose curvature ($c_i$) and scale ($s_i$) can be learned[cite: 4, 60].
    * **Inter-Level Transitions** occur via the Euclidean tangent space ($T_p(H^{n_i}) \cong \mathbb{R}^{n_i}$)[cite: 5, 27, 63]. Points are mapped using Log maps (`logmap0`), transformed by a learnable mapping (`HyperbolicInterLevelTransform`), and mapped back using Exp maps (`expmap0`) to the next level's manifold (with projection)[cite: 87].
    * Learnable **Boundary Points** (`BoundaryManifoldHyperbolic`) exist within each level's manifold[cite: 5, 26, 62].
    * **Relative Vectors** are computed in the *target tangent space* based on the transformed main point and boundary points, encoding structure relative to boundaries[cite: 7, 31, 71].
    * **Intra-Level Processing** combines the input point, aggregated relative tangent vectors, a learnable **Level Descriptor** point (transformed from the previous level), and a contextual **Spread** parameter ($\sigma_i$)[cite: 8, 33, 74, 77]. This combination happens in the tangent space, is processed by a combiner network (`tangent_combiner`),  adjusted by a **Tangent Flow** (`tangent_flow`)[cite: 9, 75, 76], and then mapped back to the manifold.
    * The **Tangent Output** (`tangent_out`) from each level's processing (vector in $T_0 H_i$) is collected.
5.  Tangent outputs from all levels are **Aggregated** (e.g., via concatenation - `concat_tangent`).
6.  This aggregated tangent space representation is **Projected** by a linear layer (`tangent_to_output`) to the final **Decoder Memory** dimension (Euclidean).
7.  The **Hyperbolic Local Decoder** (`HyperbolicLocalDecoder`) takes the target byte sequence, embeds it using the shared hyperbolic manifold, maps to tangent space, adds positional encodings (tangent), attends to the Euclidean decoder memory using a standard Transformer decoder operating in tangent space, and predicts **Output Logits** using a final linear layer (optionally hierarchical).

### Q-Learning Enhanced SGD Optimizer (`RiemannianEnhancedSGD`)

The `RiemannianEnhancedSGD` optimizer supports both standard Euclidean parameters and Riemannian parameters (identified by a `.manifold` attribute, typically a `PoincareBall` instance).
* For Riemannian parameters, it converts Euclidean gradients to Riemannian gradients, performs momentum updates and update steps in the tangent space, and uses the manifold's exponential map for retraction[cite: 126].
* It incorporates the `HAKMEMQController` to adaptively tune learning rate and momentum scales based on observed training dynamics (loss, gradient norm, oscillation). The controller uses a Q-table to learn optimal scaling factors for different training states.

## Repository Structure

```

./
├── WuBuNest\_Trainer.py             \# Main training script
├── WuBuNest\_Inference.py           \# Inference/generation script
├── README.md                       \# This file
├── requirements.txt                \# Python dependencies
├── Latex\_WuBu\_Paper.pdf            \# Conceptual paper describing WuBu Nesting
├── wubu\_results/                   \# Default output directory for checkpoints & logs
│   ├── hierarchical\_data.png       \# (Example visualization from old README)
│   ├── nested\_spheres\_epoch\_\*.png  \# WuBu Nesting 3D boundary visualizations
│   ├── test\_predictions.png        \# (Example visualization from old README)
│   └── training\_metrics.png        \# Training loss plot
├── EnhancedSGD.py                  \# ( related optimizer code)
├── HypBSFIN.py                     \# (Other model/component variants)
├── HypCD.py                        \# (Other model/component variants)
├── LIVEBSFIN.py                    \# (Other model/component variants)
├── WuBuNesting.py                  \# ( core nesting logic)
├── wubu\_nesting\_impl.py            \# ( implementation details)
├── wubu\_nesting\_visualization.py   \# (Visualization generation code)
├── WuBuHypCD-paper.md              \# (Related markdown document)
├── WuBuHypCD.tex                   \# (Related LaTeX document)
├── references.bib                  \# Bibliography for papers
├── \*.bat                           \# Batch scripts for running experiments
└── ... (Other utility scripts, older versions, experimental code)

````

## Installation

```bash
# Clone the repository
git clone [https://github.com/waefrebeorn/bytropix.git](https://github.com/waefrebeorn/bytropix.git)
cd bytropix

# Create and activate virtual environment (recommended)
# Example using venv:
# python -m venv venv_wubu
# source venv_wubu/bin/activate  # On Windows use `venv_wubu\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Or, if requirements.txt is minimal/missing:
# pip install torch numpy tqdm # Add other specific dependencies if needed
````

## Requirements

Refer to `requirements.txt`. Key dependencies likely include:

  * Python 3.8+
  * PyTorch 2.0+ (CUDA highly recommended for performance)
  * NumPy
  * tqdm (for progress bars)
  * wandb (optional, for experiment tracking)
  * matplotlib (likely needed for visualizations if run locally)

## Usage

### Data Preparation

The trainer expects byte-level data stored in NumPy (`.npy`) files containing 1D arrays of `uint8` values. Prepare your text datasets accordingly.

### Training

Train the model using the `WuBuNest_Trainer.py` script. It supports DistributedDataParallel (DDP) using `torchrun`.

```bash
# --- Single GPU / CPU Training ---
python WuBuNest_Trainer.py \
    --data_path /path/to/your/train_data.npy \
    --val_data_path /path/to/your/val_data.npy \
    --checkpoint_dir ./wubu_results \
    --batch_size 16 \
    --grad_accum_steps 4 \
    --epochs 10 \
    --learning_rate 5e-4 \
    --max_grad_norm 1.0 \
    --context_window 512 \
    --num_workers 4 \
    --local_hidden_size 256 \
    --decoder_memory_dim 512 \
    --num_levels 3 \
    --hyperbolic_dims 128 64 32 \
    --initial_curvatures 1.0 0.5 0.25 \
    --boundary_points_per_level 5 4 3 \
    [--wandb] [--wandb_project WuBuHypV4] \
    [--enable-q-controller] \
    # Add other config args as needed (--dropout, --weight_decay, etc.)

# --- Multi-GPU Training (Example: 2 GPUs) ---
# torchrun --standalone --nproc_per_node=2 WuBuNest_Trainer.py [ARGS...]
```

  * Adjust `--batch_size`, `--grad_accum_steps`, learning rate, and model dimensions based on your hardware and dataset.
  * Training progress, checkpoints, logs, and visualizations will be saved in the directory specified by `--checkpoint_dir`.

### Inference (Text Generation)

Use the `WuBuNest_Inference.py` script with a trained checkpoint:

```bash
python WuBuNest_Inference.py \
    --checkpoint ./wubu_results/checkpoint_epoch_X_final*.pt \
    --seed_text "WuBu Nesting enables models to" \
    --max_length 256 \
    --temperature 0.7 \
    --repetition_penalty 1.1 \
    --top_k 40 \
    --top_p 0.9
```

  * Adjust `--max_length`, `--temperature`, `--repetition_penalty`, `--top_k`, and `--top_p` to control the generation process.

### Visualizations

The `WuBuNest_Trainer.py` script should generate visualizations during training (if corresponding functionality is enabled and matplotlib is installed) and save them to the checkpoint directory, typically under a `visualizations` subfolder. Key visualizations include:

  * **Nested Spheres (3D Projection):** Visualizes the boundary manifolds of all levels projected into 3D space using PCA. Inner spheres represent deeper levels in the nesting hierarchy (e.g., `nested_spheres_epoch_20.png`).
  * **Training Metrics:** Plots training loss, learning rate, gradient norms, etc., over steps/epochs (e.g., `training_metrics.png`).

*(Note: Visualizations for `hierarchical_data.png` and `test_predictions.png` depend on specific example data/scripts not detailed here but were present in the older README)*.

## Hyperparameters

Configuration is managed via command-line arguments passed to `WuBuNest_Trainer.py`.

### WuBu Nesting Configuration (`wubu_config`)

*(Refer to defaults in `DEFAULT_CONFIG_WUBU` within the trainer script)*

| Parameter                   | CLI Argument                     | Default (Example) | Description                                          |
| :-------------------------- | :------------------------------- | :---------------- | :--------------------------------------------------- |
| `num_levels`                | `--num_levels`                   | 3                 | Number of nested hyperbolic levels                   |
| `hyperbolic_dims`           | `--hyperbolic_dims`              | `[128, 64, 32]`   | List of dimensions for each level                    |
| `boundary_points_per_level` | `--boundary_points_per_level`    | `[5, 4, 3]`       | List of numbers of learnable boundary points per level |
| `initial_curvatures`        | `--initial_curvatures`           | `[1.0, 0.5, 0.25]`| Initial curvature values (`c_i`) per level           |
| `initial_scales`            | `--initial_scales`               | `[1.0, 1.0, 1.0]` | Initial scale values (`s_i`) per level               |
| `initial_spread_values`     | `--initial_spread_values`        | `None`            | Initial spread values (`σ_i`) (defaults to scales)     |
| `learnable_curvature`       | `--learnable-curvature`          | `True`            | Learn curvature `c_i` (Use `--no-learnable-curvature`) |
| `learnable_scales`          | `--learnable-scales`             | `True`            | Learn scale `s_i` (Use `--no-learnable-scales`)        |
| `learnable_spread`          | `--learnable-spread`             | `True`            | Learn spread `σ_i` (Use `--no-learnable-spread`)       |
| `curvature_min_value`       | `--curvature_min_value`          | `1e-6`            | Minimum value constraint for curvature               |
| `scale_min_value`           | `--scale_min_value`              | `1e-6`            | Minimum value constraint for scale                   |
| `spread_min_value`          | `--spread_min_value`             | `1e-6`            | Minimum value constraint for spread                  |
| `use_level_descriptors`     | `--use-level-descriptors`        | `True`            | Enable Level Descriptor `ld_i` (`--no-use-...`)      |
| `level_descriptor_init_scale`| `--level_descriptor_init_scale` | `0.01`            | Init scale for `ld_i`                                |
| `use_level_spread`          | `--use-level-spread`             | `True`            | Enable Level Spread `σ_i` (`--no-use-...`)           |
| `transform_types`           | *Set internally based on defaults*| `["linear", ...]` | Mapping types (`mlp`, `linear`) for transitions        |
| `transform_hidden_dims`     | *Set internally based on defaults*| `[None, ...]`     | Hidden dims for MLP mappings (if used)               |
| `use_tangent_flow`          | `--use-tangent-flow`             | `True`            | Enable Tangent Flow `F_i` (`--no-use-...`)           |
| `tangent_flow_type`         | `--tangent_flow_type`            | `mlp`             | Type of flow map (`mlp`, `linear`, `none`)             |
| `tangent_flow_hidden_dim_ratio`| `--tangent_flow_hidden_dim_ratio`| `0.5`             | Hidden dim ratio for MLP tangent flow                |
| `tangent_flow_scale`        | `--tangent_flow_scale`           | `1.0`             | Scaling factor for tangent flow displacement         |
| `relative_vector_aggregation`| `--relative_vector_aggregation` | `mean`            | Method for relative vectors (`mean`, `sum`, `none`)    |
| `tangent_input_combination_dims`| `--tangent_input_combination_dims`| `[64]`            | Hidden dims for tangent input combiner MLP           |
| `aggregation_method`        | `--aggregation_method`           | `concat_tangent`  | Method to aggregate level tangent outputs            |
| `dropout`                   | `--dropout`                      | `0.1`             | General dropout rate                                 |

*(Note: List arguments require lengths matching `num_levels` or `num_levels - 1` as appropriate. Trainer script validates/adjusts some.)*

### Sequence Model Configuration (`sequence_config`)

| Parameter                | CLI Argument                  | Default | Description                                         |
| :----------------------- | :---------------------------- | :------ | :-------------------------------------------------- |
| `local_hidden_size`      | `--local_hidden_size`         | 256     | Hidden dim for Local Encoder/Decoder (Tangent)    |
| `decoder_memory_dim`   | `--decoder_memory_dim`      | 512     | Dimension of WuBu output / Decoder input memory (Tangent) |
| `context_window`       | `--context_window`          | 512     | Input sequence length                             |
| `n_gram_sizes`           | `--n_gram_sizes`            | `[]`    | N-gram sizes for Local Encoder features (Euclidean) |
| `n_gram_vocab_size`    | `--n_gram_vocab_size`       | 30000   | Vocab size for N-gram hashing (Euclidean)         |
| `use_hierarchical_decoder` | `--use-hierarchical-decoder`| `True`  | Use hierarchical prediction head (`--no-use-...`)    |
| `num_encoder_layers`   | `--num_encoder_layers`      | 2       | Layers in Local Encoder Transformer               |
| `num_decoder_layers`   | `--num_decoder_layers`      | 4       | Layers in Local Decoder Transformer               |
| `num_encoder_heads`    | `--num_encoder_heads`       | 8       | Heads in Local Encoder Transformer                |
| `num_decoder_heads`    | `--num_decoder_heads`       | 8       | Heads in Local Decoder Transformer                |

### Training Hyperparameters

| Parameter          | CLI Argument         | Default | Description                                |
| :----------------- | :------------------- | :------ | :----------------------------------------- |
| `learning_rate`    | `--learning_rate`    | `5e-4`  | Base learning rate for optimizer           |
| `weight_decay`     | `--weight_decay`     | `0.01`  | L2 regularization strength               |
| `grad_accum_steps` | `--grad_accum_steps` | 4       | Gradient accumulation steps                |
| `max_grad_norm`    | `--max_grad_norm`    | `1.0`   | Max gradient norm for clipping (0=disable) |
| `batch_size`       | `--batch_size`       | 32      | Global batch size across all GPUs          |
| `epochs`           | `--epochs`           | 5       | Number of training epochs                    |
| `num_workers`      | `--num_workers`      | 2       | DataLoader workers per GPU                 |
| `no_amp`           | `--no_amp`           | `False` | Disable Automatic Mixed Precision        |
| `seed`             | `--seed`             | 42      | Random seed                                |
| `detect_anomaly`   | `--detect_anomaly`   | `False` | Enable autograd anomaly detection (slow)   |

### Q-Learning Controller Hyperparameters (Optional)

| Parameter             | CLI Argument              | Default | Description                            |
| :-------------------- | :------------------------ | :------ | :------------------------------------- |
| `q_controller_enabled`| `--enable-q-controller`   | `True`  | Enable Q-controller (`--no-enable-...`) |
| `q_learning_rate`     | `--q_learning_rate`     | `0.02`  | Q-Table learning rate (alpha)          |
| `q_discount`          | `--q_discount`          | `0.95`  | Q-Learning discount factor (gamma)     |
| `q_epsilon`           | `--q_epsilon`           | `0.25`  | Initial exploration rate               |
| `q_epsilon_decay`     | `--q_epsilon_decay`     | `0.9999`| Epsilon decay rate                     |
| `q_min_epsilon`       | `--q_min_epsilon`       | `0.02`  | Minimum epsilon value                  |

## Features

  * **Byte-Level Processing**: Native handling of any UTF-8 text.
  * **Dynamic Patching**: Babylon Index focuses computation on meaningful byte segments.
  * **WuBu Nesting**: Models complex, multi-scale hierarchies with adaptive hyperbolic geometry[cite: 3, 25].
      * **Adaptive Geometry**: Learns curvature ($c\_i$) and scale ($s\_i$) per level[cite: 4].
      * **Boundary Manifolds**: Explicitly models substructures ($B\_{i,j}$) within each level's manifold[cite: 5, 26].
      * **Tangent Space Transitions**: Performs mappings ($\\tilde{T}\_i$) between levels in Euclidean tangent spaces[cite: 6, 30]. (Note: Explicit rotation $R\_i$ from paper [cite: 5, 28] may not be fully implemented in this code version).
      * **Relative Vectors**: Computes structure relative to boundaries in the target tangent space ($d\_{i+1}$)[cite: 7, 31, 71].
      * **Level Descriptors & Spread**: Captures intrinsic level characteristics ($ld\_i$) and uncertainty/density ($\\sigma\_i$)[cite: 8, 32, 33, 72, 74].
      * **Intra-Level Flow**: Models dynamics or adjustments within a level's tangent space ($F\_i$)[cite: 9, 75, 76].
  * **Hyperbolic Components**: Utilizes Poincaré Ball embeddings, GyroLinear layers (via tangent space), Riemannian Layer Normalization, and hyperbolic distance attention (experimental).
  * **Q-Learning Optimization**: Self-adapting optimizer hyperparameters (LR, Momentum) via `RiemannianEnhancedSGD` and `HAKMEMQController`.
  * **Hierarchical Decoder**: Optional two-stage byte prediction head.
  * **Gradient Monitoring**: Advanced tracking of gradient statistics (`GradientStats`).
  * **Integrated Visualizations**: Generates plots of internal WuBu structures (nested spheres) and training metrics.

## Limitations

  * **Experimental Hyperbolic Implementation**: The "fully hyperbolic" approach increases complexity and  numerical instability compared to standard Euclidean or hybrid models. Many operations are approximated via the tangent space bridge[cite: 105].
  * **Computational Intensity**: WuBu Nesting and hyperbolic operations are computationally more demanding than standard Transformers[cite: 109, 116]. Byte-level processing is inherently intensive.
  * **Training Stability**: Requires careful initialization, optimization (like `RiemannianEnhancedSGD`), and  techniques like gradient clipping and normalization[cite: 107, 108, 111].
  * **Hyperparameter Tuning**: A significant number of hyperparameters related to both the sequence model and the WuBu Nesting configuration need tuning.
  * **Memory Usage**: Byte-level processing can increase memory requirements.

## Contributing

Contributions are welcome\! Please feel free to submit a pull request or open an issue for bugs, features, or improvements.

1.  Fork the repository (`https://github.com/waefrebeorn/bytropix`)
2.  Create your feature branch (`git checkout -b feature/amazing-feature`)
3.  Commit your changes (`git commit -m 'Add some amazing feature'`)
4.  Push to the branch (`git push origin feature/amazing-feature`)
5.  Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (assuming one exists).

## Acknowledgments

  * The WuBu Nesting framework synthesizes ideas from hyperbolic geometry[cite: 13, 38], geometric deep learning, and rotation representations[cite: 20, 43].
  * Hyperbolic components inspired by research in hyperbolic neural networks[cite: 141, 130].
  * Q-learning optimization builds on reinforcement learning approaches.
  * Uses components inspired by HAKMEM concepts for entropy and patching.
  * Relies heavily on the PyTorch library and its ecosystem[cite: 137].

## Citation

```
@software{BytropixWuBuNesting,
  author = {WaefreBeorn},
  title = {Bytropix: Byte-Level Modeling with WuBu Nesting},
  year = {2024},
  url = {[https://github.com/waefrebeorn/bytropix](https://github.com/waefrebeorn/bytropix)}
}

@inproceedings{waefrebeorn2024wubunesting,
  title={WuBu Nesting: A Comprehensive Geometric Framework for Adaptive Multi-Scale Hierarchical Representation with Integrated Rotational Dynamics},
  author={WaefreBeorn, Wubu},
  booktitle={Conceptual Preprint},
  year={2024}
}
