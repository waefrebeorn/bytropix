# Bytropix BSFINModel - Semantic Field Interference Network

A quantum-inspired byte-level language model with adaptive patching, reinforcement learning optimization, and hash-indexed continual learning capabilities.

## Overview

BSFINModel (Babylon Index Semantic Field Interference Network) is an experimental language model that operates directly on byte sequences rather than tokens. This approach offers several potential advantages:

-   **Tokenizer-Free Architecture**: Works directly with raw UTF-8 bytes, eliminating the need for a predefined vocabulary and handling any language, code, or data format naturally.
-   **Adaptive Patching (Babylon Index)**: Dynamically identifies meaningful segments (patches) within the byte stream based on entropy analysis, allowing the model to focus on information-dense regions.
-   **Semantic Field Interference (SFIN)**: Utilizes quantum-inspired complex-valued representations (real and imaginary parts) and interference-based attention mechanisms to potentially capture richer semantic relationships.
-   **Reinforcement Learning Optimization (EnhancedSGD)**: Employs a Q-Learning controller integrated into the SGD optimizer to dynamically adapt hyperparameters like learning rate and momentum during training based on observed performance.
-   **Continual Learning Framework (LIVEBSFIN)**: Includes an optional framework for online adaptation using hash-indexed memory to modulate learning based on data importance and recency.

This implementation features a hybrid architecture making it potentially well-suited for multilingual text, source code, and specialized byte-level data where traditional tokenizers might struggle.

## Architecture

The BSFIN model integrates several distinct components:

```mermaid
graph LR
    A[Input Bytes] --> B(Babylon Index Patching);
    B -- Variable Patches --> C(Local Encoder);
    C -- Real Patch Embeddings --> D(Real-to-Complex Projection);
    D -- Complex Patch Embeddings --> E(Complex Positional Encoding);
    E --> F(Complex Layer Norm In);
    F --> G[SFIN Stack\nComplex Interference Layers\n+ Complex Norms];
    G -- Processed Complex Repr --> H(Complex-to-Real Projection);
    H -- Real Memory Representation --> I(Local Decoder);
    J[Target Bytes\n(Training/Generation)] --> I;
    I --> K[Output Logits (Bytes 0-255)];

    style G fill:#f9d,stroke:#333,stroke-width:2px
    style B fill:#ccf,stroke:#333,stroke-width:2px
    style I fill:#cfc,stroke:#333,stroke-width:2px
```

### 1. Babylon Index Patching

This module preprocesses the input byte sequence before encoding.

```mermaid
graph LR
    subgraph Babylon Index
        direction LR
        A[Byte Sequence] --> B{Entropy Analysis (Sliding Window)};
        B --> C(Identify High-Entropy Boundaries);
        C -- Split at Boundaries --> D[Patch 1];
        C -- Split at Boundaries --> E[Patch 2];
        C -- Split at Boundaries --> F[...];
        C -- Split at Boundaries --> G[Patch N];
    end
```

- Analyzes byte sequences using sliding windows to compute local Shannon entropy.
- Identifies potential patch boundaries at points of significant entropy change.
- Filters and merges boundaries to ensure minimum patch sizes and valid UTF-8 sequences.
- Outputs a list of variable-length byte tensors (patches).

### 2. Local Encoder

- Encodes each byte patch into a fixed-size real-valued vector representation.
- Uses a standard Transformer Encoder architecture operating on byte embeddings.
- Optionally incorporates N-gram features alongside single-byte embeddings.
- Pools the information from each patch sequence (e.g., using cross-attention with a learnable query) into a single vector per patch.

### 3. Real-to-Complex Projection

- Linearly projects the real-valued patch representations into the complex domain, creating separate real and imaginary components.

### 4. SFIN Stack (Complex Processing)

- Complex Positional Encoding: Adds positional information suitable for complex representations, potentially with learnable frequency scaling.
- Complex Layer Norm: Normalizes the complex vectors.
- Entangled Interference Layers: The core of the SFIN. These layers perform attention-like operations in the complex domain.

```mermaid
graph TD
    subgraph EntangledInterferenceLayer (Simplified)
        direction TB
        A(Input: Real, Imag) --> B(Complex Projections: Q/K/V Real/Imag);
        B --> C{Apply Rotary Pos. Emb. (RoPE)};
        C --> D{Apply Entanglement Matrix (Head Mixing)};
        D --> E{Apply Learnable Phase Shifts};
        E --> F{Complex Attention Scores\n(Quantum/Classical Interference)};
        F --> G(Weighted Sum with Value (V));
        G --> H(Output Projection Real/Imag);
        H --> I(Output: Real, Imag);
    end
```

- Projects inputs to complex Q, K, V.
- Applies Rotary Positional Embeddings (RoPE) if enabled.
- Mixes information across attention heads using a learnable "entanglement" matrix if enabled.
- Applies learnable phase shifts to Q and K.
- Calculates attention scores using complex multiplication ("quantum interference") or standard dot products ("classical").
- Computes a weighted sum using the complex Value vectors.
- Projects the result back to the complex dimension.
- Includes residual connections and dropout.

### 5. Complex-to-Real Projection

- Projects the processed complex representations back into a real-valued space (potentially higher-dimensional) to serve as memory for the decoder.
- Methods include concatenating real/imag parts or using magnitude.

### 6. Local Decoder

- A standard Transformer Decoder that attends to the projected real memory representations.
- Takes the target byte sequence (shifted appropriately for autoregressive prediction) as input.
- Predicts the logits for the next byte in the sequence.

### 7. Q-Learning Enhanced SGD Optimizer

An adaptive optimizer that adjusts its own hyperparameters during training.

```mermaid
graph TD
    A[Model Training Step] --> B(Calculate Loss & Gradients);
    B --> C(Get Grad Stats / Loss Trend / Current LR/Mom);
    C --> D(Q-Controller: Determine State);
    subgraph Q-Controller Interaction
        D --> E{Choose Action\n(LR Scale, Mom Scale)};
        D --> H(Update Q-Table\n(using Reward from prev state/action));
    end
    E --> F(EnhancedSGD: Apply Action Scales to LR/Momentum);
    F --> G(EnhancedSGD: Clip Gradients & Update Model Params);
    G --> A;

    style F fill:#f9f,stroke:#333,stroke-width:1px
```

- Monitors loss trends, gradient variance, and current LR/momentum to define a state.
- Uses a Q-table to learn the best scaling factors (actions) for LR and momentum in each state.
- Computes a reward based on loss improvement and gradient stability.
- Updates the Q-table based on the reward received.
- Applies the chosen scaling factors (within bounds) to the optimizer's parameters before the update step.

### 8. Hash-Indexed Continual Learning (LIVEBSFIN)

An optional framework for adapting the model to new data streams without full retraining.

```mermaid
graph TD
    A[Input Data Stream (Context, Target)] --> B{Hash Context};
    B --> C(Hash Memory: Update/Get Metadata);
    C --> D{Calculate Importance Score\n(Based on Recency/Frequency)};
    D --> E(Calculate Gradient Scale Factor);
    A --> F[BSFIN Model Forward];
    F --> G{Calculate Loss};
    G --> H(Backward Pass - Get Gradients);
    H -- Gradients --> I{Modulate Gradients};
    E -- Scale Factor (0.1-1.0) --> I;
    I --> J(Optimizer Step);
    J --> K[Update Model];
    C --> L(Decay Importance Periodically);

    style C fill:#ffc,stroke:#333,stroke-width:1px
    style I fill:#ccf,stroke:#333,stroke-width:1px
```

- Hashes incoming data contexts.
- Stores metadata (access time, frequency, importance score) associated with each hash in a memory buffer.
- When encountering data, retrieves its importance score from memory (or assigns a default).
- Calculates a gradient_scale_factor based on importance (higher importance = closer to 1.0, lower importance = closer to 0.1).
- Multiplies the gradients by this factor before the optimizer step, effectively reducing the impact of less important or older data.
- Periodically decays the importance scores of items in memory to simulate forgetting.

## Installation

```bash
git clone https://github.com/waefrebeorn/bytropix.git
cd bytropix
# It's recommended to use a virtual environment
# python -m venv venv
# source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows
pip install torch torchvision torchaudio numpy tqdm datasets py-cpuinfo
# Optional: Install wandb for logging
# pip install wandb
```

Note: Ensure your PyTorch installation matches your CUDA version if using GPU.

## Requirements

- Python 3.8+
- PyTorch 2.0+ (CUDA recommended for performance)
- NumPy
- tqdm
- datasets (for convertdata.py)
- py-cpuinfo
- wandb (optional, for experiment tracking)

## Usage

### Data Preparation

Use the convertdata.py script to download and preprocess datasets like Anthropic HH-RLHF into the required byte-level .npy format.

```bash
python convertdata.py
# This will create wikitext_train.npy and wikitext_val.npy in ./data by default
# (Note: You might need to adjust the paths in the script)
```

### Training

Train the BSFIN model using bsfin_main.py.

```bash
python bsfin_main.py \
    --data_path ./data/wikitext_train.npy \
    --val_data_path ./data/wikitext_val.npy \
    --batch_size 16 \
    --grad_accum_steps 4 \
    --epochs 5 \
    --learning_rate 1e-4 \
    --local_hidden_size 256 \
    --complex_dim 512 \
    --num_complex_layers 6 \
    --num_complex_heads 8 \
    --decoder_memory_dim 768 \
    --n_gram_sizes 3 4 \
    --context_window 256 \
    --checkpoint_dir ./bsfin_checkpoints_v2 \
    --save_interval 1000 \
    --log_interval 50 \
    --num_workers 2
    # --wandb # Uncomment to enable WandB logging
    # --resume path/to/checkpoint.pt # Uncomment to resume training
```

(Adjust parameters like batch_size, grad_accum_steps, learning_rate, and model dimensions based on your hardware and dataset.)

### Inference (Text Generation)

Use the sfin_inference.py script for generating text.

#### Standard Mode (Single Prompt):

```bash
python sfin_inference.py standard \
    --checkpoint_path ./bsfin_checkpoints_v2/checkpoint_step_XXXX.pt \
    --input_text "The ancient spaceship drifted through the void" \
    --max_length 150 \
    --temperature 0.75 \
    --local_hidden_size 256 \
    --complex_dim 512 \
    --num_complex_layers 6 \
    --num_complex_heads 8 \
    --decoder_memory_dim 768 \
    --n_gram_sizes 3 4
    # Add --no_entanglement or --no_rope if the checkpoint was trained without them
```

#### Interactive Mode:

```bash
python sfin_inference.py interactive \
    --checkpoint_path ./bsfin_checkpoints_v2/checkpoint_step_XXXX.pt \
    --temperature 0.8 \
    --max_length 100 \
    --local_hidden_size 256 \
    --complex_dim 512 \
    --num_complex_layers 6 \
    --num_complex_heads 8 \
    --decoder_memory_dim 768 \
    --n_gram_sizes 3 4
    # Add --no_entanglement or --no_rope if the checkpoint was trained without them
```

(Important: Ensure the architecture arguments passed to the inference script match those used for training the loaded checkpoint.)

### Continual Learning (Experimental)

Use the LIVEBSFIN.py script to adapt a pre-trained model to a new data stream.

```bash
python LIVEBSFIN.py \
    --base_checkpoint ./bsfin_checkpoints_v2/checkpoint_step_XXXX.pt \
    --data_stream_file ./new_data.txt \
    --context_size 256 \
    --batch_size 8 \
    --max_steps 10000 \
    --learning_rate 1e-6 \
    --importance_factor 0.6 \
    --save_interval 500 \
    --checkpoint_dir ./bsfin_live_checkpoints_v1 \
    --local_hidden_size 256 \
    --complex_dim 512 \
    --num_complex_layers 6 \
    --num_complex_heads 8 \
    --decoder_memory_dim 768 \
    --n_gram_sizes 3 4
    # Add --no_entanglement or --no_rope if the base checkpoint was trained without them
```

## Features

- **Byte-Level Processing**: Native handling of any UTF-8 text or byte data.
- **Adaptive Patching**: Focuses computation on information-rich segments via entropy analysis.
- **Quantum-Inspired Architecture**: Leverages complex numbers and interference for potentially richer representations.
- **Self-Tuning Optimization**: EnhancedSGD adapts LR and momentum dynamically.
- **Flexible Generation**: Entropy-aware sampling strategies (greedy, top-k, full) based on output distribution uncertainty.
- **Continual Learning**: Experimental framework for online adaptation using hash-indexed memory.

## Included Scripts

- **bsfin_main.py**: Main model definition and training (Trainer) logic.
- **sfin_inference.py**: Script for text generation (standard and interactive modes).
- **convertdata.py**: Utilities for downloading and preprocessing text data into byte arrays.
- **LIVEBSFIN.py**: Implements the hash-indexed continual learning framework.
- **README.md**: This file.

(Note: EnhancedSGD optimizer is implemented within bsfin_main.py and LIVEBSFIN.py)

## Limitations

- **Experimental**: The architecture combines several novel techniques and may require careful tuning and further research for optimal performance across diverse tasks.
- **Computational Cost**: Complex-valued operations and the multi-stage architecture might be more computationally intensive than standard Transformers.
- **Sensitivity**: Complex-valued networks can sometimes be sensitive to initialization and training dynamics. The EnhancedSGD optimizer aims to mitigate this.
- **Continual Learning Stability**: The hash-indexed approach is experimental and might face challenges like catastrophic forgetting or memory capacity limitations, although importance decay helps mitigate this.

## Contributing

Contributions, issues, and feature requests are welcome! Please feel free to submit a Pull Request or open an issue on the GitHub repository.

## License

MIT License

## Acknowledgments

This project draws inspiration from various research areas:
- Byte-level language modeling (e.g., ByT5, CANINE)
- Quantum-inspired machine learning concepts (complex representations, interference)
- Reinforcement learning for hyperparameter optimization (Q-Learning)
- Continual learning techniques

## Citation

If you use this code or ideas from this project in your research, please consider citing:

```
@software{bsfin_model_2025,
  author = {WaefreBeorn},
  title = {BSFINModel: Babylon Index Semantic Field Interference Network},
  year = {2025},
  url = {https://github.com/waefrebeorn/bytropix}
}
```