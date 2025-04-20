# Bytropix BSFINModel - Semantic Field Interference Network

A quantum-inspired byte-level language model with adaptive patching and reinforcement learning optimization.

## Overview

BSFINModel (Babylon Index Semantic Field Interference Network) is an experimental language model that operates directly on byte sequences rather than tokens, offering several advantages:

- **Tokenizer-Free Architecture**: Works with raw UTF-8 bytes instead of predefined vocabulary tokens
- **Semantic Field Interference**: Uses quantum-inspired complex representations for better semantic understanding
- **Adaptive Patching**: Automatically identifies important boundaries in text using entropy analysis
- **Reinforcement Learning Optimization**: Self-tunes hyperparameters during training

This implementation features a hybrid architecture combining byte-level processing with quantum-inspired representations, making it especially well-suited for multilingual text, code, and specialized content where traditional tokenizers might struggle.

## Architecture

The model consists of several innovative components:

### Babylon Index

An entropy-based patching mechanism that:
- Analyzes byte sequences to find meaningful boundaries
- Creates variable-sized patches based on information density
- Ensures proper UTF-8 character boundaries are respected

### Quantum-Inspired Interference

- Uses complex-valued representations (real and imaginary components)
- Implements quantum interference patterns through specialized attention
- Supports entangled multi-head attention with phase shifts
- Includes rotary positional embeddings for better sequence understanding

### Q-Learning Optimization

- Dynamically adjusts learning rates and momentum parameters
- Monitors gradient statistics for stable training
- Adapts optimization strategy based on loss landscape
- Provides insights into training dynamics

## Installation

```bash
git clone https://github.com/yourusername/bsfin.git
cd bsfin
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- tqdm
- wandb (optional, for tracking experiments)

## Usage

### Text Generation

```python
import torch
from bsfin_main import BSFINModel, SamplerConfig, ByteTokenizer

# Initialize tokenizer
tokenizer = ByteTokenizer()

# Initialize model (or load checkpoint)
model = BSFINModel(
    local_hidden_size=256,
    complex_dim=512,
    num_complex_layers=6,
    num_complex_heads=8,
    decoder_memory_dim=768
)
model.load_state_dict(torch.load("path/to/checkpoint.pt")["model_state_dict"])
model.eval()

# Prepare input text
input_text = "Once upon a time"
input_bytes = torch.tensor([tokenizer.encode(input_text)], dtype=torch.long)

# Generate continuation
sampling_config = SamplerConfig(
    low_entropy_threshold=0.3,
    medium_entropy_threshold=1.2,
    high_entropy_threshold=2.5
)

generated = model.generate(
    seed_bytes=input_bytes,
    max_length=100,
    temperature=0.8,
    sampling_config=sampling_config
)

# Decode and print
output_text = tokenizer.decode(generated[0].tolist())
print(output_text)
```

### Training

```python
from bsfin_main import BSFINModel, ByteIterableDataset, EnhancedSGD
from torch.utils.data import DataLoader

# Initialize model
model = BSFINModel(
    local_hidden_size=256,
    complex_dim=512,
    num_complex_layers=6,
    num_complex_heads=8
)

# Prepare data
train_dataset = ByteIterableDataset("training_data.npy", context_size=256)
train_loader = DataLoader(train_dataset, batch_size=16, num_workers=4)

# Initialize optimizer with Q-learning
optimizer = EnhancedSGD(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,
    q_learning_config={
        "learning_rate": 0.02,
        "discount": 0.97,
        "epsilon": 0.15
    }
)

# Training loop
for epoch in range(10):
    for batch in train_loader:
        context, target = batch
        logits = model(byte_seq=context, target_byte_seq=context)
        loss = model.compute_loss(logits, context)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    # Save checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, f"checkpoint_epoch_{epoch}.pt")
```

## Interactive Inference

For interactive text generation, use the provided inference script:

```bash
python sfin_inference.py interactive \
    --checkpoint_path path/to/checkpoint.pt \
    --temperature 0.8 \
    --max_length 150
```

## Features

- **Byte-Level Processing**: Works with any text in any language or code without vocabulary limitations
- **Adaptive Complexity**: Uses entropy-based patching to focus compute on complex regions
- **Quantum-Inspired Architecture**: Uses complex-valued interference for richer representations
- **Self-Tuning Hyperparameters**: EnhancedSGD optimizer dynamically adjusts learning rates
- **Flexible Generation**: Supports different sampling strategies based on text entropy

## Included Scripts

- `bsfin_main.py`: Main model implementation and training logic
- `sfin_inference.py`: Script for text generation and interactive inference
- `convertdata.py`: Data preprocessing utilities
- `EnhancedSGD.py`: Implementation of the Q-learning optimizer
- `LIVEBSFIN.py`: Continual learning framework for online adaptation

## Limitations

- May require more computational resources than standard transformer models
- Experimental architecture that might need tuning for specific applications
- Complex-valued operations can be sensitive to initialization and training dynamics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Acknowledgments

This implementation draws inspiration from research in quantum computing, byte-level language models, and reinforcement learning for hyperparameter optimization.

- The ByteLatentTransformer approach (Meta AI)
- Quantum-inspired machine learning techniques
- Q-Learning for hyperparameter optimization

## Citation

If you use this code for research, please cite:

```
@software{bytropix,
  author = {[WaefreBeorn]},
  title = {BSFINModel: Babylon Index Semantic Field Interference Network},
  year = {2025},
  url = {https://github.com/waefrebeorn/bytropix}
}
```