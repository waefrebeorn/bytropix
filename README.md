# Bytropix

A PyTorch implementation of a Byte-level Transformer with dynamic patching and adaptive optimization.

## Overview

Bytropix is an experimental implementation of a byte-level language model inspired by recent research in tokenizer-free transformer architectures. The model directly processes raw bytes rather than tokenized inputs, offering potential benefits in robustness, multilingual processing, and handling rare or out-of-vocabulary tokens.

Key features include:

- **Byte-level Processing**: Operates directly on UTF-8 bytes instead of tokens
- **Dynamic Entropy-based Patching**: Allocates compute resources adaptively based on data complexity
- **Q-Learning Optimization**: Implements reinforcement learning for hyperparameter tuning
- **Mixed Precision Training**: Automatic mixed precision for efficient training
- **Distributed Training Support**: Compatible with single and multi-GPU setups

## Architecture

Bytropix consists of three main components:

1. **Local Encoder**: Processes raw bytes with n-gram embeddings
2. **Global Latent Transformer**: Handles patch-level processing
3. **Local Decoder**: Generates output bytes based on latent representations

The architecture draws inspiration from the Byte Latent Transformer (BLT) approach described by Pagnoni et al. (2023), which demonstrates that byte-level models with dynamic patching can match or exceed the performance of traditional tokenization-based models while improving efficiency.

### Babylon Index for Entropy-based Patching

The `BabylonIndex` component analyzes entropy in byte sequences to determine optimal patch boundaries. This allows the model to:

- Process predictable parts of text efficiently with larger patches
- Devote more computational resources to complex or unpredictable segments
- Adapt dynamically to different languages and text domains

### Q-Learning Controller for Adaptive Optimization

The `QController` and `EnhancedSGD` optimizer implement a reinforcement learning approach to dynamically adjust optimization hyperparameters during training:

- Learning rate adaptation based on loss trends
- Momentum tuning for optimal convergence
- Gradient handling with adaptive clipping

## Installation

```bash
git clone https://github.com/yourusername/bytropix.git
cd bytropix
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- numpy
- tqdm
- wandb (for experiment tracking)

## Usage

### Training

```python
from bytropix.model import BLTModel
from bytropix.optimizer import EnhancedSGD
from bytropix.train import RLHFTrainer

# Initialize model
model = BLTModel(
    local_hidden_size=256,
    global_hidden_size=1024,
    num_local_encoder_layers=1,
    num_global_layers=8,
    num_local_decoder_layers=4
)

# Initialize optimizer with Q-learning
optimizer = EnhancedSGD(
    model.parameters(),
    lr=0.003,
    momentum=0.9,
    weight_decay=0.005
)

# Initialize trainer
trainer = RLHFTrainer(
    model=model,
    optimizer=optimizer,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Train model
for epoch in range(num_epochs):
    for batch in dataloader:
        context, target = batch
        loss = trainer.train_step(context, target)
        # Log metrics, save checkpoints, etc.
```

### Text Generation

```python
from bytropix.model import BLTModel
from bytropix.utils import SamplerConfig

# Load model
model = BLTModel()
model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()

# Configure sampling
sampler_config = SamplerConfig(
    low_entropy_threshold=0.3,
    medium_entropy_threshold=1.2,
    high_entropy_threshold=2.5
)

# Generate text
seed_text = "The quick brown fox"
seed_bytes = torch.tensor([ord(c) for c in seed_text], dtype=torch.long).unsqueeze(0)
generated = model.generate(
    seed_bytes=seed_bytes,
    max_length=100,
    temperature=0.8,
    sampling_config=sampler_config
)

# Decode output
output_text = ''.join([chr(b) for b in generated[0].cpu().numpy()])
print(output_text)
```

## Research Background

This implementation is inspired by recent research on byte-level language models and dynamic patching approaches. The key concepts include:

1. **Byte-level Processing**: Operating directly on UTF-8 bytes rather than tokenized inputs, as explored in several transformer architectures (Meta AI's Byte Latent Transformer)

2. **Dynamic Patching**: Allocating compute resources adaptively based on data complexity, allowing more efficient processing of varied text

3. **Q-Learning for Hyperparameter Optimization**: Using reinforcement learning to dynamically tune model hyperparameters during training

## References

- Pagnoni, A., Pasunuru, R., Rodriguez, P., et al. (2023). "Byte Latent Transformer: Patches Scale Better Than Tokens." arXiv:2412.09871.
- Hansen, S. (2016). "Using Deep Q-Learning to Control Optimization Hyperparameters." arXiv:1602.04062.
- Qi, X., Xu, B. (2023). "Hyperparameter optimization of neural networks based on Q-learning." Signal, Image and Video Processing, 17, 1669–1676.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This is an experimental research implementation and is provided as-is without any guarantees of performance or suitability for production use cases.