# Bytropix 🚀

A highly optimized PyTorch implementation of the **Byte Latent Transformer (BLT)** with advanced training and optimization capabilities. Based on the research by Pagnoni et al. (2024), Bytropix focuses on efficient byte-level language modeling using dynamic patching, entropy-based compute allocation, and distributed training.

## Features ✨

- **Dynamic Patching**: Efficient byte-level processing with entropy-based patch boundaries.
- **Enhanced SGD Optimizer**: Q-Learning powered optimization with adaptive momentum.
- **Local-Global Architecture**: Separate processing streams for bytes and patches.
- **Mixed Precision Training**: Automatic mixed precision for faster and memory-efficient training.
- **Distributed Training**: Seamless support for single and multi-GPU setups with Distributed Data Parallel (DDP).
- **Efficient N-gram Processing**: Hash-based n-gram embeddings for enhanced context understanding.
- **Real-time Monitoring**: Integration with Weights & Biases (WandB) for comprehensive training monitoring.
- **Memory Optimizations**: Memory-mapped data loading, gradient accumulation, and manual garbage collection to reduce VRAM usage.
- **Automatic Backend Selection**: Chooses the appropriate distributed backend based on the operating system.

## Table of Contents 📚

- [Features](#features-)
- [Installation 🛠️](#installation-️)
- [Quick Start 🏃‍♂️](#quick-start-)
- [Architecture 🏗️](#architecture-️)
- [Training Guide 💡](#training-guide-)
- [Text Generation 📝](#text-generation-)
- [Configuration](#configuration)
- [License 📄](#license-)
- [Citations 📚](#citations-)
- [Contributions 🤝](#contributions-)
- [Acknowledgments 🙏](#acknowledgments-)
- [Contact 📬](#contact-)

## Installation 🛠️

```bash
git clone https://github.com/waefrebeorn/bytropix.git
cd bytropix
pip install -r requirements.txt
```

Ensure you have a compatible version of Python (3.8+) and PyTorch (2.0+) installed.

## Quick Start 🏃‍♂️

### **Training the Model**

```python
from bytropix.model import ByteLatentTransformer
from bytropix.optimizer import EnhancedSGD
from bytropix.train import train_model, prepare_dataloader
from bytropix.utils import SamplerConfig

# Initialize model
model = ByteLatentTransformer(
    hidden_size=256,
    num_layers=4,
    num_heads=8,
    dropout_rate=0.1,
    context_size=8
)

# Prepare DataLoader
dataloader = prepare_dataloader(
    batch_size=32,
    context_size=8,
    num_workers=4,
    distributed=True  # Set to False for single GPU or CPU training
)

# Initialize optimizer
optimizer = EnhancedSGD(
    model.parameters(),
    lr=1e-3,
    entropy_threshold=0.3,
    patch_size=6
)

# Training configuration
config = SamplerConfig(
    low_entropy_threshold=0.3,
    medium_entropy_threshold=1.2,
    high_entropy_threshold=2.5
)

# Train the model
train_model(
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    epochs=5,
    learning_rate=1e-3,
    device='cuda',  # or 'cpu'
    config=config,
    distributed=True  # Set to False for single GPU or CPU training
)
```

### **Generating Text**

```python
from bytropix.model import ByteLatentTransformer
from bytropix.utils import generate_text, SamplerConfig

# Load trained model
model = ByteLatentTransformer.load_from_checkpoint('checkpoint_epoch_5.pt')
model.to('cuda')

# Define sampling configuration
sampler_config = SamplerConfig(
    low_entropy_threshold=0.3,
    medium_entropy_threshold=1.2,
    high_entropy_threshold=2.5
)

# Generate text
seed_text = "The quick brown fox jumps over the lazy dog."
generated_text = generate_text(
    model=model,
    seed_text=seed_text,
    length=100,
    config=sampler_config,
    device='cuda',
    temperature=0.8
)

print(generated_text)
```

## Architecture 🏗️

The Bytropix implementation follows the **Byte Latent Transformer (BLT)** architecture with three main components:

1. **Local Encoder**: Processes raw bytes with n-gram embeddings.
2. **Global Latent Transformer**: Handles patch-level processing using Transformer blocks.
3. **Local Decoder**: Generates output bytes based on latent representations.

### **Model Components**

- **TransformerBlock**: Core building block with multi-head attention and feed-forward layers.
- **PositionalEncoding**: Adds positional information to token embeddings.
- **LocalEncoder**: Embeds byte sequences and n-gram features for local processing.
- **LocalDecoder**: Decodes latent representations back to byte sequences.
- **ByteLatentTransformer**: Integrates local and global components into a cohesive model.

## Training Guide 💡

### **Optimizations for Efficiency and Memory Usage**

- **Memory-Mapped Data Loading**: Uses `numpy.memmap` to handle large datasets without loading them entirely into RAM.
- **Gradient Accumulation**: Simulates larger batch sizes by accumulating gradients over multiple steps.
- **Mixed Precision Training**: Leverages `torch.cuda.amp` for faster computations and reduced memory footprint.
- **Distributed Training**: Utilizes `DistributedDataParallel (DDP)` for multi-GPU setups, automatically selecting the appropriate backend (`nccl` for Linux/macOS and `gloo` for Windows).
- **Manual Garbage Collection**: Periodically invokes garbage collection and clears CUDA caches to prevent memory fragmentation.
- **Efficient Data Loading**: Configures `DataLoader` with `prefetch_factor` and `pin_memory` for optimal data transfer speeds.

### **Training Tips**

- **Context Reset on Newlines**: Enables context reset on newline characters to improve training stability and coherence.
- **Patch Size**: Start with a patch size of 6 for optimal performance based on research recommendations.
- **Monitor Entropy**: Keep an eye on entropy values to dynamically adjust patch boundaries and allocate compute resources effectively.
- **Gradient Clipping**: Apply gradient clipping to prevent exploding gradients and stabilize training.

### **Distributed Training Setup**

#### **Single GPU Training**

Simply run the training script without enabling distributed mode:

```bash
python main.py --distributed False
```

#### **Multi-GPU Distributed Training**

Ensure you have multiple GPUs available and run the script with the appropriate distributed launch utility. For Unix-based systems (Linux/macOS), use:

```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py --distributed True
```

For Windows, consider using the `gloo` backend as NCCL is not supported:

```bash
python main.py --distributed True --backend gloo
```

**Note**: Adjust `nproc_per_node` based on the number of available GPUs.

## Text Generation 📝

After training, you can generate text using the trained model. The `generate_text` function utilizes entropy-based sampling to produce coherent and contextually relevant byte sequences.

```python
from bytropix.model import ByteLatentTransformer
from bytropix.utils import generate_text, SamplerConfig

# Load trained model
model = ByteLatentTransformer.load_from_checkpoint('checkpoint_epoch_5.pt')
model.to('cuda')
model.eval()

# Define sampling configuration
sampler_config = SamplerConfig(
    low_entropy_threshold=0.3,
    medium_entropy_threshold=1.2,
    high_entropy_threshold=2.5
)

# Generate text
seed_text = "The quick brown fox jumps over the lazy dog."
generated_text = generate_text(
    model=model,
    seed_text=seed_text,
    length=100,
    config=sampler_config,
    device='cuda',
    temperature=0.8
)

print(generated_text)
```

## Configuration

### **Command-Line Arguments**

- `--distributed`: Enable Distributed Data Parallel (DDP) training.
- `--rank`: Rank of the current process (used in distributed training).
- `--local_rank`: Local rank for distributed training.

### **Training Parameters**

- **Context Size**: Number of bytes used as context for predicting the next byte.
- **Batch Size**: Number of samples per batch (per GPU).
- **Epochs**: Number of training epochs.
- **Learning Rate**: Initial learning rate for the optimizer.
- **Number of Workers**: Number of subprocesses for data loading.

These parameters can be adjusted in the `config` dictionary within the `main()` function or extended to accept additional command-line arguments for flexibility.

## License 📄

MIT License

Copyright (c) 2024 Bytropix

Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the "Software"), to deal  
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:  

The above copyright notice and this permission notice shall be included in all  
copies or substantial portions of the Software.  

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.

## Citations 📚

```bibtex
@article{pagnoni2024blt,
    title={Byte Latent Transformer: Patches Scale Better Than Tokens},
    author={Pagnoni, Artidoro and Pasunuru, Ram and Rodriguez, Pedro and Nguyen, John and Muller, Benjamin and Li, Margaret and Zhou, Chunting and Yu, Lili and Weston, Jason and Zettlemoyer, Luke and Ghosh, Gargi and Lewis, Mike and Holtzman, Ari and Iyer, Srinivasan},
    journal={arXiv},
    year={2024}
}
```

## Requirements 📋

- Python 3.8+
- PyTorch 2.0+
- numpy
- tqdm
- wandb

## Contributions 🤝

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments 🙏

This implementation is based on the research paper "Byte Latent Transformer: Patches Scale Better Than Tokens" by the FAIR team at Meta and University of Washington researchers.

Special thanks to:
- The FAIR team at Meta
- The University of Washington researchers
- The open source community

## Contact 📬

For questions and feedback:
- Create an issue in the GitHub repository
- Contact the maintainers directly through GitHub

---

Made with ❤️ by the Bytropix team