import torch
import os
import argparse
import logging
from typing import List, Optional, Dict, Union
import numpy as np
import time
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
EPS = 1e-7

# Simple byte tokenizer
class ByteTokenizer:
    """Simple stateless tokenizer for converting between text and utf-8 byte sequences."""
    def encode(self, text: str) -> List[int]:
        return list(text.encode('utf-8', errors='replace'))
    
    def decode(self, byte_sequence) -> str:
        valid_bytes = []
        for b in byte_sequence:
            try:
                val = b.item() if hasattr(b, 'item') else int(b)
                if 0 <= val <= 255:
                    valid_bytes.append(val)
            except:
                continue
        return bytes(valid_bytes).decode('utf-8', errors='replace')

# Sampler configuration
class SamplerConfig:
    low_entropy_threshold: float = 0.3
    medium_entropy_threshold: float = 1.2
    high_entropy_threshold: float = 2.5
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

@torch.no_grad()
def generate_with_model(model, seed_text: str, max_length: int = 100, 
                       temperature: float = 0.7, repetition_penalty: float = 1.1, 
                       top_k: int = 0, top_p: float = 0.0, device="cuda"):
    """Generate text using the loaded model with proper seed text handling."""
    model.eval()  # Set to evaluation mode
    
    # Convert seed text to byte tensor
    tokenizer = ByteTokenizer()
    seed_bytes = tokenizer.encode(seed_text)
    if not seed_bytes:
        logger.warning("Empty seed text. Using newline character as default.")
        seed_bytes = [10]  # newline as default
    
    seed_tensor = torch.tensor([seed_bytes], dtype=torch.long).to(device)
    
    logger.info(f"Starting generation with seed text: '{seed_text}'")
    logger.info(f"Parameters: Max Length={max_length}, Temperature={temperature}, RepPenalty={repetition_penalty}")
    start_time = time.time()
    
    # Configure sampling
    sampling_config = SamplerConfig()
    
    # Call the model's generate method
    with torch.no_grad():
        try:
            generated = model.generate(
                seed_bytes=seed_tensor,
                max_length=max_length,
                temperature=temperature,
                sampling_config=sampling_config,
                repetition_penalty=repetition_penalty,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p if 0.0 < top_p < 1.0 else None
            )
            
            # Get generated text
            generated_bytes = generated[0].cpu().tolist()
            generated_text = tokenizer.decode(generated_bytes)
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Generated {len(generated_bytes) - len(seed_bytes)} new bytes in {duration:.2f} seconds")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return seed_text + " [GENERATION ERROR]"

def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load the model checkpoint with proper metadata handling."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint to CPU first to avoid GPU memory issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract args and metadata from checkpoint
    if 'args' not in checkpoint or checkpoint['args'] is None:
        logger.error("Checkpoint missing args data needed to reconstruct model")
        raise ValueError("Invalid checkpoint format - missing args")
    
    # Get configuration from saved args
    args = checkpoint['args']
    
    # Import model class from training code - assumes WuBuNestingSequenceModel is available
    from WuBuNest_TrainerV1 import WuBuNestingSequenceModel
    
    # Reconstruct configuration dicts
    wubu_config = {
        "num_levels": args.num_levels,
        "hyperbolic_dims": args.hyperbolic_dims,
        "boundary_points_per_level": args.boundary_points_per_level,
        "initial_curvatures": args.initial_curvatures,
        "initial_scales": args.initial_scales,
        "initial_spread_values": args.initial_spread_values,
        "learnable_curvature": args.learnable_curvature,
        "learnable_scales": args.learnable_scales,
        "learnable_spread": args.learnable_spread,
        "use_level_descriptors": args.use_level_descriptors,
        "use_level_spread": args.use_level_spread,
        "use_tangent_flow": args.use_tangent_flow,
        "curvature_min_value": args.curvature_min_value,
        "scale_min_value": args.scale_min_value,
        "spread_min_value": args.spread_min_value,
        "level_descriptor_init_scale": args.level_descriptor_init_scale,
        "relative_vector_aggregation": args.relative_vector_aggregation,
        "tangent_input_combination_dims": args.tangent_input_combination_dims,
        "tangent_flow_type": args.tangent_flow_type,
        "tangent_flow_hidden_dim_ratio": args.tangent_flow_hidden_dim_ratio,
        "tangent_flow_scale": args.tangent_flow_scale,
        "rotation_types": args.rotation_types,
        "transform_types": args.transform_types,
        "transform_hidden_dims": args.transform_hidden_dims,
        "aggregation_method": args.aggregation_method,
        "dropout": args.dropout
    }
    
    sequence_config = {
        "local_hidden_size": args.local_hidden_size,
        "decoder_memory_dim": args.decoder_memory_dim,
        "context_window": args.context_window,
        "n_gram_sizes": args.n_gram_sizes,
        "n_gram_vocab_size": args.n_gram_vocab_size,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "num_encoder_heads": args.num_encoder_heads,
        "num_decoder_heads": args.num_decoder_heads,
        "use_hierarchical_decoder": args.use_hierarchical_decoder,
        "vocab_size": 256
    }
    
    logger.info("Creating model with configuration from checkpoint")
    model = WuBuNestingSequenceModel(wubu_config=wubu_config, sequence_config=sequence_config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        # Handle DDP model saving prefix if present
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        logger.info("Model state loaded")
    else:
        logger.error("Checkpoint missing model state dict")
        raise ValueError("Invalid checkpoint - missing model weights")
    
    # Move model to device
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully ({sum(p.numel() for p in model.parameters()):,} parameters)")
    return model

def main():
    parser = argparse.ArgumentParser(description="WuBu Nesting Model Text Generation")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--seed_text', type=str, default="Roses are red,", help='Seed text to start generation')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum new bytes to generate')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for sampling (lower = more deterministic)')
    parser.add_argument('--repetition_penalty', type=float, default=1.1, help='Penalty for repeating tokens')
    parser.add_argument('--top_k', type=int, default=0, help='Top-K sampling (0 to disable)')
    parser.add_argument('--top_p', type=float, default=0.0, help='Top-P nucleus sampling (0 to disable)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device)
    if args.device == 'cuda' and torch.cuda.is_available():
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU for inference")
    
    try:
        # Load model from checkpoint
        model = load_checkpoint(args.checkpoint, device)
        
        # Generate text
        generated_text = generate_with_model(
            model=model,
            seed_text=args.seed_text,
            max_length=args.max_length,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device
        )
        
        # Print results
        print("\n" + "="*50)
        print(f"Seed: {args.seed_text}")
        print("-"*50)
        print(f"Generated:\n{generated_text}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error during inference process: {e}", exc_info=True)

if __name__ == "__main__":
    main()