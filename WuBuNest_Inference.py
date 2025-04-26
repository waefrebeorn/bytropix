# -*- coding: utf-8 -*-
"""
WuBuNest_Inference.py
Inference script for Fully Hyperbolic WuBu Nesting Model (v0.04 - Experimental Hyperbolic Core)
Compatible with checkpoints from WuBuNest_Trainer.py v0.04.
"""

import torch
import os
import argparse
import logging
from typing import List, Optional, Dict, Union, Any
import numpy as np
import time
from tqdm import tqdm
import sys
import math # Added for potential math ops if needed

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WuBuNestInference")

# Constants
EPS = 1e-7

# --- Attempt to import necessary classes from the Trainer script ---
# Add the directory containing the trainer script to the Python path if necessary
# Example: Assuming the trainer script is in the same directory
try:
    # Import necessary components defined *within* WuBuNest_Trainer.py
    from WuBuNest_Trainer import (
        WuBuNestingSequenceModel,
        ByteTokenizer,
        SamplerConfig,
        DEFAULT_CONFIG_WUBU # Import default config as a fallback if needed
    )
    logger.info("Successfully imported components from WuBuNest_Trainer.py")
except ImportError as e:
    logger.error(f"Failed to import components from WuBuNest_Trainer.py: {e}", exc_info=True)
    logger.error("Ensure WuBuNest_Trainer.py is in the same directory or accessible via PYTHONPATH.")
    sys.exit(1)
except Exception as e_import:
    logger.error(f"An unexpected error occurred during import: {e_import}", exc_info=True)
    sys.exit(1)
# --- End Import Section ---


@torch.no_grad()
def generate_with_model(model: WuBuNestingSequenceModel, # Use specific type hint
                        seed_text: str,
                        max_length: int = 100,
                        temperature: float = 0.7,
                        repetition_penalty: float = 1.1,
                        top_k: int = 0,
                        top_p: float = 0.0,
                        device: Union[str, torch.device] = "cuda") -> str:
    """Generate text using the loaded model's internal generate method."""
    model.eval()  # Set to evaluation mode

    # Convert seed text to byte tensor
    tokenizer = ByteTokenizer()
    seed_bytes = tokenizer.encode(seed_text)
    if not seed_bytes:
        logger.warning("Empty seed text provided. Using a default seed (newline).")
        seed_bytes = [10] # Default to newline character if empty

    # Prepare seed tensor for the model (needs batch dimension)
    seed_tensor = torch.tensor([seed_bytes], dtype=torch.long).to(device)

    logger.info(f"Starting generation with seed: '{seed_text[:50]}{'...' if len(seed_text)>50 else ''}'")
    logger.info(f"Generation Parameters: MaxNewBytes={max_length}, Temp={temperature:.2f}, RepPenalty={repetition_penalty:.2f}, TopK={top_k}, TopP={top_p:.2f}")

    start_time = time.time()

    # Use the model's built-in generate method
    try:
        # Create SamplerConfig instance (defaults are usually fine for inference)
        sampling_config = SamplerConfig()

        generated_tensor = model.generate(
            seed_bytes=seed_tensor,
            max_length=max_length,          # max_length here means *new* bytes to generate
            temperature=temperature,
            sampling_config=sampling_config, # Pass the config object
            repetition_penalty=repetition_penalty,
            top_k=top_k if top_k > 0 else None, # Pass None if 0
            top_p=top_p if 0.0 < top_p < 1.0 else None # Pass None if invalid
        )

        # Decode the generated sequence (tensor includes the seed)
        generated_bytes = generated_tensor[0].cpu().tolist() # Get first item from batch
        generated_text = tokenizer.decode(generated_bytes)

        end_time = time.time()
        duration = end_time - start_time
        num_new_bytes = len(generated_bytes) - len(seed_bytes)
        bytes_per_sec = num_new_bytes / duration if duration > 0 else float('inf')
        logger.info(f"Generated {num_new_bytes} new bytes in {duration:.2f} seconds ({bytes_per_sec:.2f} bytes/sec)")

        return generated_text

    except AttributeError as e:
        logger.error(f"AttributeError during generation: {e}. Does the loaded model have a 'generate' method?", exc_info=True)
        return seed_text + f" [GENERATION ATTRIBUTE ERROR: {e}]"
    except Exception as e:
        logger.error(f"Error during model.generate call: {e}", exc_info=True)
        return seed_text + f" [GENERATION RUNTIME ERROR: {e}]"

def load_checkpoint(checkpoint_path: str, device: torch.device) -> WuBuNestingSequenceModel:
    """Load the model checkpoint, reconstructing the model using saved configuration."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load checkpoint to CPU first to avoid GPU memory issues during model creation
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Checkpoint loaded. Keys: {list(checkpoint.keys())}")
    except Exception as e:
        logger.error(f"Failed to load checkpoint file: {e}", exc_info=True)
        raise

    # --- Reconstruct Configuration ---
    # Prioritize configs saved directly in the checkpoint
    wubu_config = checkpoint.get('wubu_config')
    sequence_config = checkpoint.get('sequence_config')
    args = checkpoint.get('args') # The argparse Namespace object

    if wubu_config is None or sequence_config is None:
        logger.warning("Checkpoint does not contain 'wubu_config' or 'sequence_config'. Attempting reconstruction from 'args'.")
        if args is None:
            logger.error("Checkpoint missing 'args' and direct configs. Cannot reconstruct model.")
            raise ValueError("Invalid checkpoint format - missing configuration ('args' or 'wubu_config'/'sequence_config')")

        # Reconstruct wubu_config from args (mimic logic from trainer's run())
        wubu_config = DEFAULT_CONFIG_WUBU.copy() # Start with defaults
        for key in wubu_config.keys():
            if hasattr(args, key):
                arg_val = getattr(args, key)
                # Handle boolean args specifically if necessary (argparse handles store_true/false well)
                if isinstance(wubu_config[key], bool) and isinstance(arg_val, bool):
                    wubu_config[key] = arg_val
                elif arg_val is not None: # Don't override defaults with None from args
                    wubu_config[key] = arg_val
        # Validate/correct list lengths based on num_levels
        num_levels = wubu_config['num_levels']
        num_transitions = max(0, num_levels - 1)
        for key in ['transform_types', 'transform_hidden_dims', 'rotation_types']: # rotation_types might not be used but check anyway
             if key in wubu_config and isinstance(wubu_config[key], list) and len(wubu_config[key]) != num_transitions:
                 logger.warning(f"Correcting length of loaded WuBu config '{key}'. Expected {num_transitions}. Repeating first element.")
                 first_val = wubu_config[key][0] if wubu_config[key] else DEFAULT_CONFIG_WUBU[key][0]
                 wubu_config[key] = [first_val] * num_transitions

        # Reconstruct sequence_config from args
        sequence_config = {}
        seq_keys = [ "local_hidden_size", "decoder_memory_dim", "context_window", "n_gram_sizes",
                     "n_gram_vocab_size", "num_encoder_layers", "num_decoder_layers",
                     "num_encoder_heads", "num_decoder_heads", "use_hierarchical_decoder" ]
        for key in seq_keys:
            if hasattr(args, key):
                sequence_config[key] = getattr(args, key)
            else:
                # Attempt to find a default if missing (though should be in args)
                logger.warning(f"Argument '--{key}' not found in saved args. Model reconstruction might fail or use unexpected defaults.")
                sequence_config[key] = None # Or some sensible default?

        sequence_config["vocab_size"] = 256 # Always 256 for bytes

    else:
        logger.info("Using 'wubu_config' and 'sequence_config' found directly in checkpoint.")

    # --- Create Model Instance ---
    logger.info("Creating model instance with loaded configuration...")
    try:
        model = WuBuNestingSequenceModel(wubu_config=wubu_config, sequence_config=sequence_config)
        logger.info("Model architecture instantiated.")
    except Exception as model_init_err:
        logger.error(f"Failed to instantiate WuBuNestingSequenceModel: {model_init_err}", exc_info=True)
        logger.error("Loaded WuBu Config:")
        for k, v in wubu_config.items(): logger.error(f"  {k}: {v}")
        logger.error("Loaded Sequence Config:")
        for k, v in sequence_config.items(): logger.error(f"  {k}: {v}")
        raise

    # --- Load State Dict ---
    if 'model_state_dict' not in checkpoint:
        logger.error("Checkpoint missing 'model_state_dict'.")
        raise ValueError("Invalid checkpoint - missing model weights")

    state_dict = checkpoint['model_state_dict']
    # Handle potential 'module.' prefix if saved using DDP
    if any(k.startswith('module.') for k in state_dict.keys()):
        logger.info("Removing 'module.' prefix from state_dict keys (likely saved with DDP).")
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    # Load weights, allow missing/unexpected keys for flexibility during development
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    if incompatible_keys.missing_keys:
        logger.warning(f"State dict missing keys: {incompatible_keys.missing_keys}")
    if incompatible_keys.unexpected_keys:
        logger.warning(f"State dict has unexpected keys: {incompatible_keys.unexpected_keys}")
    logger.info("Model state loaded successfully.")

    # Move model to the target device BEFORE returning
    model.to(device)
    model.eval() # Set to evaluation mode

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded to {device}. Total Params: {total_params:,}. Trainable: {trainable_params:,} (should be 0 in eval mode).")

    # Log some key config details loaded
    logger.info(f"Model Config | Levels: {wubu_config.get('num_levels', 'N/A')}, Dims: {wubu_config.get('hyperbolic_dims', 'N/A')}, CtxWin: {sequence_config.get('context_window', 'N/A')}")

    return model

def main():
    parser = argparse.ArgumentParser(description="Hyperbolic WuBu Nesting Model Text Generation (v0.04)")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file (.pt)')
    parser.add_argument('--seed_text', type=str, default="The universe is", help='Seed text to initialize generation')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum number of *new* bytes to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature (e.g., 0.7). Lower values are more deterministic.')
    parser.add_argument('--repetition_penalty', type=float, default=1.15, help='Penalty for repeating bytes (e.g., 1.1). 1.0 disables.')
    parser.add_argument('--top_k', type=int, default=40, help='Filter logits to the top K most likely bytes. 0 disables.')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling probability threshold. 0.0 disables.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Setup device
    try:
        device = torch.device(args.device)
        if args.device == 'cuda':
            if torch.cuda.is_available():
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            else:
                logger.warning("CUDA selected but not available. Falling back to CPU.")
                device = torch.device('cpu')
        else:
            logger.info("Using CPU for inference.")
    except Exception as e:
        logger.error(f"Error setting up device '{args.device}': {e}. Using CPU.")
        device = torch.device('cpu')

    try:
        # Load model from checkpoint
        model = load_checkpoint(args.checkpoint, device)

        # Generate text using the loaded model
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

        # Print the final generated text
        print("\n" + "="*60)
        print(f"Seed Text:\n{args.seed_text}")
        print("-"*60)
        print(f"Generated Text (Seed + Output):\n{generated_text}")
        print("="*60)

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
    except ValueError as e:
        logger.error(f"Configuration or Value Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the inference process: {e}", exc_info=True)

if __name__ == "__main__":
    main()


