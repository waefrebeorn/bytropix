# -*- coding: utf-8 -*-
"""
Poem_Generator_HypV04.py
Poetry Structure Generator using models trained with WuBuNest_Trainer.py (v0.04 Hyperbolic).
"""

import torch
import os
import argparse
import logging
import time
import sys
import math
import re  # Import regex for postprocessing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PoemGeneratorHypV04")

# --- Attempt to import necessary classes from the Trainer script ---
# Ensure WuBuNest_Trainer.py is in the same directory or accessible via PYTHONPATH
try:
    from WuBuNest_Trainer import (
        WuBuNestingSequenceModel,
        ByteTokenizer,
        SamplerConfig,
        DEFAULT_CONFIG_WUBU # Import default config as a fallback
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


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> WuBuNestingSequenceModel:
    """Load the WuBuNestingSequenceModel from a checkpoint, reconstructing from saved config."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load checkpoint to CPU first
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Checkpoint loaded. Keys: {list(checkpoint.keys())}")
    except Exception as e:
        logger.error(f"Failed to load checkpoint file: {e}", exc_info=True)
        raise

    # --- Reconstruct Configuration ---
    wubu_config = checkpoint.get('wubu_config')
    sequence_config = checkpoint.get('sequence_config')
    args = checkpoint.get('args') # Namespace object

    if wubu_config is None or sequence_config is None:
        logger.warning("Checkpoint missing 'wubu_config' or 'sequence_config'. Attempting reconstruction from 'args'.")
        if args is None:
            logger.error("Checkpoint missing 'args' and direct configs. Cannot reconstruct model.")
            raise ValueError("Invalid checkpoint format - missing configuration ('args' or 'wubu_config'/'sequence_config')")

        # Reconstruct wubu_config from args
        wubu_config = DEFAULT_CONFIG_WUBU.copy()
        for key in wubu_config.keys():
            if hasattr(args, key):
                arg_val = getattr(args, key)
                if arg_val is not None: wubu_config[key] = arg_val
        # Ensure transition list lengths match num_levels
        num_levels = wubu_config['num_levels']
        num_transitions = max(0, num_levels - 1)
        for key in ['transform_types', 'transform_hidden_dims', 'rotation_types']:
             if key in wubu_config and isinstance(wubu_config[key], list) and len(wubu_config[key]) != num_transitions:
                 logger.warning(f"Correcting length of loaded WuBu config '{key}'. Expected {num_transitions}. Repeating first.")
                 first_val = wubu_config[key][0] if wubu_config[key] else DEFAULT_CONFIG_WUBU.get(key, [None])[0] # Use default if needed
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
                 logger.warning(f"Argument '--{key}' not found in saved args. Using None or default.")
                 sequence_config[key] = None # Or default if available

        sequence_config["vocab_size"] = 256 # Always 256
        logger.info("Reconstructed configuration from 'args'.")
    else:
        logger.info("Using 'wubu_config' and 'sequence_config' found directly in checkpoint.")

    # --- Create Model Instance ---
    logger.info("Creating model instance with loaded configuration...")
    try:
        # Use the imported model class
        model = WuBuNestingSequenceModel(wubu_config=wubu_config, sequence_config=sequence_config)
        logger.info("Model architecture instantiated.")
    except Exception as model_init_err:
        logger.error(f"Failed to instantiate WuBuNestingSequenceModel: {model_init_err}", exc_info=True)
        logger.error("Loaded WuBu Config:"); [logger.error(f"  {k}: {v}") for k,v in wubu_config.items()]
        logger.error("Loaded Sequence Config:"); [logger.error(f"  {k}: {v}") for k,v in sequence_config.items()]
        raise

    # --- Load State Dict ---
    if 'model_state_dict' not in checkpoint:
        logger.error("Checkpoint missing 'model_state_dict'.")
        raise ValueError("Invalid checkpoint - missing model weights")

    state_dict = checkpoint['model_state_dict']
    # Handle potential 'module.' prefix from DDP
    if any(k.startswith('module.') for k in state_dict.keys()):
        logger.info("Removing 'module.' prefix from state_dict keys.")
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    if incompatible_keys.missing_keys:
        logger.warning(f"State dict missing keys: {incompatible_keys.missing_keys}")
    if incompatible_keys.unexpected_keys:
        logger.warning(f"State dict has unexpected keys: {incompatible_keys.unexpected_keys}")
    logger.info("Model state loaded successfully.")

    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded to {device}. Total Params: {total_params:,}.")
    logger.info(f"Model Config | Levels: {wubu_config.get('num_levels', 'N/A')}, Dims: {wubu_config.get('hyperbolic_dims', 'N/A')}, CtxWin: {sequence_config.get('context_window', 'N/A')}")

    return model

# SamplerConfig is now imported from WuBuNest_Trainer

@torch.no_grad()
def generate_structure(model: WuBuNestingSequenceModel, # Type hint with imported class
                       seed_text: str, max_length: int = 150, temperature: float = 0.4,
                       repetition_penalty: float = 1.2, top_k: int = 0, top_p: float = 0.0,
                       device: str = "cuda") -> str:
    """Generate a poem structure using the model's generate method."""
    model.eval()
    tokenizer = ByteTokenizer() # Use imported/defined tokenizer

    # Format line breaks correctly in the seed text
    seed_text_formatted = seed_text.replace("\\n", "\n")
    seed_bytes = tokenizer.encode(seed_text_formatted)

    # Handle empty seed
    if not seed_bytes:
        logger.warning("Empty seed text provided. Using newline character.")
        seed_bytes = [10] # Default to newline character
        seed_text_formatted = "\n" # Update the formatted seed text as well

    # Prepare seed tensor
    seed_tensor = torch.tensor([seed_bytes], dtype=torch.long).to(device)

    # Print header and seed
    print("\nGENERATING POEM STRUCTURE:\n" + "-"*30)
    print(seed_text_formatted, end="") # Print the seed text without extra newline

    logger.info(f"Starting generation with seed: '{seed_text_formatted[:50]}{'...' if len(seed_text_formatted)>50 else ''}'")
    logger.info(f"Settings: MaxNewBytes={max_length}, Temp={temperature:.2f}, RepPenalty={repetition_penalty:.2f}, TopK={top_k}, TopP={top_p:.2f}")
    start_time = time.time()

    # Create sampling config (using the imported class)
    # Use defaults or allow passing custom values if needed later
    sampling_config = SamplerConfig()

    # Generate using the model's method
    with torch.no_grad():
        try:
            generated_tensor = model.generate(
                seed_bytes=seed_tensor,
                max_length=max_length, # max_length is max *new* bytes
                temperature=temperature,
                sampling_config=sampling_config, # Pass the config object
                repetition_penalty=repetition_penalty,
                top_k=top_k if top_k > 0 else None, # Pass None if 0 or less
                top_p=top_p if 0.0 < top_p < 1.0 else None # Pass None if invalid
            )

            # Decode the full sequence (includes seed)
            generated_bytes = generated_tensor[0].cpu().tolist()
            generated_text_full = tokenizer.decode(generated_bytes)

            # Print only the *newly* generated part continuously
            # Check if the generated text starts *exactly* with the seed used for generation
            if generated_text_full.startswith(seed_text_formatted):
                 newly_generated_text = generated_text_full[len(seed_text_formatted):]
                 print(newly_generated_text, end="") # Continue printing on the same line
            else:
                 # If it doesn't start as expected (unlikely but possible), print the whole thing after a separator
                 logger.warning("Generated text did not start with the exact seed. Printing full output.")
                 print("\n[Full Output Below]\n" + generated_text_full, end="")

            print("\n" + "-"*30) # Add the closing separator line

            duration = time.time() - start_time
            num_new_bytes = len(generated_bytes) - len(seed_bytes)
            bytes_per_sec = num_new_bytes / duration if duration > 0 else float('inf')
            logger.info(f"Generated {num_new_bytes} new bytes in {duration:.2f} seconds ({bytes_per_sec:.2f} bytes/sec)")

            return generated_text_full # Return the full text (seed + generated)

        except AttributeError as e:
             logger.error(f"AttributeError during generation: {e}. Does the loaded model have a 'generate' method?", exc_info=True)
             return seed_text_formatted + f" [GENERATION ATTRIBUTE ERROR: {e}]"
        except Exception as e:
            logger.error(f"Error during model.generate call: {e}", exc_info=True)
            return seed_text_formatted + f" [GENERATION RUNTIME ERROR: {e}]"


def postprocess_output(text: str) -> str:
    """Clean up the generated text to make it more presentable."""
    # Replace excessively repeated letters (more than 3 times) with a single instance
    # Handles cases like 'aaaaaa' -> 'a', but keeps 'apple'
    text = re.sub(r'([a-zA-Z])\1{3,}', r'\1', text)

    # Replace specific sequences of repeated vowels with meaningful words (example substitutions)
    text = text.replace("eeeee", "evening")
    text = text.replace("iiiii", "silence")
    text = text.replace("ooooo", "sorrow")
    text = text.replace("aaaaa", "always")
    # Add more replacements as needed based on observed artifacts

    # Consolidate multiple newlines into a maximum of two (for stanza breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace from the entire text and from each line
    lines = [line.strip() for line in text.strip().split('\n')]
    text = '\n'.join(lines)

    return text

def main():
    parser = argparse.ArgumentParser(description="Hyperbolic WuBu Nesting Poetry Structure Generator (v0.04)")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file (.pt)')
    parser.add_argument('--seed_text', type=str, default="Poem Title: Whispers of Time\n\nIn shadows deep,", help='Seed text to initialize generation. Use \\n for newlines.')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum number of *new* bytes to generate after the seed')
    parser.add_argument('--temperature', type=float, default=0.65, help='Sampling temperature (e.g., 0.7). Lower values are more deterministic.')
    parser.add_argument('--repetition_penalty', type=float, default=1.15, help='Penalty for repeating bytes (e.g., 1.1). 1.0 disables.')
    parser.add_argument('--top_k', type=int, default=40, help='Filter logits to the top K most likely bytes. 0 disables.')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling probability threshold (e.g., 0.9). 0.0 disables.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    parser.add_argument('--cleanup', action='store_true', help='Apply post-processing cleanup to the generated text')

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
        model = load_model_from_checkpoint(args.checkpoint, device)

        # Generate poem structure using the loaded model
        generated_text_full = generate_structure(
            model=model,
            seed_text=args.seed_text,         # Pass raw seed text with \n
            max_length=args.max_length,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device
        )

        # Optional post-processing cleanup
        if args.cleanup:
            logger.info("Applying post-processing cleanup...")
            final_text_to_print = postprocess_output(generated_text_full)
            print("\nCleaned Text:\n" + "-"*30)
            print(final_text_to_print)
            print("-" * 30)
        # else: # If no cleanup, the text was already printed during generation
        #     pass

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
    except ValueError as e:
        logger.error(f"Configuration or Value Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the inference process: {e}", exc_info=True)

if __name__ == "__main__":
    main()