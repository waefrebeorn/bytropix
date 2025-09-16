# =================================================================================================
#
#        MANIFOLD CONDUCTOR: DATA SYNCHRONIZATION SCRIPT (V3.0 - Trimming)
#
#   This script provides the definitive fix for data misalignment caused by batched
#   preprocessing with `drop_remainder=True`.
#
#   It loads all data files, identifies the shortest length (which comes from the
#   .npy files), and trims the longer .pkl file to match. It saves the result
#   to a new `_synced` file, ensuring perfect 1:1 correspondence for the trainer.
#
# =================================================================================================

import os
import sys
import argparse
from pathlib import Path
import pickle
import numpy as np
from rich.console import Console

def main():
    parser = argparse.ArgumentParser(
        description="Synchronize Manifold Conductor datasets by trimming longer files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data-dir', type=str, required=True, help="Directory containing the prepared data files.")
    parser.add_argument('--basename', type=str, required=True, help="Base name used during data preparation (e.g., 'my_model').")
    parser.add_argument('--depth-layers', type=int, required=True, help="Number of depth layers used (e.g., 128).")
    args = parser.parse_args()

    console = Console()
    console.print("--- 🚀 Manifold Data Synchronizer (Trimming Method) ---", style="bold yellow")

    # --- 1. Define File Paths ---
    data_dir = Path(args.data_dir)
    original_paired_path = data_dir / f"paired_data_{args.basename}.pkl"
    depth_path = data_dir / f"depth_maps_{args.basename}_{args.depth_layers}l.npy"
    alpha_path = data_dir / f"alpha_maps_{args.basename}.npy"
    synced_paired_path = data_dir / f"paired_data_synced_{args.basename}.pkl"

    # --- 2. Check for File Existence ---
    required_files = [original_paired_path, depth_path, alpha_path]
    if not all(f.exists() for f in required_files):
        console.print("[bold red]FATAL: One or more required data files not found![/bold red]")
        sys.exit(1)

    # --- 3. Load All Datasets ---
    console.print("-> Loading all datasets into memory...", style="cyan")
    with open(original_paired_path, 'rb') as f:
        paired_data = pickle.load(f)
    latents = np.asarray(paired_data['latents'])
    embeddings = np.asarray(paired_data['embeddings'])
    depth_maps = np.load(depth_path)
    alpha_maps = np.load(alpha_path)

    # --- 4. Identify Lengths and Determine Sync Target ---
    len_paired = len(latents)
    len_depth = len(depth_maps)
    len_alpha = len(alpha_maps)

    console.print("\n--- 📊 Initial Dataset Lengths ---", style="bold yellow")
    console.print(f"- Paired Data (.pkl):   {len_paired}")
    console.print(f"- Depth Maps (.npy):    {len_depth}")
    console.print(f"- Alpha Maps (.npy):    {len_alpha}")

    min_len = min(len_paired, len_depth, len_alpha)

    if len_paired == len_depth == len_alpha:
        console.print("\n[bold green]✅ Success! All datasets are already synchronized.[/bold green]")
        # It's good practice to rename the file anyway for the trainer to use a consistent name.
        console.print(f"Creating a '{synced_paired_path.name}' copy for consistency.")
        import shutil
        shutil.copy(original_paired_path, synced_paired_path)
        sys.exit(0)

    console.print(f"\n-> Mismatch detected. Trimming all datasets to the shortest length: [bold cyan]{min_len}[/bold cyan]")

    # --- 5. Trim Datasets to Target Length ---
    final_latents = latents[:min_len]
    final_embeddings = embeddings[:min_len]

    # --- 6. Save the new, synchronized .pkl file ---
    console.print("\n--- 💾 Saving new synchronized data file... ---", style="bold yellow")

    try:
        with open(synced_paired_path, 'wb') as f:
            pickle.dump({'latents': final_latents, 'embeddings': final_embeddings}, f)
        console.print(f"✅ Successfully saved new aligned data to: [bold green]{synced_paired_path}[/bold green]")
        console.print(f"   New Length: {len(final_latents)}")
        console.print("\n[bold]IMPORTANT:[/] Your trainer will now be updated to load this `_synced` file.")
    except Exception as e:
        console.print(f"[bold red]FATAL: Failed to save the new file: {e}[/bold red]")
        sys.exit(1)

    console.print("\n🎉 [bold green]Data synchronization complete![/bold green]")

if __name__ == "__main__":
    main()