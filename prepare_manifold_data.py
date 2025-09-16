# =================================================================================================
#
#        MANIFOLD CONDUCTOR: DIRECT ONNX DATA PREPARATION SCRIPT (V4.1 - FINAL)
#
#   This script is a high-performance, fully GPU-accelerated pipeline that generates
#   SOTA ground-truth depth and alpha maps. It removes the 'rembg' dependency
#   and performs salience mapping directly using onnxruntime for maximum stability.
#
# =================================================================================================

import os
import sys
import argparse
from pathlib import Path
import pickle
import importlib.util

# --- Intelligent Dependency Checking ---
def check_dependencies():
    required_packages = {
        "tensorflow": ("tensorflow-cpu", "for data loading"),
        "numpy": ("numpy", "for numerical operations"),
        "PIL": ("Pillow", "for image processing"),
        "tqdm": ("tqdm", "for progress bars"),
        "torch": ("torch torchvision", "for ML models"),
        "rich": ("rich", "for console output"),
        "transformers": ("transformers", "for the depth model"),
        "onnxruntime": ("onnxruntime-gpu", "as the engine for salience model (use 'onnxruntime' for CPU)")
    }
    missing_packages = []
    for module_name, (package_name, reason) in required_packages.items():
        spec = importlib.util.find_spec(module_name)
        if spec is None: missing_packages.append(package_name)
    if missing_packages:
        unique_packages = sorted(list(set(missing_packages)))
        install_command = f"pip install {' '.join(unique_packages)}"
        from rich.console import Console, Panel
        console = Console()
        console.print(Panel(
            f"[bold]Missing packages required:[/bold]\n\n- [cyan]{', '.join(unique_packages)}[/cyan]\n\n"
            f"Please run the following command to install them:\n\n"
            f"[bold yellow]  {install_command}  [/bold yellow]\n\n"
            f"[dim]Note: If you do not have an NVIDIA GPU, install 'onnxruntime' instead of 'onnxruntime-gpu'.[/dim]",
            title="[bold red]FATAL: Missing Dependencies[/bold red]", border_style="red"
        ))
        sys.exit(1)

check_dependencies()

# --- Environment Setup & Imports ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms.functional as F
from rich.console import Console
from transformers import DPTImageProcessor, DPTForDepthEstimation
import onnxruntime as ort

# --- Core Functions ---

def load_models(console: Console, onnx_path="u2net.onnx"):
    """Loads both the depth model (PyTorch) and salience model (ONNX) onto the GPU."""
    console.print("--- 🧠 Loading SOTA Models to GPU... ---", style="bold yellow")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"-> Using device: [bold cyan]{device}[/bold cyan]")

    console.print("-> Loading MiDaS (DPT-Large)...")
    depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
    console.print("   ✅ Depth model loaded.")
    
    console.print(f"-> Loading Salience Model (U2-Net from {onnx_path})...")
    if not Path(onnx_path).exists():
        console.print(f"[bold red]FATAL: {onnx_path} not found.[/bold red]\nPlease download it, e.g., from Hugging Face: tomjackson2023/rembg")
        sys.exit(1)
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    salience_session = ort.InferenceSession(onnx_path, providers=providers)
    console.print(f"   ✅ Salience model loaded. Using ONNX provider: [cyan]{salience_session.get_providers()}[/cyan]")
    
    return depth_model, depth_processor, salience_session, device

def create_image_dataset(data_dir: str):
    data_p = Path(data_dir)
    info_file = data_p / "dataset_info.pkl"
    if not info_file.exists(): sys.exit(f"FATAL: dataset_info.pkl not found.")
    with open(info_file, 'rb') as f: info = pickle.load(f)
    image_paths = info.get('image_paths')
    if not image_paths: sys.exit("[FATAL] 'image_paths' key not found.")
    return tf.data.Dataset.from_tensor_slices(image_paths)

def process_batch(image_paths_batch, models, device, depth_layers):
    """Processes a batch of images entirely on the GPU."""
    depth_model, depth_processor, salience_session = models
    
    images_pil = [Image.open(path.numpy().decode('utf-8')).convert("RGB").resize((512,512)) for path in image_paths_batch]
    
    # --- 1. Depth Estimation (PyTorch - Batched) ---
    with torch.no_grad():
        inputs = depth_processor(images=images_pil, return_tensors="pt").to(device)
        predicted_depth = depth_model(**inputs).predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1), size=(512, 512), mode="bicubic", align_corners=False
        ).squeeze(1)

    min_vals = torch.amin(prediction, dim=(1, 2), keepdim=True)
    max_vals = torch.amax(prediction, dim=(1, 2), keepdim=True)
    normalized_depth = (prediction - min_vals) / (max_vals - min_vals + 1e-6)
    quantized_depth = (normalized_depth * (depth_layers - 1)).cpu().numpy().astype(np.uint8)

    # --- 2. Salience (Alpha) Estimation (Direct ONNX - Iterated) ---
    alpha_maps = []
    for img in images_pil:
        # Prepare a single image for U2-Net
        img_tensor_u2net = F.to_tensor(img.resize((320, 320))).unsqueeze(0)
        img_tensor_u2net = F.normalize(img_tensor_u2net, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Run inference on the single image
        ort_inputs = {salience_session.get_inputs()[0].name: img_tensor_u2net.cpu().numpy()}
        ort_outs = salience_session.run(None, ort_inputs)
        mask_tensor = torch.from_numpy(ort_outs[0]) # Shape: (1, 1, 320, 320)
        
        # --- [THE DEFINITIVE FIX] ---
        # Use BICUBIC interpolation, which is supported for 4D tensors.
        mask_tensor = F.resize(mask_tensor, [24, 24], interpolation=F.InterpolationMode.BICUBIC, antialias=True)
        
        # Post-process the single mask
        min_mask = torch.amin(mask_tensor, dim=(1, 2, 3), keepdim=True)
        max_mask = torch.amax(mask_tensor, dim=(1, 2, 3), keepdim=True)
        norm_mask = (mask_tensor - min_mask) / (max_mask - min_mask + 1e-6)
        
        alpha_map = (norm_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
        alpha_maps.append(alpha_map[:, :, np.newaxis])

    return quantized_depth, np.stack(alpha_maps)
def main():
    parser = argparse.ArgumentParser(description="Professional Data Preparation for the Manifold Conductor.")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--basename', type=str, required=True)
    parser.add_argument('--depth-layers', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    console = Console()

    output_dir = Path(args.data_dir)
    depth_output_path = output_dir / f"depth_maps_{args.basename}_{args.depth_layers}l.npy"
    alpha_output_path = output_dir / f"alpha_maps_{args.basename}.npy"

    if depth_output_path.exists() and alpha_output_path.exists():
        console.print(f"✅ Data already exists at {output_dir}. Skipping.")
        return

    # --- [THE FIX] --- Correct unpacking
    depth_model, depth_processor, salience_session, device = load_models(console)
    models_tuple = (depth_model, depth_processor, salience_session)

    console.print(f"--- 📂 Loading image paths from {args.data_dir}... ---", style="bold yellow")
    image_ds = create_image_dataset(args.data_dir)
    num_images = len(list(image_ds))
    console.print(f"Found {num_images} images to process.")
    
    batched_ds = image_ds.batch(args.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    all_depth_maps, all_alpha_maps = [], []

    console.print(f"--- 🚀 Starting batch processing (Batch Size: {args.batch_size})... ---", style="bold yellow")
    
    for image_paths_batch in tqdm(batched_ds, total=num_images // args.batch_size, desc="Processing Batches"):
        # --- [THE FIX] --- Pass the correct tuple of models
        depth_batch, alpha_batch = process_batch(image_paths_batch, models_tuple, device, args.depth_layers)
        all_depth_maps.append(depth_batch)
        all_alpha_maps.append(alpha_batch)
        
    console.print("--- 💾 Concatenating and saving results... ---", style="bold yellow")
    
    final_depth_maps = np.concatenate(all_depth_maps, axis=0)
    final_alpha_maps = np.concatenate(all_alpha_maps, axis=0)
    
    console.print(f"Resizing depth maps to 24x24...")
    resized_depth_maps = np.array([
        np.array(Image.fromarray(dm).resize((24, 24), Image.Resampling.NEAREST)) 
        for dm in tqdm(final_depth_maps, desc="Resizing depth")
    ])

    np.save(depth_output_path, resized_depth_maps)
    console.print(f"✅ Ground-truth depth maps saved to: [green]{depth_output_path}[/green]")
    console.print(f"   Shape: {resized_depth_maps.shape}, Dtype: {resized_depth_maps.dtype}")
    
    np.save(alpha_output_path, final_alpha_maps)
    console.print(f"✅ Ground-truth alpha maps saved to: [green]{alpha_output_path}[/green]")
    console.print(f"   Shape: {final_alpha_maps.shape}, Dtype: {final_alpha_maps.dtype}")
    
    console.print("\n🎉 [bold green]Professional data preparation complete![/bold green]")

if __name__ == "__main__":
    main()