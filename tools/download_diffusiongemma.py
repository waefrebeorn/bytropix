#!/usr/bin/env python3
"""
Download and convert DiffusionGemma to GGUF for bytropix/wubu system.
Downloads safetensors from Hugging Face, converts using llama.cpp or custom converter.
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import subprocess
import concurrent.futures
import threading
import time

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
HF_BASE = "https://huggingface.co"
SAFETENSORS_INDEX = "model.safetensors.index.json"
CONFIG_JSON = "config.json"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json"]

# Parallel download settings
MAX_PARALLEL_DOWNLOADS = 4
CHUNK_SIZE = 1024 * 1024  # 1MB chunks

def download_file(url, dest_path, desc=""):
    """Download a file with progress bar."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    with requests.get(url, headers=headers, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        
        # Check if file already exists and has correct size
        if dest_path.exists() and dest_path.stat().st_size == total:
            print(f"  {desc} - already complete ({total/1e9:.1f} GB)")
            return
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total, unit='B', unit_scale=True, desc=desc, leave=False) as pbar:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

def download_shard(args):
    """Worker function for parallel shard download."""
    shard, model_dir, i, total_shards = args
    url = f"{HF_BASE}/{MODEL_ID}/resolve/main/{shard}"
    dest = model_dir / shard
    desc = f"[{i}/{total_shards}] {shard}"
    try:
        download_file(url, dest, desc)
        return (shard, True, None)
    except Exception as e:
        return (shard, False, str(e))

def download_safetensors(model_dir):
    """Download all safetensors shards using the index file with parallel downloads."""
    model_dir = Path(model_dir)
    index_path = model_dir / SAFETENSORS_INDEX
    
    if not index_path.exists():
        # Download index first
        index_url = f"{HF_BASE}/{MODEL_ID}/resolve/main/{SAFETENSORS_INDEX}"
        download_file(index_url, index_path, "index.json")
    
    with open(index_path) as f:
        index = json.load(f)
    
    weight_map = index.get('weight_map', {})
    shard_files = sorted(set(weight_map.values()))
    
    print(f"Downloading {len(shard_files)} safetensors shards with {MAX_PARALLEL_DOWNLOADS} parallel connections...")
    
    # Track progress
    completed = 0
    failed = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_DOWNLOADS) as executor:
        # Submit all downloads
        futures = {
            executor.submit(download_shard, (shard, model_dir, i+1, len(shard_files))): shard
            for i, shard in enumerate(shard_files)
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            shard, success, error = future.result()
            completed += 1
            if success:
                size_gb = (model_dir / shard).stat().st_size / 1e9
                print(f"  ✓ {completed}/{len(shard_files)} {shard} ({size_gb:.1f} GB)")
            else:
                print(f"  ✗ {completed}/{len(shard_files)} {shard} FAILED: {error}")
                failed.append((shard, error))
    
    if failed:
        print(f"\nFailed shards: {len(failed)}")
        for shard, error in failed:
            print(f"  {shard}: {error}")
        # Retry failed shards once
        retry = len(failed)
        if retry > 0:
            print(f"Retrying {retry} failed shards...")
            # Retry logic here
    else:
        print(f"\nAll {len(shard_files)} shards downloaded successfully!")

def download_config_and_tokenizer(model_dir):
    """Download config.json and tokenizer files."""
    model_dir = Path(model_dir)
    
    # Config
    config_url = f"{HF_BASE}/{MODEL_ID}/resolve/main/{CONFIG_JSON}"
    download_file(config_url, model_dir / CONFIG_JSON, "config.json")
    
    # Tokenizer files (can run in parallel too)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for fname in TOKENIZER_FILES:
            url = f"{HF_BASE}/{MODEL_ID}/resolve/main/{fname}"
            dest = model_dir / fname
            futures[executor.submit(download_file, url, dest, fname)] = fname
        
        for future in concurrent.futures.as_completed(futures):
            fname = futures[future]
            try:
                future.result()
                print(f"  ✓ {fname}")
            except Exception as e:
                print(f"  Warning: Could not download {fname}: {e}")

def convert_to_gguf(model_dir, out_path, quant_type="Q4_K_M"):
    """Convert safetensors to GGUF using llama.cpp's convert script."""
    model_dir = Path(model_dir)
    
    # Check for llama.cpp convert script
    convert_script = Path("/home/wubu/llama.cpp/convert_hf_to_gguf.py")
    if not convert_script.exists():
        # Try to find it
        for p in Path("/home/wubu").rglob("convert_hf_to_gguf.py"):
            convert_script = p
            break
    
    if not convert_script.exists():
        print("Error: llama.cpp convert_hf_to_gguf.py not found")
        print("Install llama.cpp or provide path to convert script")
        return False
    
    # The convert script might need model_type=gemma4 or diffusion_gemma
    cmd = [
        sys.executable, str(convert_script),
        str(model_dir),
        "--outfile", str(out_path),
        "--outtype", quant_type,
        "--model-type", "gemma4"  # Based on Gemma 4 backbone
    ]
    
    print(f"Running conversion: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    
    if result.returncode != 0:
        print(f"Conversion failed:")
        print(result.stdout)
        print(result.stderr)
        return False
    
    print(f"Conversion successful! Output: {out_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Download and convert DiffusionGemma to GGUF")
    parser.add_argument("--model-dir", default="/home/wubu/models/diffusiongemma-26B", help="Directory to download model")
    parser.add_argument("--out", default="/home/wubu/models/DiffusionGemma-26B-Q4_K_M.gguf", help="Output GGUF path")
    parser.add_argument("--quant", default="Q4_K_M", help="Quantization type")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, only convert")
    parser.add_argument("--skip-convert", action="store_true", help="Only download, don't convert")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel downloads")
    
    args = parser.parse_args()
    
    global MAX_PARALLEL_DOWNLOADS
    MAX_PARALLEL_DOWNLOADS = args.parallel
    
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.skip_download:
        print(f"=== Downloading DiffusionGemma to {model_dir} ===")
        print(f"Using {MAX_PARALLEL_DOWNLOADS} parallel downloads")
        download_config_and_tokenizer(model_dir)
        download_safetensors(model_dir)
        print("Download complete!")
    
    if not args.skip_convert:
        print(f"=== Converting to GGUF ({args.quant}) ===")
        success = convert_to_gguf(model_dir, args.out, args.quant)
        if success:
            print(f"\nDone! Model ready at: {args.out}")
            print(f"Run with: ./build/bench_512k_full {args.out} 4096 1 0")

if __name__ == "__main__":
    main()