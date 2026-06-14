#!/usr/bin/env python3
"""
Fast, resumable download of DiffusionGemma using huggingface_hub.
Uses parallel downloads with retries and larger chunk sizes.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Set larger chunk size for faster downloads
os.environ["HF_HUB_DOWNLOAD_CHUNK_SIZE"] = "104857600"  # 100MB chunks

from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

MODEL_ID = "google/diffusiongemma-26B-A4B-it"

def download_model(model_dir, revision="main"):
    """Download model using snapshot_download which handles parallel + retries."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {MODEL_ID} to {model_dir}")
    print("Using huggingface_hub snapshot_download with 8 parallel workers...")
    
    try:
        # Download with parallel workers, resume support
        snapshot_download(
            repo_id=MODEL_ID,
            repo_type="model",
            revision=revision,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            max_workers=8,
            resume_download=True,
            allow_patterns=[
                "*.safetensors",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json", 
                "chat_template.jinja",
                "processor_config.json",
                "model.safetensors.index.json",
                "generation_config.json"
            ],
            ignore_patterns=["*.md", "*.txt", "*.pdf", "*.png", "*.jpg"]
        )
        print("Download complete!")
        return True
    except HfHubHTTPError as e:
        print(f"HF Hub error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fast download DiffusionGemma using huggingface_hub")
    parser.add_argument("--model-dir", default="/home/wubu/models/diffusiongemma-26B", help="Directory to download model")
    parser.add_argument("--revision", default="main", help="Git revision")
    
    args = parser.parse_args()
    
    success = download_model(args.model_dir, args.revision)
    if success:
        print(f"\nDone! Model ready at: {args.model_dir}")
    else:
        print("\nDownload failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()