#!/usr/bin/env python3
"""
Simple, reliable sequential download with retries for DiffusionGemma.
Uses single connection per file but with resume support and retries.
"""

import os
import sys
import json
import requests
import time
from pathlib import Path
from tqdm import tqdm

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
HF_BASE = "https://huggingface.co"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def download_file(url, dest_path, max_retries=3):
    """Download a file with resume support and retries."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            # Check existing file size
            existing_size = dest_path.stat().st_size if dest_path.exists() else 0
            
            # Get total file size
            r = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
            if r.status_code not in (200, 302):
                return False, f"HEAD failed: {r.status_code}"
            total_size = int(r.headers.get('Content-Length', 0))
            
            if existing_size >= total_size:
                return True, "Already complete"
            
            # Resume download
            range_header = f"bytes={existing_size}-"
            dl_headers = headers.copy()
            dl_headers['Range'] = range_header
            
            mode = 'ab' if existing_size > 0 else 'wb'
            
            with requests.get(url, headers=dl_headers, stream=True, timeout=60) as r:
                if r.status_code not in (200, 206):
                    return False, f"GET failed: {r.status_code}"
                
                with open(dest_path, mode) as f:
                    with tqdm(
                        total=total_size, 
                        initial=existing_size,
                        unit='B', 
                        unit_scale=True, 
                        desc=dest_path.name[:25],
                        leave=False
                    ) as pbar:
                        for chunk in r.iter_content(chunk_size=1024*1024):  # 1MB chunks
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            
            # Verify
            final_size = dest_path.stat().st_size
            if final_size == total_size:
                return True, "Complete"
            else:
                return False, f"Size mismatch: {final_size}/{total_size}"
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt+1}/{max_retries} after error: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return False, str(e)
    
    return False, "Max retries exceeded"

def main():
    model_dir = Path("/home/wubu/models/diffusiongemma-26B")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load index
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        idx_url = f"{HF_BASE}/{MODEL_ID}/resolve/main/model.safetensors.index.json"
        r = requests.get(idx_url, headers=headers)
        index_path.write_bytes(r.content)
    
    with open(index_path) as f:
        index = json.load(f)
    
    shard_files = sorted(set(index['weight_map'].values()))
    print(f"Found {len(shard_files)} shards")
    
    for i, shard in enumerate(shard_files):
        url = f"{HF_BASE}/{MODEL_ID}/resolve/main/{shard}"
        dest = model_dir / shard
        
        print(f"\n[{i+1}/{len(shard_files)}] {shard}")
        success, msg = download_file(url, dest)
        if success:
            print(f"  ✓ {msg}")
        else:
            print(f"  ✗ FAILED: {msg}")
            # Continue to next file anyway

if __name__ == "__main__":
    main()