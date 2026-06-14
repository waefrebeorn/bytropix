#!/usr/bin/env python3
"""
High-speed parallel download for DiffusionGemma safetensors.
Uses multiple connections per file with byte-range requests.
"""

import os
import sys
import json
import argparse
import requests
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import threading
import time

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
HF_BASE = "https://huggingface.co"

# Settings
MAX_PARALLEL_FILES = 8          # Parallel files
CONNECTIONS_PER_FILE = 4        # Connections per file
CHUNK_SIZE = 1024 * 1024        # 1MB chunks
REQUEST_TIMEOUT = 30

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def get_file_info(url):
    """Get file size and check if range requests supported."""
    try:
        r = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        if r.status_code in (200, 302, 206):
            size = int(r.headers.get('Content-Length', 0))
            accept_ranges = r.headers.get('Accept-Ranges', '') == 'bytes'
            return size, accept_ranges
    except:
        pass
    return 0, False

def download_chunk(url, dest_path, start, end, progress_bar, lock):
    """Download a single byte range."""
    range_header = f"bytes={start}-{end}"
    chunk_headers = headers.copy()
    chunk_headers['Range'] = range_header
    
    try:
        with requests.get(url, headers=chunk_headers, stream=True, timeout=REQUEST_TIMEOUT) as r:
            if r.status_code != 206:
                return False, f"HTTP {r.status_code}"
            
            # Write to correct position in file
            with open(dest_path, 'r+b') as f:
                f.seek(start)
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        with lock:
                            progress_bar.update(len(chunk))
        return True, None
    except Exception as e:
        return False, str(e)

def download_file_parallel(url, dest_path, file_size, progress_lock, file_pbar):
    """Download a single file using multiple connections."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create empty file of correct size
    with open(dest_path, 'wb') as f:
        f.truncate(file_size)
    
    # Calculate chunk ranges
    num_chunks = CONNECTIONS_PER_FILE
    chunk_size = file_size // num_chunks
    ranges = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = file_size - 1 if i == num_chunks - 1 else (i + 1) * chunk_size - 1
        ranges.append((start, end))
    
    # Download chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
        futures = {
            executor.submit(download_chunk, url, dest_path, s, e, file_pbar, progress_lock): i
            for i, (s, e) in enumerate(ranges)
        }
        
        for future in concurrent.futures.as_completed(futures):
            success, error = future.result()
            if not success:
                return False, error
    
    return True, None

def main():
    parser = argparse.ArgumentParser(description="Parallel download for DiffusionGemma")
    parser.add_argument("--model-dir", default="/home/wubu/models/diffusiongemma-26B")
    parser.add_argument("--parallel-files", type=int, default=4)
    parser.add_argument("--connections-per-file", type=int, default=4)
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load index
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        idx_url = f"{HF_BASE}/{MODEL_ID}/resolve/main/model.safetensors.index.json"
        print("Downloading index...")
        r = requests.get(idx_url, headers=headers)
        index_path.write_bytes(r.content)
    
    with open(index_path) as f:
        index = json.load(f)
    
    shard_files = sorted(set(index['weight_map'].values()))
    print(f"Found {len(shard_files)} shards")
    
    # Get file sizes
    file_sizes = {}
    for shard in shard_files:
        url = f"{HF_BASE}/{MODEL_ID}/resolve/main/{shard}"
        size, ranges = get_file_info(url)
        file_sizes[shard] = size
        if size > 0:
            print(f"  {shard}: {size/1e9:.1f} GB, range: {ranges}")
        else:
            print(f"  {shard}: UNKNOWN SIZE")
    
    # Filter existing complete files
    to_download = []
    for shard in shard_files:
        dest = model_dir / shard
        expected = file_sizes.get(shard, 0)
        if dest.exists() and dest.stat().st_size == expected:
            print(f"  ✓ {shard} already complete ({expected/1e9:.1f} GB)")
        else:
            to_download.append(shard)
    
    if not to_download:
        print("All files already downloaded!")
        return
    
    print(f"\nDownloading {len(to_download)} shards with {args.parallel_files} parallel files × {args.connections_per_file} connections...")
    
    # Global progress tracking
    total_bytes = sum(file_sizes[s] for s in to_download if s in file_sizes and file_sizes[s] > 0)
    completed_bytes = 0
    completed_lock = threading.Lock()
    
    with tqdm(total=total_bytes, unit='B', unit_scale=True, desc="Overall") as overall_pbar:
        progress_lock = threading.Lock()
        
        def update_overall(n):
            nonlocal completed_bytes
            with completed_lock:
                completed_bytes += n
                overall_pbar.update(n)
        
        # Download files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_files) as executor:
            futures = {}
            for shard in to_download:
                if shard not in file_sizes or file_sizes[shard] == 0:
                    print(f"Skipping {shard}: unknown size")
                    continue
                
                url = f"{HF_BASE}/{MODEL_ID}/resolve/main/{shard}"
                dest = model_dir / shard
                size = file_sizes[shard]
                
                # Create file pbar
                file_pbar = tqdm(total=size, unit='B', unit_scale=True, 
                                desc=shard[:20], leave=False, position=1)
                
                def make_download(s, u, d, sz, fp):
                    def download():
                        nonlocal completed_bytes
                        # Custom progress callback
                        class ChunkTracker:
                            def __init__(self, pbar, overall_cb):
                                self.pbar = pbar
                                self.overall_cb = overall_cb
                                self.lock = threading.Lock()
                            def update(self, n):
                                with self.lock:
                                    self.pbar.update(n)
                                    self.overall_cb(n)
                        
                        tracker = ChunkTracker(file_pbar, update_overall)
                        return download_file_parallel(u, d, sz, tracker.lock, tracker)
                    
                    return download
                
                futures[executor.submit(make_download(shard, url, dest, size, file_pbar))] = (shard, file_pbar)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                shard, file_pbar = futures[future]
                success, error = future.result()
                file_pbar.close()
                if success:
                    print(f"\n  ✓ {shard} complete")
                else:
                    print(f"\n  ✗ {shard} FAILED: {error}")

if __name__ == "__main__":
    main()