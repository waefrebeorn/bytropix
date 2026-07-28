#!/usr/bin/env python3
"""Resumable download of KAT-Coder-V2.5-Dev with xet disabled."""
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_download, HfFileSystem

REPO = "Kwaipilot/KAT-Coder-V2.5-Dev"
OUT = "/home/wubu/models/KAT-Coder-V2.5-Dev"
EXTRA = ["tokenizer.json", "tokenizer_config.json", "config.json",
         "model.safetensors.index.json", "generation_config.json"]

def main():
    os.makedirs(OUT, exist_ok=True)
    fs = HfFileSystem()
    files = [f.split("/")[-1] for f in fs.ls(REPO, detail=False)]
    shards = sorted(f for f in files if f.startswith("model-") and f.endswith(".safetensors"))
    targets = shards + [e for e in EXTRA if e in files]
    print(f"{REPO}: {len(shards)} shards, targets={len(targets)}", flush=True)
    for fn in targets:
        p = os.path.join(OUT, fn)
        have = os.path.getsize(p) if os.path.exists(p) else 0
        print(f"  {fn}: have {have/1e9:.2f} GB -> fetching", flush=True)
        hf_hub_download(repo_id=REPO, filename=fn, local_dir=OUT)
        print(f"  {fn}: done {os.path.getsize(p)/1e9:.2f} GB", flush=True)
    print("KAT DOWNLOAD COMPLETE", flush=True)

if __name__ == "__main__":
    main()
