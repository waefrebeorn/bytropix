#!/usr/bin/env python3
"""Persistent, resumable download of the four Colonel models + tokenizers.

Targets (all on the persistent /dev/sdd ext4, NOT /tmp this time):
  /home/wubu/models/Qwen3.6-27B/           (15 shards, base for BTL-3)
  /home/wubu/models/Agents-A1-4B/          (2 shards)
  /home/wubu/models/KAT-Coder-V2.5-Dev/    (13 shards)
  /home/wubu/models/BTL-3/                 (LoRA adapter + adapter_config.json)

hf_hub_download resumes via .incomplete + verifies, so re-running is safe.
"""
import os
from huggingface_hub import hf_hub_download, HfFileSystem

REPOS = {
    "Qwen3.6-27B":         ("Qwen/Qwen3.6-27B",          "/home/wubu/models/Qwen3.6-27B"),
    "Agents-A1-4B":        ("InternScience/Agents-A1-4B", "/home/wubu/models/Agents-A1-4B"),
    "KAT-Coder-V2.5-Dev":  ("Kwaipilot/KAT-Coder-V2.5-Dev","/home/wubu/models/KAT-Coder-V2.5-Dev"),
    "BTL-3":               ("badtheorylabs/BTL-3",        "/home/wubu/models/BTL-3"),
}

EXTRA = ["tokenizer.json", "tokenizer_config.json", "config.json", "adapter_config.json"]

def list_shards(repo):
    fs = HfFileSystem()
    files = [f.split("/")[-1] for f in fs.ls(repo, detail=False)]
    shards = sorted([f for f in files if f.startswith("model-") and f.endswith(".safetensors")])
    extras = [f for f in files if f in EXTRA]
    return shards, extras

def main():
    for name, (repo, out) in REPOS.items():
        os.makedirs(out, exist_ok=True)
        shards, extras = list_shards(repo)
        print(f"\n===== {name}  ({repo}) -> {out} =====", flush=True)
        print(f"  shards={len(shards)} extras={extras}", flush=True)
        targets = list(shards) + [e for e in extras if e not in shards]
        for fn in targets:
            p = os.path.join(out, fn)
            have = os.path.getsize(p) if os.path.exists(p) else 0
            print(f"  {fn}: have {have/1e9:.2f} GB -> fetching", flush=True)
            hf_hub_download(repo_id=repo, filename=fn, local_dir=out,
                            local_dir_use_symlinks=False)
            print(f"  {fn}: done {os.path.getsize(p)/1e9:.2f} GB", flush=True)
    print("\nALL COLONEL MODELS DOWNLOADED", flush=True)

if __name__ == "__main__":
    main()
