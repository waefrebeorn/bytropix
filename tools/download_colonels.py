#!/usr/bin/env python3
"""Resumable Colonel weights downloader -> /home/wubu/models/<Name>/.

Uses huggingface_hub (Xet-aware, resume via .incomplete). Pulls ONLY the gaps:
  - KAT-Coder-V2.5-Dev: shards 00010..00012 (10/13 present) + index
  - Qwen3.6-27B:        model.safetensors.index.json (15/15 shards present)
  - Agents-A1-4B:       full checkpoint (configs + shards + index)
  - BTL-3:              LoRA adapter (adapter_model.safetensors + config)
Never deletes existing weights.
"""
import os, sys, time
from huggingface_hub import hf_hub_download, list_repo_files

MODELS = "/home/wubu/models"

# repo -> (local_subdir, list_of_relative_paths)
TARGETS = {
    "Kwaipilot/KAT-Coder-V2.5-Dev": (
        "KAT-Coder-V2.5-Dev",
        ["model-00010-of-00013.safetensors",
         "model-00011-of-00013.safetensors",
         "model-00012-of-00013.safetensors",
         "model.safetensors.index.json"],
    ),
    "Qwen/Qwen3.6-27B": (
        "Qwen3.6-27B",
        ["model.safetensors.index.json"],
    ),
    "InternScience/Agents-A1-4B": (
        "Agents-A1-4B",
        ["config.json", "generation_config.json", "tokenizer.json",
         "tokenizer_config.json", "vocab.json", "merges.txt",
         "chat_template.jinja", "model.safetensors.index.json"],
    ),
    "badtheorylabs/BTL-3": (
        "BTL-3",
        ["adapter_model.safetensors", "adapter_config.json", "README.md"],
    ),
}

def main():
    ok = 0
    failed = []
    for repo, (sub, files) in TARGETS.items():
        local = os.path.join(MODELS, sub)
        os.makedirs(local, exist_ok=True)
        print(f"\n=== {repo} -> {local} ===", flush=True)
        # For Agents-A1-4B also pull the actual shards from the index.
        want = list(files)
        if repo.endswith("Agents-A1-4B"):
            try:
                repo_files = list_repo_files(repo, repo_type="model")
                for rf in repo_files:
                    if rf.startswith("model-") and rf.endswith(".safetensors"):
                        want.append(rf)
            except Exception as e:
                print(f"  index list failed: {e}", flush=True)
        for rel in want:
            dst = os.path.join(local, rel)
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                print(f"  skip (present): {rel}", flush=True)
                ok += 1
                continue
            for attempt in range(1, 7):
                try:
                    p = hf_hub_download(repo_id=repo, filename=rel,
                                        local_dir=local, local_dir_use_symlinks=False)
                    print(f"  OK: {rel} -> {os.path.basename(p)}", flush=True)
                    ok += 1
                    break
                except Exception as e:
                    print(f"  retry {attempt} {rel}: {e}", flush=True)
                    time.sleep(3)
            else:
                failed.append((repo, rel))
    print(f"\nDONE ok={ok} failed={len(failed)}", flush=True)
    for r, f in failed:
        print(f"  FAILED {r}/{f}", flush=True)
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())
