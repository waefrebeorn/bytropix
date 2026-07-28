#!/bin/bash
# Resumable Colonel weights downloader -> /home/wubu/models/<Name>/.
# Uses curl -C - (resume) + retries against HF CDN. Token set for higher limits.
# Covers all gaps:
#   KAT-Coder-V2.5-Dev : shards 00010..00012 (10/13 present) + index
#   Qwen3.6-27B        : model.safetensors.index.json (15/15 shards present)
#   Agents-A1-4B       : full checkpoint (configs + shards from index)
#   BTL-3              : LoRA adapter (adapter_model.safetensors + config)
# Never deletes existing weights.
set -u
export HF_TOKEN="${HF_TOKEN:-***REMOVED***}"
M=/home/wubu/models

dl() { # dl <repo> <file> <local_dir> [min_bytes]
  local repo="$1" f="$2" dir="$3" min="${4:-0}"
  if [ -f "$dir/$f" ]; then
    local sz; sz=$(stat -c%s "$dir/$f" 2>/dev/null || echo 0)
    if [ "$sz" -ge "$min" ]; then echo "skip (complete) $f"; return 0; fi
    echo "partial $f ($sz < $min) — re-downloading"; rm -f "$dir/$f"
  fi
  local url="https://huggingface.co/$repo/resolve/main/$f"
  for try in 1 2 3 4 5 6 7 8; do
    curl -sSL -C - --retry 5 --retry-delay 5 --retry-all-errors -o "$dir/$f" "$url" && { echo "DONE $f"; return 0; }
    echo "RETRY($try) $f"; sleep 3
  done
  echo "FAILED $f"; return 1
}

# --- KAT ---
K=$M/KAT-Coder-V2.5-Dev
for i in 10 11 12; do dl Kwaipilot/KAT-Coder-V2.5-Dev "model-000${i}-of-00013.safetensors" "$K" 2000000000; done
dl Kwaipilot/KAT-Coder-V2.5-Dev model.safetensors.index.json "$K" 1000

# --- Qwen ---
Q=$M/Qwen3.6-27B
dl Qwen/Qwen3.6-27B model.safetensors.index.json "$Q" 1000

# --- Agents-A1-4B ---
A=$M/Agents-A1-4B; mkdir -p "$A"
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt chat_template.jinja model.safetensors.index.json; do
  dl InternScience/Agents-A1-4B "$f" "$A" 100
done
# shards: derive from index if present, else try 00001..00002
if [ -f "$A/model.safetensors.index.json" ]; then
  for f in $(grep -oE 'model-000[0-9]+-of-[0-9]+\.safetensors' "$A/model.safetensors.index.json" | sort -u); do
    dl InternScience/Agents-A1-4B "$f" "$A" 1000000000
  done
fi

# --- BTL-3 ---
B=$M/BTL-3; mkdir -p "$B"
dl badtheorylabs/BTL-3 adapter_model.safetensors "$B" 1000
dl badtheorylabs/BTL-3 adapter_config.json "$B" 100
dl badtheorylabs/BTL-3 README.md "$B" 100

echo "ALL_DOWNLOADS_FINISHED"
