#!/bin/bash
# Parallel resumable download of KAT-Coder-V2.5-Dev shards + configs via HF CDN
set -u
OUT=/home/wubu/models/KAT-Coder-V2.5-Dev
BASE=https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev/resolve/main
mkdir -p "$OUT"
FILES="config.json generation_config.json model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json merges.txt chat_template.jinja"
for i in $(seq -w 0 12); do FILES="$FILES model-000$i-of-00013.safetensors"; done
echo "$FILES" | tr ' ' '\n' | xargs -P 5 -I{} bash -c '
  f={}
  for try in 1 2 3 4 5; do
    curl -sSL -C - --retry 5 --retry-delay 5 -o '"$OUT"'/$f '"$BASE"'/$f && { echo "DONE $f"; exit 0; }
    echo "RETRY($try) $f"; sleep 5
  done
  echo "FAILED $f"; exit 1
'
echo "ALL XARGS JOBS FINISHED"
