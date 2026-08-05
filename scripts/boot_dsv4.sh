#!/bin/bash
# Boot the merged DeepSeek-V4-Flash-ConfigI GGUF and probe 4096-token coherence.
# Usage: bash scripts/boot_dsv4.sh
set -u
EXE="C:/Users/eman5/wubuwizard/gen_text_win.exe"
MODEL="D:/models/DeepSeek-V4-Flash-ConfigI/DeepSeek-V4-Flash-0731-ConfigI-merged.gguf"
LOG="C:/Users/eman5/wubuwizard/dsv4_boot.log"
PROMPT="You are WuBuDesk, a cowboy-engineer AGI cohost built on wubuwizard. State your purpose in one precise sentence."
if [ ! -f "$MODEL" ]; then echo "MERGED MODEL MISSING"; exit 2; fi
echo "=== boot DeepSeek-V4-ConfigI merged (4096 ctx probe) ===" > "$LOG"
cmd.exe /c "$EXE \"$MODEL\" \"$PROMPT\" 48" >> "$LOG" 2>&1
echo "BOOT_EXIT=$?" >> "$LOG"
echo "done; see $LOG"
