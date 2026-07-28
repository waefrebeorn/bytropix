#!/bin/bash
# Watchdog: runs the Colonel downloader; if it exits, restarts it (up to N times)
# so a transient curl death never stalls the full pull. Logs to logs/dl_all_curl.log
# (which must exist; created by the caller).
export HF_TOKEN="${HF_TOKEN:-***REMOVED***}"
cd /home/wubu/bytropix
LOG=/home/wubu/bytropix/logs/dl_all_curl.log
MAX=20
n=0
while [ $n -lt $MAX ]; do
  echo "[watchdog $(date +%H:%M:%S)] run #$n" >> "$LOG"
  bash tools/download_all_colonels.sh >> "$LOG" 2>&1
  rc=$?
  echo "[watchdog $(date +%H:%M:%S)] run #$n exited rc=$rc" >> "$LOG"
  if [ $rc -eq 0 ]; then echo "[watchdog] DONE" >> "$LOG"; break; fi
  n=$((n+1))
  sleep 5
done
echo "[watchdog] finished (rc-cycle=$n)" >> "$LOG"
