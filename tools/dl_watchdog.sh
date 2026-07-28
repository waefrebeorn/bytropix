#!/bin/bash
# Watchdog: runs the Colonel downloader; if it exits, restarts it.
# Self-detaches via setsid on first invocation so the worker survives even
# if the Hermes-tracked parent shell is reaped. Uses a lockfile so only ONE
# downloader ever runs (prevents concurrent writers corrupting shards).
export HF_TOKEN="${HF_TOKEN:-***REMOVED***}"
cd /home/wubu/bytropix
LOG=/home/wubu/bytropix/logs/dl_all_curl.log
PIDF=/home/wubu/bytropix/logs/dl_pid.txt
LOCK=/home/wubu/bytropix/logs/dl.lock

# Self-detach: if not already in our own session, re-exec under setsid and exit.
if [ -z "${DL_WATCHDOG_DETACHED:-}" ]; then
  export DL_WATCHDOG_DETACHED=1
  exec setsid bash "$0" "$@" < /dev/null >> "$LOG" 2>&1 &
  echo $! > "$PIDF"
  exit 0
fi

# Single-instance lock (non-blocking).
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "[watchdog $(date +%H:%M:%S)] another instance holds lock — exiting" >> "$LOG"
  exit 0
fi

echo "[watchdog $(date +%H:%M:%S)] detached worker started pid=$$" >> "$LOG"
MAX=50
n=0
while [ $n -lt $MAX ]; do
  echo "[watchdog $(date +%H:%M:%S)] run #$n" >> "$LOG"
  bash tools/download_all_colonels.sh >> "$LOG" 2>&1
  rc=$?
  echo "[watchdog $(date +%H:%M:%S)] run #$n exited rc=$rc" >> "$LOG"
  if [ $rc -eq 0 ]; then echo "[watchdog] ALL DOWNLOADS DONE" >> "$LOG"; break; fi
  n=$((n+1))
  sleep 5
done
echo "[watchdog] finished (cycles=$n)" >> "$LOG"
