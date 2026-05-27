# vault/runner-analysis.md — May 27, 2026

## Remaining tasks after output proj fix

### 1. run-harness.sh — patch to serve_local.py
`~/hermes-test/run-harness.sh` uses `inference-server.py` (proxy to DeepSeek). Replace with `serve_local.py`:
```
# Replace:
python3 tools/inference-server.py --port 8001 ...
# With:
MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf \
  OMP_NUM_THREADS=4 python3 tools/serve_local.py --port 8001
```

### 2. NES emulator PPU
`~/hermes-test/projects/nes-emulator/` needs proper tile/nametable rendering + iNES loader.

### 3. test-hermes-headless.sh
Uses proxy sandbox — update for real local mode.
