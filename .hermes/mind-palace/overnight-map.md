# WuBuText AI — Overnight Navigation Map (May 17 PM v9 — HONEST)

## Where We Are
**1 minor fix + 1 discovery this session:**
- IQ1_M dequant fixed (3 bugs: scale idx, missing dl1/dl2, -1.0f delta). Not used in this model.
- **CRITICAL DISCOVERY**: Python `dump_gguf.py` has WRONG type labels (18→IQ2_S should be IQ3_XXS, 23→IQ1_M should be IQ4_XS). Prior DA analysis was misled.

**1 bug remains: SSM divergence at L0 (~0.40 cos_sim vs reference)**

## Actual Down_exps Tensor Types (corrected)
- IQ3_XXS (type 18): layers 0-33, 35-37 (37/40)
- IQ4_XS (type 23): layers 34, 38, 39 (3/40)

## Build Command
```bash
cd /home/wubu/bytropix && make infer_text
```

## Reference
```bash
cd ~/llama.cpp/build/bin && ./llama-cli -m /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf -p "Hello" -n 5 --temp 0.0
```

## Priority for Next Session
1. **P0: Find SSM root cause** — SSM diverges at L0 (cos_sim 0.40) before MoE. Not dequant. Check conv1d, recurrence, output proj.
2. **P1: Verify IQ4_XS dequant** — cross-dump vs llama.cpp on actual IQ4_XS tensor (L34 down_exps)

## Known Bugs
- SSM output wrong at L0 (cos_sim ~0.40)
- Output: `<|endoftext|>Hello_vendor` vs ref "Hello Here's a"

## Fixed This Session
- IQ1_M dequant (3 bugs: scale index ib/4→ib/2, added dl1/dl2, removed -1.0f delta)
- Python `dump_gguf.py` type labels corrected

Archived: `overnight-map-v8-May17-PM.md` in `vault/bins/`.
