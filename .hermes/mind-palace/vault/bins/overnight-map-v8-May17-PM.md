# WuBuText AI — Overnight Navigation Map (May 17 PM v8 — HONEST)

## Where We Are

**2 bugs fixed this session:**
- IQ3_XXS block size 104→98 (MoE down_exps stride bug, MoE output rms 690k→0.25)
- IQ4_XS enum + dequant + raw_size support added

**1 bug remains: SSM divergence at L0 (~0.40 cos_sim vs reference)**

## DA v8 Findings

- "SSM verified vs Python (cos_sim=1.0)" — survivorship bias. Only checks Python=C consistency, NOT correctness vs llama.cpp. NEED to compare gegen reference.
- "IMRoPE is root cause" (DA v3/v7) — CONFIRMED STALE. SSM doesn't use RoPE. SSM-only cos_sim=0.018 proves independent SSM bug.
- IQ4_XS dequant: written from ref but never empirically tested against IQ4_XS tensor data.
- Model config values (D_MODEL, D_FF, etc.) hardcoded, not verified from GGUF metadata.

## Build Command
```bash
cd /home/wubu/bytropix && make infer_text
```

## Reference
```bash
cd ~/llama.cpp/build/bin && ./llama-cli -m /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf -p "Hello" -n 5 --temp 0.0
```

## Priority for Next Session
1. **P0: Verify SSM formula against llama.cpp** — dump QKV, recurrence, output from both, find exact divergence point
2. **P1: Verify model config from GGUF metadata** — check hidden_size, n_heads, etc. match hardcoded
3. **P2: Test IQ4_XS dequant on actual IQ4_XS tensor**

## Known Bugs
- SSM output wrong at L0 (cos_sim ~0.40)
- Possible: SSM output weight dequant or shape
- Possible: SSM recurrence formula differs from reference

## Fixed This Session
- IQ3_XXS block size 104→98 (MoE down_exps garbage → correct)
- IQ4_XS support (enum, raw_size, dequant function)
- DA v8 written with honest audit

Archived: `overnight-map-v7-May16-AM.md` in `vault/bins/`.
