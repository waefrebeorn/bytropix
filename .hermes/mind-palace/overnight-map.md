# WuBuText AI — Overnight Navigation Map (May 16 v16 — DA AUDITED)

## Where We Are
All P0-P3 infrastructure complete. Output STILL WRONG — root cause unknown.
Embeds and BOS were both wrong but fixing them didn't change output.

## What's New
- Auto-embedding extraction from GGUF at model load time
- BOS handling gated by ADD_BOS env var
- DA: "noise floor" argument debunked — two engines MUST match on same quantized GGUF

## Remaining (critical)
- Output token "Plot" instead of "Here" — hidden layer bug
- Possible: Q5_K dequant, SSM recurrence, GQA dimensions, output weight dequant
- GGUF data blob prevents in-memory token_embd + out_weight (OOM)
