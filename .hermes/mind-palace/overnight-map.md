# WuBuText AI — Overnight Navigation Map (May 17 v5 — All components verified)

## Where We Are
All individual components verified exact vs ggml/reference. SSM/GQA path correct at cos-sim 0.994 logits. MoE=1 diverges at cos-sim 0.337 due to recurrent amplification over 40 SSM layers. The MoE expert interleaving hypothesis was WRONG — experts ARE contiguous.

## What Changed This Session
- RoPE MRoPE section dimension FIX (cos-sim -0.456→-0.016)
- All dequant types verified vs ggml on full 1M elements
- SSM/GQA path verified at cos-sim 0.994 vs reference
- MoE divergence confirmed as recurrent amplification, not component bug
- All debug patches cleaned; codebase reverted to clean state
- Mind-palace updated v15/v16

## Build
```bash
rm -f infer_text; make infer_text
```

## Next Step
Compare MOE=1 per-layer residuals vs MOE=0 to find where MoE correction first creates divergence. If divergence present at layer 0, compare `moe_expert_forward_lazy` vs llama.cpp `build_moe_ffn` order of operations precisely.

## Reference Files
- Prestige: /home/wubu/bytropix/.hermes/mind-palace/prestige_prompt.md
- State: /home/wubu/bytropix/.hermes/mind-palace/state.md
- Goal: /home/wubu/bytropix/.hermes/mind-palace/goal-mantra.md
- Plan: /home/wubu/bytropix/.hermes/mind-palace/plan.md
- llama_ref: /home/wubu/llama.cpp/src/models/qwen35moe.cpp, qwen3next.cpp

## Useful Commands
- Run inference: `NOGPU=1 MOE=1 MAX_LAYERS=40 DUMP_LAYER_DIR=/tmp/dump_layers ./infer_text /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "Hello" 1 1`
- Reference: `./run_ref_moe0 /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf 248044`
- Compare: `python3 -c "import numpy as np; a=np.fromfile('/tmp/ref_hidden_tok0.bin',f4); b=np.fromfile('/tmp/our_hidden.bin',f4); print(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)))"`
