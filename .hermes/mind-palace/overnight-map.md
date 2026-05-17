# WuBuText AI — Overnight Navigation Map (May 17 v4 — MoE expert layout bug)

## Where We Are
MoE expert extraction bug FOUND via Devil's Advocate audit. The output projection transpose fix removed the anti-correlation (cos-sim -0.457→-0.001) but revealed the deeper MoE bug. Expert tensor dims=[2048,512,256] with expert as innermost dim — data interleaved, dequant code reads contiguous.

## What Changed This Session
- Output projection TRANSPOSE found and fixed (weight[j*D_MODEL+k]→weight[k*vocab_size+j])
- Reference extraction tool: dump_llama_logits now dumps BOTH logits + hidden states
- MoE expert layout discovered: interleaved, not contiguous per expert
- DA audit of all claimed verifications → 3 false ✅s identified and stripped

## Build
```bash
rm -f infer_text; make infer_text
```

## Next Step
Fix `dequant_one_expert_contiguous` to stride-extract per expert from the interleaved blocks. Each IQ2_XXS block (66 bytes, 256 values) encodes ALL 256 experts' values at ONE (i,j) position. Extract float[eid] from each dequantized block.

## Reference Files
- Prestige: /home/wubu/bytropix/.hermes/mind-palace/prestige_prompt.md
- State: /home/wubu/bytropix/.hermes/mind-palace/state.md
- Goal: /home/wubu/bytropix/.hermes/mind-palace/goal-mantra.md
- Plan: /home/wubu/bytropix/.hermes/mind-palace/plan.md
- Skill: ~/.hermes/skills/mlops-inference/bytropix-moe-expert-layout/SKILL.md
- llama_ref: /home/wubu/llama.cpp/src/models/qwen35moe.cpp, qwen3next.cpp

## Useful Commands
- Dump tensor: `./dump_tensor_our /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf <tensor_name> /tmp/out.bin`
- Run inference: `NOGPU=1 MOE=1 MAX_LAYERS=40 DUMP_LOGITS=/tmp/out.bin ./infer_text /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "Hello" 1 1`
- Reference: `./dump_llama_logits /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf /tmp/ref.bin "Hello"`
