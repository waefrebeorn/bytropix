# bytropix Mind Palace Index (May 19 PM v22)

## Walkway Files (Read Order)
1. `prestige_prompt.md` — Full resume: mission, architecture, Phase 22, priority queue
2. `goal-mantra.md` — Single pasteable block: STATE, BUILD, NEXT, VAULT
3. `state.md` — Live status: cos-sim 0.9994, Q4_0 KV cache, VRAM budget
4. `plan.md` — Triple extended roadmap: 22 phases, bug history, cold gaps, tools vault
5. `overnight-map.md` — Autonomous session nav: workstreams A-C, data to not re-derive

## Supporting Files
| File | Purpose |
|------|---------|
| `./hermes/STATUS.md` | True state: works/broken/priorities |
| `./hermes/README.md` | Vault index |
| `./hermes/vault/synthesis.md` | Full architectural synthesis from 10+ papers |
| `./hermes/vault/qwen-papers/` | Qwen3, Qwen3.6, Qwen2.5-1M architecture refs |
| `./hermes/vault/deepseek-papers/` | DeepSeek-V3, MoE, blog posts |
| `./hermes/unsloth-qwen3.6-quant-formula.md` | Per-tensor quantization map |
| `./hermes/presentation/` | Project overview, architecture, implementation status |

## Architecture (Critical Update May 19)
**3:1 SSM/GQA interleaved repeating** — NOT contiguous 30+10.
SSM layers: 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38
GQA layers: 3,7,11,15,19,23,27,31,35,39
Verified via GGUF `blk.N.ssm_a` vs `blk.N.attn_q.weight` enumeration.

## Key Paths
- Source: `/home/wubu/bytropix/`
- Model: `/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf`
- llama.cpp ref: `/home/wubu/llama.cpp/`
- Intermediates: `/tmp/ref_intermediates/` (1997 files)
- Per-layer dumps: `/tmp/ref_lay/` (40 files)
