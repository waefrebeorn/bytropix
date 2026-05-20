# MoE vs Dense Offload on 8GB VRAM

**Source**: witcheer (X/Twitter, May 18 2026)
**Hardware**: RTX 4060 Ti 8GB, Windows
**Dataset**: https://huggingface.co/datasets/witcheer/windows-rtx-4060ti-8gb-moe-offload-bench-2026-05
**Collection**: https://huggingface.co/collections/witcheer/8gb-vram-local-llms-practitioner-tested-69fa0e855c51e3c15a9d95d4

## Key Result
| Model | Active Params | Speed |
|-------|:-:|:-:|
| Qwen3.6-35B-A3B (MoE, -ncmoe 30) | 3B | **35.4 tok/s** |
| Qwen3.6-27B (dense, -ngl 20) | 27B | 3.28 tok/s |
| Ratio | | **10.8x** |

At 24K context: gap grows to **16.7x** (MoE: zero context degradation via SSM, dense: -35.4%)

## Why MoE wins on 8GB
- MoE expert offload keeps the hot path (3B active params) entirely in VRAM
- Only inactive experts move to CPU when selected
- Dense layer offload splits every layer across GPU/CPU → every token bounces through PCIe for all 64 layers
- SSM layers have zero context degradation (unlike dense attention)

## Quality
- Dense 27B: 5/6 hallucination rating (best of 9 tested)
- MoE 35B: 4/6
- Trade-off: speed vs quality

## Key Takeaway for bytropix
- Our approach (keep 3B active params on GPU, quantized weights for inactive experts) is validated by this external benchmark
- 35.4 tok/s on RTX 4060 Ti 8GB is the target to beat
- The benchmark confirms: MoE with expert offload is 10-17x faster than dense layer offload on 8GB VRAM
- SSM layers provide zero context degradation - major advantage at 256k
