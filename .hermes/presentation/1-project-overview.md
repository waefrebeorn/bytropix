# WuBuText AI — Project Overview (May 19 PM v22)

## What We're Building
**bytropix** — pure C inference engine for Qwen3.6-35B-A3B (Gated DeltaNet + MoE).
Cos-sim 0.9994 vs llama.cpp. 256k context on 8GB laptop GPU.

### Architecture (Discovered May 19)
40 layers with **3:1 SSM/GQA interleaved repeating** pattern (NOT 30+10 contiguous).
SSM: 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38
GQA: 3,7,11,15,19,23,27,31,35,39

### All 22 Phases Complete ✅
| Phase | Component | Key Result |
|-------|-----------|------------|
| 0-11 | Foundation | GQA attn, vec_dot, MoE, KV cache |
| 12 | MTP Spec Decode | Free-tokens mode |
| 13 | GPU Output Proj | 0.1ms vs CPU 10ms |
| 14-17 | GPU SSM, MoE, Recurrence | All on GPU |
| 18-21 | Full GPU pipeline + sliding window | 9 tok/s at 256k |
| **22** | **Q4_0 KV cache + architecture discovery** | **4:1 compression, interleaved pattern** |

### Key Innovations
- Q4_0 KV cache: 720MB vs 2.56GB at 256k
- DUMP_INTERMEDIATE_DIR: 53 tensor types/layer reference tracing
- Self-hosted vec_dot: zero dependency on libggml-cpu.so
- GPU pipeline: sliding window, MoE cache, SSM recurrence
