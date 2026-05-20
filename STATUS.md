# STATUS — bytropix Inference Engine (May 20, Phase 28e)

**GPU SSM decode (C==1): ~5.9 tok/s | Q6_K dequant BUG FIXED | GPU still diverges from CPU (cos-sim -0.66)**

## What Works (✅ Verified at runtime)
- GPU `gen_text_gpu`: full 40-layer, all 30 SSM layers on GPU, Q4_0 KV cache
- Q6_K dequant offset corrected (was `32.0`, now `d*sc*(v6-32)`)
- CPU SSM path matches llama.cpp at cos-sim 0.994 (FORCE_CPU_SSM)
- Fused SSM kernels (beta/alpha, conv/silu/split, L2 norm, recurrence, gated norm, ssm_out)
- Vision encoder: 384 LoC 3D ViT port + mmproj projection + text model pipeline
- F32 dequant waste removed (`#if 0`, saved ~2.2 GB VRAM)
- **API server: `tools/api_server.c` — OpenAI-compatible HTTP API (sandbox verified)**

## Remaining Bugs
- ❌ GPU SSM state divergence: cos-sim -0.66 vs CPU path (anti-correlated)
- ❌ CPU `gen_text` build broken (GPU symbols in wubu_model.o without .cu objects)
- 🔴 8 commits not pushed to remote

## Phase 29 Plan
1. Layer-by-layer GPU vs CPU hidden state comparison → find first divergence point
2. Check recurrence state persistence between layers on GPU
3. Check conv state initialization and shifting on GPU
4. Fix state management → cos-sim > 0.99

## Phase 30-35 Roadmap (Feature Cream)
| Phase | Theme | Key Deliverable |
|-------|-------|----------------|
| 30 | Infrastructure | Fix CPU build, push commits |
| 31 | Vision verification | Build + run test_vision_real E2E |
| 32 | Multi-modal inference | Vision→text full pipeline |
| 33 | Feature cream | Sigmoid gating, load balancing, chunked prefill, RoPE ext |
| 34 | 256K multi-modal | Full context vision+text at scale |
| 35 | Profile & optimize | CUDA events, bottleneck analysis, 10+ tok/s target |

## Build
```bash
make gen_text_gpu       # GPU inference (GPU=1 env var)
make test_vision_real   # Vision encoder test
make api_server         # OpenAI-compatible API server
```
