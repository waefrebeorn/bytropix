# State — May 19, 2026 (Triple DA v6 Complete)

## TRIPLE DA FINDINGS (May 19)
```
DA-1: SSM recurrence math IDENTICAL (1/sqrt(128) scale, exp(gate), same state update)
DA-2: All vault papers cross-referenced. No theoretical gaps.
DA-3: Cold gap = quantized matmul precision, NOT algorithm
```

## Current Reality
| Metric | Value | Status | Evidence |
|--------|-------|--------|----------|
| Logit cos-sim vs llama.cpp (BOS) | **0.7944** | ✅ Pre-existing at IQ2_M | Verified this session — both agree top-1=220 |
| Per-layer cos-sim (avg 40) | **0.88** | ✅ New measurement | Range 0.45–0.97, all 40 layers unique |
| SSM recurrence math | **IDENTICAL** | ✅ Verified vs ggml_gated_delta_net kernel | 1/sqrt(128) scale, same decay/output formula |
| Top-1 agreement (BOS) | token **220** | ✅ Both implementations agree | bytropix=220, llama.cpp=220 |
| BOS embedding match | **cos=1.0** | ✅ File vs GGUF-extracted | Two independent bytropix runs identical |

## Infrastructure Built This Session
1. `/home/wubu/llama.cpp/src/llama-graph.h` — `t_layer_h` vector for per-layer extraction
2. `/home/wubu/llama.cpp/src/models/qwen35moe.cpp` — `ggml_set_output` + `t_layer_h.push_back`
3. `/home/wubu/llama.cpp/src/llama-context.cpp` — Deep-copy F32 dump on `DUMP_LAYER_DIR`
4. `/home/wubu/bytropix/tools/run_bos.c` — Standalone single-token forward pass
5. `/home/wubu/bytropix/tools/ref_dumper.cpp` — Already existed, uses libllama

## Critical Knowledge
- `ggml_set_output()` REQUIRED to prevent scheduler buffer reuse
- `ggml_gated_delta_net` is a custom GGML op — NOT manual C code in llama.cpp
- SSM recurrence uses `scale = 1/sqrt(S_v)` where S_v = 128 = ssm_d_state
- Quantized matmul dequant precision is the divergence source, not algorithm
- BOS 248044 for Qwen3.6-35B. Top-1 = 220 for single BOS forward.

## Tools Vault (tmp code copied)
- `run_bos` at `/home/wubu/bytropix/run_bos`
- `ref_dumper` at `/home/wubu/bytropix/ref_dumper`  
- Layer dumps: `/tmp/dump_layers_ref/` (llama.cpp), `/tmp/dump_layers_our/` (bytropix)
