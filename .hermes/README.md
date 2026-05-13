# WuBuText AI — Project Root

Building a pure-C language model with WuBu nested hyperbolic geometry,
using Qwen3.6-35B-A3B as the embedding source, running on RTX 5050 6.4GB VRAM.

## Quick Links

### Mind Palace (project planning)
- **Entry**: `.hermes/mind-palace/README.md` — start here
- **Tier 1 (Core)**: WuBu theory, arch reference, C baseline
- **Tier 2 (Research)**: DeepSeek, Qwen, fast attention, hyperbolic papers
- **Tier 3 (Implementation)**: Embedding graft → attention → training → MoE → vision
- **Tier 4 (Validation)**: Benchmarks, debugging

### Plans
- **Devil's Advocate**: `.hermes/plans/2026-05-12-devil-advocate-roadmap.md`
- **1000-Step Roadmap**: `.hermes/plans/2026-05-12-wubuGPT-roadmap.md`

### Research
- **Papers**: `.hermes/research/papers/`
- **Qwen**: `.hermes/research/papers/Qwen/` — full architecture references

### Source Code
- **Current C baseline**: `include/` + `src/` (768 hidden, 6 layers, GQA)
- **JAX reference**: `wubu_nest_gpt_v2.py` (63M params, MLA+MoE+gyration)

### Lean Proofs
- `MATH/lean/wubu_proofs/` — 4 verified proofs

## Build Order
1. GGUF reader → extract Qwen embeddings
2. Poincaré mapping → verify quality
3. Gated DeltaNet in C → hyperbolic gyration
4. Training loop → CUDA kernels
5. MoE routing → nested geometry
6. Vision encoder

## Target Spec
- Model: Qwen3.6-35B-A3B (3B active params)
- Hidden: 2048, Layers: 40, Vocab: 248320
- Attention: 75% Gated DeltaNet (hyperbolic), 25% GQA
- MoE: 256 experts, 8 active + 1 shared
- Context: 262K native, 1M extensible
- Hardware: RTX 5050 6.4GB VRAM
- Language: Pure C with CUDA kernels
