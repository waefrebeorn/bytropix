# 20-Pillar Architecture → wubuwizard / WuBuOS / slermes Mapping

## Completed / Strong (✅)
| # | Pillar | Status | Code |
|---|--------|--------|------|
| 1 | Pure-C inference engine | ✅ Complete | wubuwizard: `src/wubu_model.c`, `src/wubu_ssm.c`, `src/wubu_moe.c`, quant matmul, GGUF+safetensors readers |
| 3 | Native agent runtime (no Python) | ✅ Complete | slermes: pure C11 agent binary, `src/agent/`, Telegram/Discord/Slack/Signal/WhatsApp/Matrix gateways |
| 7 | Persistent structured memory | ✅ Complete | slermes: `src/tools/memory.c`, `src/tools/memory_storage.c`, session DB |
| 10 | Error recovery & self-critique | ✅ Implemented | wubuwizard: `wubu_repetition.c` (repeat_penalty + DRY), `src/wubu_model_adapter.c` (LoRA merge) |
| 12 | Resource & stability management | Partial | wubuwizard: `src/wubu_affinity.c` (P-core pinning), `src/wubu_arena.c` (bounded alloc), ds4-ssd slot-bank |
| 14 | Multi-platform human interface | ✅ Partial | slermes: CLI + TUI + Wayland/X11/Win32/macOS GUI; WuBuOS: Win98/XP shell |
| 16 | Continuous autonomy (scheduling) | ✅ Implemented | slermes: `src/cron/scheduler.c`, `cronjob` tool in Hermes |
| 19 | Pure-C integration bus | ✅ Complete | wubuwizard: `include/wubu_*.h` opaque-struct API, `src/wubu_*.c` modules |

## In Progress (🔄)
| # | Pillar | Status | Code |
|---|--------|--------|------|
| 2 | Custom OS kernel/RT for long-running intelligence | 🔄 Partial | WuBuOS runs as hosted binary on Linux; ZealOS kernel exists but not booting metal on this box |
| 4 | Document/productivity engine | 🔄 Stub | WuBuOffice not started in this checkout |
| 5 | Code editor / dev environment | 🔄 Stub | WuBuPad not started in this checkout |
| 6 | Unified tool surface | 🔄 Partial | slermes: 40+ tools wired; wubuwizard has `gen_text`, `test_*`, `api_server` |
| 8 | Hierarchical planning / multi-agent orchestration | ❌ Missing | No planner, no sub-agent spawning in slermes or wubuwizard |
| 11 | Continuous evaluation / benchmarking | 🔄 Partial | wubuwizard: `test_*` suite for inference; no multi-hour computer-task benchmark |
| 13 | Sandboxing / permission model | ❌ Missing | slermes has no sandbox; WuBuOS has cgroups/styx but not agent sandbox |
| 15 | Web access / external world | ❌ Missing | No browser, no web search tool in slermes (Hermes has it but slermes reimpl hasn't ported it yet) |
| 17 | Skill creation / self-improvement curriculum | ❌ Missing | slermes has skills parsing but not self-modifying skill extraction |
| 18 | Geometric/math research foundations | 🔄 Research | `ENCODERS/`, `THEORY/` in wubuwizard (Poincaré, GAAD, DFT/DCT) — not yet wired into inference |
| 20 | Sustained autonomous productivity demo | ❌ Not yet | Requires all pillars integrated |

## Key Gaps to Close (in priority order)
1. **Pillar 8 — Hierarchical planning**: No planner in slermes or wubuwizard. Add a goal-decomposition loop to slermes `src/agent/`
2. **Pillar 13 — Sandboxing**: slermes runs with full host access. Need container isolation for agent processes
3. **Pillar 15 — Web access**: slermes has no web search/browser tool. Needs `web_search` + `web_extract` wired as C tools
4. **Pillar 17 — Self-improving skills**: slermes reads skills but doesn't create/refine them autonomously
5. **Pillar 11 — Benchmark suite**: Needs a multi-hour computer-task eval harness
6. **Pillar 2 — OS kernel**: WuBuOS hosted binary works; needs metal boot on WSL for full control

## WSL as Agnostic Accelerator Design
WSL2 on this machine is the current compute substrate:
- 6 P-cores pinned (core0=0), F16 KV-cache, llvmpipe Vulkan
- ~13 GB RAM available to the process
- No dedicated GPU (no `/dev/dri`), only CPU compute + Vulkan software rasterizer
- The `wubu_affinity.c` already pins the engine to P-cores; this is the correct starting point for an agnostic accelerator abstraction

Next step: build a `wubu_accel.c` / `wubu_accel.h` that abstracts the compute backend (CPU AVX512, CUDA, Vulkan/llvmpipe) behind a uniform interface, so the inference engine, agent runtime, and document tools all route through the same accelerator surface regardless of which hardware is present.

## WuBuOS AGI Design Orientation
WuBuOS = ZealOS kernel + Win98 shell + Styx/9P namespace + Arch containers.
The 20 pillars map directly to WuBuOS subsystems:
- Pillars 1-3 → `src/bear/` (RL training) + inference engine (wubuwizard)
- Pillar 2 → ZealOS kernel in WuBuOS (boot on metal, not yet WSL-hosted)
- Pillar 4 → WuBuOffice (C11 OOXML/ODF/PDF)
- Pillar 5 → WuBuPad (piece-table editor)
- Pillar 6 → Styx/9P tool surface
- Pillars 7-10 → slermes memory + session + error recovery
- Pillars 11-13 → WuBuOS container isolation (cgroups + seccomp-bpf)
- Pillars 14-16 → Slermes TUI/CLI + cron scheduler
- Pillars 17-20 → Integration goal across all repos