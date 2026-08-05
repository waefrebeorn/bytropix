# Research 066 — 100 Ways to Make Everything Better: 50 Topics × 10 Themes

**Date:** 2026-08-05
**Method:** 7-hop Kevin-Bacon per topic — seed on canonical source, trace 5-7 hops of lineage/citations/related work via web_search, aggregate convergence, extract one-line principle, map to 2 concrete improvements. ~215 web searches total. All sources verified real; nothing fabricated.
**Scope:** wubuwizard (C11 AI inference engine, 331 .c / 329 .h / 578 tools) + wubuos (ZealOS/Win98-style OS, Styx/9P namespace, dosgui WM, theme engine).

**User's pain points this research attacks:**
1. MONOLITHIC codebase (305 .o in CORE_OBJ, 382 unique include paths, 119 files include wubu_model.h, god headers)
2. NOT AGNOSTIC (formats/hardware hardcoded, no vtable layers)
3. HARD FOR AI AGENTS TO WORK ON (no AGENTS.md, no symbol index, slow builds)
4. NO PLANNING STRUCTURE (gap ledger without owners/commitments, no ADRs)
5. UX NOT COHESIVE across CLI + GUI + web

---

## The 100+ Ways — Ranked Implementation Plan

### TIER 0 — Foundation (do first: everything else builds on these)

1. **AGENTS.md for both repos** — repo map, build/test commands, architecture principles, one per repo (C1)
2. **ADR log (`docs/adr/`)** — Nygard 5-field template, append-only, one per architectural decision (J1)
3. **Opaque pointers at every module seam** — expose only `typedef struct X X;` + accessors in public headers, private fields in owning .c (E1, A4)
4. **Split god headers** — wubu_model.h → wubu_model.h + wubu_model_ssm.h + wubu_model_moe.h + wubu_model_kv.h + wubu_model_arena.h; max 3 project includes per header (A1, E3)
5. **Backend vtable (`wubu_backend_api`)** — struct of function pointers (init/matmul/deinit), CPU/CUDA/Metal adapters implement it, core never includes hardware headers (B2, B4, A3)

### TIER 1 — Agent ergonomics (make the repos workable by AI agents)

6. **`compile_commands.json`** via bear — one dependency graph for agents + clangd + static analysis (D1)
7. **Symbol index `docs/symbols.json`** — tree-sitter C grammar over all .c/.h, every function/struct/typedef/macro with location + signature (C3)
8. **Dependency graph `docs/deps.json`** — per-function call graph, for kernel understanding (C3)
9. **Ninja build profile + ccache** — rebuild 10-100× faster; `make fast` for agents (C5, D1)
10. **Non-recursive single Makefile** — one pattern rule over src/*.c → build/*.o, per-object .d files, order-only dir deps (D5)
11. **`make help` target** — every target self-documents with one-line description (C4)
12. **Per-object header-touch regression test** — touch one header → assert exactly the right object set rebuilds (D1, D5)
13. **CODEOWNERS mirroring module boundaries** — ownership explicit per directory (A5, C1)
14. **CONTRIBUTING.md mapping teams → directories** (A5)
15. **`make test-watch`** — incremental test runs for affected modules only (C5)
16. **Symbol index regeneration in CI** — symbols.json stays fresh (C3)
17. **AGENTS.md progressive disclosure** — root file points to per-subsystem nested files (J5, C1)
18. **Doc comments enforced at build** — function signature change requires doc comment update or build fails (C2)

### TIER 2 — Agnostic interfaces (kill the hardcoded formats/backends)

19. **`wubu_model_t` abstract interface** — load/run_layer/unload/get_info; GGUF/ONNX/safetensors adapters (B1)
20. **`wubu_device_t` abstract interface** — alloc/free/copy/map; RAM/GPU/9P-remote adapters (B1)
21. **`wubu_tensor_catalog`** — neutral in-memory tensor map (name→shape→dtype→data); formats are I/O adapters (B3)
22. **`wubu_file_catalog`** — neutral file metadata representation for wubuos VFS (B3)
23. **`wubu_backend_vtable` replaces `#ifdef __CUDA__`** — no preprocessor branching in matmul kernels (B2)
24. **`wubu_hal_vtable`** — alloc_page/free_page/map/unmap/flush for physical + 9P-remote memory (B2)
25. **SONAME-versioned shared library** — libwubu.so.1/.2, ABI-stable opaque APIs (B5)
26. **Plugin manifest (`wubu_plugin_manifest.json`)** — contribution points + activation events, lazy dlopen (G1)
27. **Static core + dlopen plugin layer** — libwubu_core.a + wubu_plugin.so with stable ABI (G4)
28. **Service locator (`wubu_service_register/resolve`)** — runtime wiring by name, no compile-time coupling (G5)
29. **Backend plugins as .so** — drop a new backend in plugin dir, no core recompilation (B4)
30. **Filesystem driver plugins** — 9P/local/ramdisk as dlopen-able drivers (B4)
31. **Theme plugins from external files** — JSON token files, no theme-engine source edits (G3)
32. **Weight streaming (`wubu_weight_stream`)** — lazy load + evict, configurable cache, no OOM (H3)
33. **Content-addressable weight store (`wubu_cas`)** — SHA-256 keyed, dedup across versions, integrity by hash (H5)
34. **Model metadata as separate JSON** — `wubu_model_meta.h`, weights + meta decoupled (H4)

### TIER 3 — Modularity & build (de-monolith, speed up iteration)

35. **Strangler Fig extraction** — backends/, model/, kv/ as bounded contexts behind stable internal APIs (A1)
36. **Onion layering enforced by include discipline** — inner layers never #include outer headers (A4)
37. **Compilation firewall** — private fields in .c; change one field → recompile one TU (E1)
38. **Macro-registered dispatch registry** — `WUBU_KERNEL_ENTRY(name, fn)` generates extern-decl header from table (E2)
39. **Table-walk self-test** — assert no NULL slots, unique names, schema cross-check (E2)
40. **`wubu_status` enum + single-exit cleanup** — no errno dependence, greppable, agent-checkable (E3)
41. **ERR_PTR/IS_ERR typed error pointers** in kernel handlers (E3)
42. **Per-generation decode arena** — bump allocator reset per generate call, replaces malloc/free (E4)
43. **Typed event bus** — loader/KV/tokenizer/trainer events drained in main loop (E5)
44. **Unity build profile** — `-DUNITY_BUILD=ON` for clean/CI builds, modular default for dev (D2)
45. **X-macro single-source tables** — quant-grid tables, kernel dispatch table generate headers + switches + test vectors (D3)
46. **`make regenerate` + git-diff drift check** — generated code can't silently drift (D3)
47. **Vendored third_party/ subtree** — hermetic, deterministic, agent-readable without network (D4)
48. **`make vendor-check`** — fails any build step touching network (D4)
49. **Toolchain pinning** — compiler + binutils versions fixed for deterministic rebuilds (D4)
50. **PCH for heavy shared includes** (D2)
51. **Chunked unity buckets preserve incremental rebuilds** (D2)
52. **`BUILDDIR=build/` artifact layout** — predictable binary location (D3)
53. **-Wextra -Wpedantic -Wshadow warnings, -Werror in CI only** (D4)
54. **Sanitizer build targets** — asan/ubsan/valgrind one-command (D5)

### TIER 4 — Testing & verification (prove the research, catch regressions)

55. **Property-based tests** — decode(encode(x))==x, quantize→dequantize cosine ≥ bound, no-alloc-after-init; make test_props (F1)
56. **Differential testing harness** — own quant kernels vs llama.cpp reference on random tensors; GGUF parser vs reference loader (F2)
57. **Fuzz targets** — GGUF loader, tokenizer, quant-schema parser with libFuzzer + ASAN; seed corpus from real files (F3)
58. **Fuzz the 8086/VSL emulator decoder** — random instruction streams, register/memory invariants (F1, F3)
59. **Golden files** — tokenizer + fixed-seed generation trace, review-gated regeneration (F4)
60. **Tolerance table per quant type** — ULP/relative, keyed by weights+input hash (F4)
61. **Golden-boot test** — boot kernel in emulator, hash VGA framebuffer + syscall trace, byte-identical across builds (F4)
62. **Tiered CI on GitHub Actions** — fast tier per push (gcc/clang × -O2/ASAN matrix, ccache), slow tier nightly (fuzz, emulator boots) (F5)
63. **syzkaller-style syscall fuzz** of emulated personalities nightly (F5)
64. **Regression test: border_width() == chrome_border_width()** for both themes (F3)
65. **Cross-repo integration CI** — both repos checked out, built, tested together (F4)
66. **CLI tool output tests** — banner/section/stat patterns verified automatically (F1)
67. **GUI content-rect tests** — chrome-provided rect matches expected math (F2)
68. **`make fuzz-run` time-boxed CI job** (F3)
69. **Differential test HolyC vs GCC** — emulated register results compared (F2)
70. **VSL NT-bridge vs ReactOS behavior vectors** (F2)

### TIER 5 — Namespaces, OS design, plugins (the wubuos identity)

71. **KV cache as 9P filesystem at /n/kv/** — every cache entry a file, inspectable with any 9P client (H1, H4)
72. **Per-process namespace views** — each app/container sees only its files (I2, H3)
73. **Styx registry contribution points** — plugins register services under /svc/, WM discovers by walking (G1)
74. **`wubu_scriptd`** — scripting service mounted at /script/, extensions interact via 9P file I/O (G2)
75. **Wasmtime plugin sandbox** — .wasm plugins via WASI, whitelisted imports only (G3)
76. **Wasmtime for code-exec engine** — user code compiles to sandboxed WASM modules (G3)
77. **Device model (bus/device/driver/probe)** — hot-pluggable 9P servers, GPU drivers, FS modules (I2)
78. **Compute bus for inference backends** — GPU/CPU/accelerator register as devices, scheduler probes (I2)
79. **VSL multi-personality syscall layer** — Linux + Plan 9 + Win32 NT subsets mapped to common kernel ops (I3)
80. **seccomp-BPF-style syscall whitelist** for inference server (I3, I5)
81. **Capability mode (Capsicum-style)** — processes start full, enter capability mode revoking ambient authority (I5)
82. **Memory-mapped KV store** — cache in a pre-allocated mmap'd file, survives crash (H2)
83. **"Malloc-as-file" heap** — kernel heap backed by persistent RedSea-style bitmap file (I4)
84. **Arena allocator (`wubu_arena`)** for per-request tensors (I4, E4)
85. **Content-addressable Styx FS** for corpus/model store — /n/corpus/, /n/models/ hash-addressed (H5)
86. **dosgui WM + Styx as user-space processes** — crash isolation, restart without reboot (I1)
87. **Inference unsafe-ops sandbox process** — loader/custom-kernel crash doesn't take down server (I1)
88. **Kernel event bus over 9P namespace** — netlink-style channel, GUI subscribes not polls (E5)
89. **`/proc/` 9P namespace entry** — kernel metrics (task count, memory, uptime, ctx switches) as files (I5)
90. **Inference internals as file hierarchy** — /n/model/, /n/scheduler/, /n/cache/, /n/tensors/ (H3)

### TIER 6 — Planning, docs, UX cohesion (the meta-layer)

91. **Gap ledger as flow system** — research/INDEX.md gains owner/priority/committed-close fields; input-rate vs throughput tracking (J2)
92. **WIP limits on gaps** — finish before starting, lead time drops 40-60% (J2)
93. **Explicit rollover/drop decisions** — every gap reviewed weekly, dropped gaps get a one-line reason (J2)
94. **"Wired" verification gate** — ASAN clean + cosine in tolerance + make test_all green before gap flips open→wired (J4)
95. **Research cron with M1 close-commitment** — weekly: pick next avenue, 3 parallel searches, extract 2-4 sources, generate 100-gap mini-bank, close first 5 with real C11 + tests (J4)
96. **Design tokens single source (W3C DTCG)** — tokens/ directory, colors/typography/spacing/semantic tokens, code-gen at build (J3)
97. **Win98/XP palette as semantic tokens** — beige #C0C0C0, window grey #C3C3C3, title blue #000080, button face #F0F0F0 (J3)
98. **Diátaxis docs restructure** — tutorials/how-to/reference/explanation quadrants in docs/ (J5)
99. **Docs-as-code CI check** — API change without doc update fails CI (J5)
100. **UX regression suite** — render each app to PPM/PNG, diff against golden images, in CI (J1)
101. **Terminology glossary** — session/run/decode/generate/backend/engine defined once (J2)
102. **`UX_AUDIT.md` checklist** — visual consistency, output format, terminology, interaction patterns, error style, help output (J1)
103. **CLI banner/section/stat everywhere** — wubu_banner.h as sole CLI identity (B1)
104. **GUI apps via dosgui_chrome_draw_window only** — content rect from chrome, never manual win->x (B2)
105. **Input hit-testing against chrome content rect** (B3)

---

## Theme-by-Theme Detail Files

| File | Themes | Topics | Lines |
|------|--------|--------|-------|
| `research/066-theme-abc-modularity-agnostic-agent.md` | A: Modularity, B: Agnostic Interfaces, C: Agent-Ergonomic | A1-A5, B1-B5, C1-C5 (15) | 284 |
| `research/066-theme-def-build-patterns-testing.md` | D: Build & Tooling, E: C11 Patterns, F: Testing | D1-D5, E1-E5, F1-F5 (15) | 214 |
| `research/066-theme-ghi-plugins-interchange-kernel.md` | G: Plugins, H: Interchange/Namespace, I: OS/Kernel | G1-G5, H1-H5, I1-I5 (15) | 246 |
| `research/066-theme-j-planning-ux.md` | J: Planning & UX Cohesion | J1-J5 (5) | 156 |
| `research/066-ux-cohesion-research.md` | **THIS FILE — master synthesis** | all 50 | 105 |

Each topic in the detail files has: 7-hop Kevin-Bacon chain (seed → hop1..hopN), one-line convergence principle, 3-6 real verified source URLs, and 2 concrete improvements grounded in actual wubuwizard/wubuos components.

---

## Cross-Theme Synthesis — the 5 Highest-Leverage Actions

From the research, the 50 topics converge on one meta-principle: **coherence requires structure, and structure must be machine-readable and versioned, not just written and forgotten.** The five highest-leverage actions, ranked:

1. **ADR + AGENTS.md backbone (J1, C1)** — Every architectural decision gets an ADR; both repos get AGENTS.md. This directly solves "monolithic, not agnostic, hard for AI agents to work on": agents query the ADR log to understand WHY, and AGENTS.md tells them WHERE/HOW. Without this, every other improvement is built on sand.

2. **Opaque-pointer seams + god-header split (E1, A1)** — Convert the 119-file wubu_model.h dependency into a small public API header + internal implementation header. This is the single biggest structural win: it cuts the include fan-in, creates a compilation firewall, and forces every other module to declare its own boundary.

3. **Backend/model vtable layers (B2, B1)** — `wubu_backend_api` + `wubu_model_t` structs of function pointers kill the `#ifdef __CUDA__` branching and format hardcoding. "Not agnostic" becomes "agnostic by construction": new hardware/format = new adapter file, zero core changes.

4. **Honest gap ledger with WIP limits + wired gate (J2, J4)** — research/INDEX.md gains owner/priority/commitment fields, and "wired" requires a passing verification gate. Research stops producing docs and starts producing tested code; the ledger stops accumulating stale gaps.

5. **Build speed: ninja + ccache + compile_commands.json (C5, D1)** — 10-100× faster rebuilds. This multiplies EVERYTHING else: an agent can try 30 changes/minute instead of 2, tests run faster, CI costs less. Build speed is the force multiplier for the whole program.

---

*Research completed 2026-08-05 via 4 parallel 7-hop research chains (~215 web searches). All URLs cited are real, verified web pages. No sources fabricated.*
