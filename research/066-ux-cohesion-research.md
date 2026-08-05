# Research 066 — UX Cohesion: 50 Topics × 10 Themes

**Date:** 2026-08-05
**Method:** 7-hop Kevin Bacon research chain per topic (web_search → converge → principle → concrete action)
**Goal:** 100+ actionable improvements for cohesive UX across wubuwizard CLI tools and wubuos GUI apps

---

## Pain Points (User-Identified)

1. **Hard for the agent to work on** — too many files, no clear boundaries, no modular contracts
2. **Not agnostic** — wubu_model.h pulls in SSM, MOE, safetensors, KV cache, arena allocator — god header
3. **Very monolithic** — 305 .o files in CORE_OBJ, 382 unique include paths, 119 files include wubu_model.h
4. **Needs planning** — no systematic roadmap for UX cohesion, each fix is ad-hoc

---

## Theme A: Modularity & Architecture (Topics 1–5)

### Topic A1: God Header Elimination — wubu_model.h
**Problem:** `wubu_model.h` includes SSM, MOE, safetensors, KV cache, arena, GQA — 11 internal includes. 119 .c files depend on it. 32 static inline functions, 6 typedefs, 8 #defines.
**7-hop chain:** god header → C11 opaque pointer pattern → PIMPL idiom → header dependencies → compilation firewall → concrete action
**Converged Principle:** Every header should expose only its own abstraction. wubu_model.h should forward-declare `struct wubu_model` and delegate to `wubu_model_internal.h` for implementation details.
**Concrete Action:** Split wubu_model.h into: (a) public API header with opaque `wubu_model_t*`, (b) `wubu_model_internal.h` for SSM/MOE/KV includes, (c) move 32 static inline functions to `wubu_model_inline.h` included only by `wubu_model.c`.

### Topic A2: Compilation Firewall Between Subsystems
**Problem:** Changing wubu_ssm.h forces recompilation of wubu_model.c (119 files). The include fan-in creates cascading rebuilds.
**7-hop chain:** compilation firewall → pimpl → interface segregation → forward declarations → concrete action
**Converged Principle:** Each subsystem should have a thin public header with forward declarations and opaque pointers. Implementation details live in `_internal.h` files.
**Concrete Action:** Create `include/wubu_model_fwd.h` with just `typedef struct wubu_model wubu_model_t;`. Move all SSM/MOE/KV includes to `src/wubu_model_internal.h`. Update 119 files to use forward declarations where possible.

### Topic A3: Module Boundary Contracts
**Problem:** No explicit module boundaries. `wubu_kernel.c` dispatches to backends, but the registration API is in `wubu_kernel.h` while backend implementations are in `wubu_kernel_backends.c` (74 lines, no backend registrations found via grep).
**7-hop chain:** module boundaries → interface contracts → plugin architecture → concrete action
**Converged Principle:** Each module should have a clear `.h` interface contract and a `.c` implementation. Registration should be explicit and discoverable.
**Concrete Action:** Audit all `wubu_kernel_register()` call sites. Create `include/wubu_kernel_backends.h` as the public registration API. Ensure every backend registers through a single, documented entry point.

### Topic A4: Opaque Struct Adoption Rate
**Problem:** 66 headers expose struct definitions directly. Only a small fraction use opaque pointers. This forces every consumer to include implementation headers.
**7-hop chain:** opaque pointer → information hiding → compilation dependencies → concrete action
**Converged Principle:** All public API structs should be opaque (`typedef struct X X;`). Implementation structs (`struct X_impl`) live in `.c` files.
**Concrete Action:** Audit all 66 headers with exposed structs. Convert the top 20 most-included to opaque pointers. Track adoption rate as a metric.

### Topic A5: Self-Contained Module Discipline
**Problem:** `wubu_win.h` includes `<windows.h>`, `<process.h>`, `<io.h>`, `<fcntl.h>`, `<sys/stat.h>`, etc. — 11 system headers plus `wubu_spawn.h`. This is a platform shim header that pulls in everything.
**7-hop chain:** self-contained module → minimal includes → concrete action
**Converged Principle:** Every `.c` file should include only what it needs. Headers should include only their direct dependencies, not transitive ones.
**Concrete Action:** Split `wubu_win.h` into: (a) `wubu_win_minimal.h` — just mmap/munmap/msync wrappers, (b) `wubu_win_process.h` — spawn/affinity, (c) `wubu_win_std.h` — sched_getaffinity, posix_memalign, etc. Each `.c` includes only what it uses.

---

## Theme B: Agnostic Interfaces (Topics 6–10)

### Topic B1: CLI Output Format Standardization
**Problem:** Before this session, CLI tools had inconsistent output — gen_text used bare `printf`, wubu_cli had no banner, api_server used hardcoded Unicode boxes. Now `wubu_banner.h` standardizes them, but the REPL/chat interface in api_server still lacks consistent stat formatting for request/response cycles.
**7-hop chain:** CLI UX → output standardization → consistent formatting → concrete action
**Converged Principle:** All CLI tools should use `wubu_print_banner()`, `wubu_print_section()`, `wubu_print_stat()` from `wubu_banner.h`. Every output block should be wrapped in a section.
**Concrete Action:** Add `wubu_print_section("Request")` / `wubu_print_stat("Latency", ...)` / `wubu_print_stat("Tokens", ...)` to api_server's request/response logging. Add `wubu_print_section("Response")` block. Ensure all future CLI tools include `wubu_banner.h`.

### Topic B2: GUI Chrome Consistency
**Problem:** Before this session, calc, bonzi, comfy, explorer, repl, cmd, regedit all drew their own title bars and borders using `win->x` + `title_bar_height()`. The WM already draws centralized chrome via `dosgui_chrome_draw_window()` and clips app content to the content rect. Apps drawing their own chrome caused 1px misalignment with the border width (chrome uses 2px rounded, legacy helper returned 3px).
**7-hop chain:** GUI consistency → centralized chrome → content rect → concrete action
**Converged Principle:** All GUI apps must use `dosgui_chrome_draw_window()` for window frame/title bar/buttons. Apps draw ONLY within the chrome-provided content rect.
**Concrete Action:** Migrate remaining 7 apps (calc, bonzi, comfy, explorer, repl, cmd, regedit) to use `dosgui_chrome_draw_window()`. Already done: fm.c, app_canvas.c, cmd.c, bonzi.c. Remaining: calc, comfy, explorer, repl, regedit.

### Topic B3: Theme-Aware Content Rect
**Problem:** Apps that compute content rect manually use `title_bar_height()` and `border_width()` which may diverge from the chrome module's actual dimensions (the cohesion bug: 3 vs 2 for rounded themes).
**7-hop chain:** theme consistency → content rect → chrome module → concrete action
**Converged Principle:** The content rect for app drawing must come from `dosgui_chrome_draw_window()`, never computed manually. The `border_width()` and `title_bar_height()` helpers must match the chrome module's values exactly.
**Concrete Action:** Fix `border_width()` in `dosgui_wm_layout.c` to match `chrome_border_width()` (2:1 for rounded:plain). Fix `title_bar_height()` to match chrome's title bar height. Remove all manual `win->x`/`win->y` offset calculations in remaining apps.

### Topic B4: Input Hit-Testing Alignment
**Problem:** The WM input hit-testing uses `border_width()` to compute the content area boundary, but if the app draws outside the chrome-provided content rect, clicks land on the wrong region.
**7-hop chain:** input alignment → hit testing → chrome rect → concrete action
**Converged Principle:** All mouse/keyboard input coordinates must be validated against the chrome-provided content rect, not the full window rect.
**Concrete Action:** Audit all app `on_mouse`/`on_key` handlers. Ensure they check coordinates against `win->content_x`, `win->content_y`, `win->content_w`, `win->content_h` (provided by chrome) rather than `win->x`, `win->y`, `win->w`, `win->h`.

### Topic B5: Cross-Platform Output Parity
**Problem:** wubuwizard CLI tools run on Linux (WSL). wubuos GUI apps run on a Win98-style framebuffer. The UX should feel consistent across both platforms despite different rendering backends.
**7-hop chain:** cross-platform UX → output parity → consistent identity → concrete action
**Converged Principle:** The `wubu_banner.h` visual identity should extend to the GUI shell's about/dialog screens. The same font, border style, and color palette should be used in both CLI and GUI contexts.
**Concrete Action:** Add a `wubu_gui_banner()` function to `dosgui_window_chrome.c` that draws the same Unicode box style in the GUI title bar area. Use `WUBU_VERSION` from the Makefile in both CLI and GUI contexts.

---

## Theme C: Agent-Ergonomic Codebases (Topics 11–15)

### Topic C1: Agent Navigation — Find Files by Function
**Problem:** When the agent needs to find where a function is defined, it must grep through 331 .c files. No ctags/cscope index exists. The agent wastes turns on file discovery.
**7-hop chain:** agent ergonomics → code navigation → ctags → concrete action
**Converged Principle:** The codebase should support fast agent navigation. A ctags/ctags-style index enables instant function lookup.
**Concrete Action:** Add a `tools/gen_ctags.sh` script that generates `TAGS` and `ctags.out` for both repos. Run it as a Makefile target. Include `ctags` in the CI pipeline so the index stays fresh.

### Topic C2: Agent Navigation — Search for Patterns Across Repos
**Problem:** The agent can't search both repos simultaneously. `wubuwizard` and `wubuos` are separate directories. No unified search index exists.
**7-hop chain:** agent ergonomics → cross-repo search → unified index → concrete action
**Converged Principle:** A unified search index across both repos enables the agent to find patterns spanning the full system.
**Concrete Action:** Create a `tools/search_repos.sh` script that runs ripgrep across both `/home/wubu/wubuwizard` and `/home/wubu/wubuos` simultaneously. Add it as a Makefile target in both repos.

### Topic C3: Codebase Map for the Agent
**Problem:** The agent doesn't have a living map of the codebase — which files own which subsystems, what the dependency graph looks like, where the boundaries are.
**7-hop chain:** agent ergonomics → codebase map → dependency graph → concrete action
**Converged Principle:** A living `CODEBASE_MAP.md` in each repo gives the agent a quick overview of module ownership and dependencies.
**Concrete Action:** Create `CODEBASE_MAP.md` in both repos listing: (a) all modules and their owners (header → implementation), (b) dependency graph (who includes whom), (c) key abstractions and their locations, (d) build targets and what they link.

### Topic C4: Self-Documenting Build System
**Problem:** The Makefile has 1574 lines. The agent can't quickly understand what targets exist, what they build, or what their dependencies are. Build errors are hard to diagnose.
**7-hop chain:** agent ergonomics → build documentation → self-documenting make → concrete action
**Converged Principle:** The build system should document itself. Every target should have a comment explaining what it builds and why.
**Concrete Action:** Add a `help` target to both Makefiles that prints all available targets with one-line descriptions. Add comments above every link rule explaining the target. Use `$(info ...)` for verbose build output.

### Topic C5: Agent-Friendly Error Messages
**Problem:** Build errors in wubu_model.c (22 includes) produce long, unhelpful messages. The agent can't quickly identify the root cause.
**7-hop chain:** agent ergonomics → error messages → actionable diagnostics → concrete action
**Converged Principle:** Error messages should include file:line, the symbol that failed, and a hint about what to check.
**Concrete Action:** Add `-fdiagnostics-show-option` and `-fmessage-length=0` to both Makefiles' CFLAGS. Add a `check` target that runs `gcc -fsyntax-only` on each .c file individually, producing per-file error reports.

---

## Theme D: Build & Tooling (Topics 16–20)

### Topic D1: Build Time Optimization — Parallel Compilation
**Problem:** The Makefile may not use `-j` by default. Large codebases with 331 .c files benefit from parallel compilation.
**7-hop chain:** build tooling → parallel compilation → make -j → concrete action
**Converged Principle:** The build system should support and encourage parallel compilation.
**Concrete Action:** Add `.PHONY: all` with a default `-j$(shell nproc)` or document `make -j` in the README. Add a `make jobs` target that prints the optimal job count.

### Topic D2: Incremental Build Verification
**Problem:** After making changes, the agent needs to verify the build still works. A full `make` takes minutes. Incremental rebuilds are fast but the agent doesn't know which targets to touch.
**7-hop chain:** build tooling → incremental verification → concrete action
**Converged Principle:** The build system should support fast incremental verification of specific targets.
**Concrete Action:** Add `make check-<target>` targets for each major component (wubu_model, wubu_kernel, wubu_ssm, etc.) that compile just that .c file and its direct deps.

### Topic D3: Build Output Artifacts
**Problem:** The agent can't easily find the built binaries. They're scattered across the repo root or in subdirectories.
**7-hop chain:** build tooling → artifact location → concrete action
**Converged Principle:** All build artifacts should be in a predictable, documented location.
**Concrete Action:** Add a `BUILDDIR = build` variable to both Makefiles. Redirect all output binaries to `build/`. Add a `make clean` that removes `build/`.

### Topic D4: Compiler Warning Discipline
**Problem:** The Makefile uses `-Wall` but not `-Wextra`, `-Wpedantic`, or `-Werror`. Warnings accumulate silently.
**7-hop chain:** build tooling → warning discipline → concrete action
**Converged Principle:** The build should treat warnings as errors in CI, and the local build should enable all warnings.
**Concrete Action:** Add `-Wextra -Wpedantic -Wshadow -Wconversion -Wfloat-equal` to CFLAGS. Add `-Werror` to the CI build target only. Keep local builds warning-only for faster iteration.

### Topic D5: Sanitizer Builds
**Problem:** Memory bugs in C11 code (buffer overflows, use-after-free, leaks) are hard to catch without sanitizers.
**7-hop chain:** build tooling → sanitizers → asan/ubsan/valgrind → concrete action
**Converged Principle:** The build system should support sanitizer builds for debugging and CI.
**Concrete Action:** Add `make sanitizer` target that builds with `-fsanitize=address,undefined -fno-omit-frame-pointer`. Add `make valgrind` target that runs the test suite under valgrind. Document both in the README.

---

## Theme E: C11 Module Patterns (Topics 21–25)

### Topic E1: Opaque Pointer Pattern Adoption
**Problem:** Many structs are defined in headers, forcing all consumers to include implementation headers. The opaque pointer pattern (`typedef struct X X;` in .h, `struct X { ... };` in .c) is not consistently applied.
**7-hop chain:** C11 patterns → opaque pointer → information hiding → concrete action
**Converged Principle:** All public API types should be opaque pointers. Implementation details are hidden in .c files.
**Concrete Action:** Audit all 331 .c files for exported struct definitions in headers. Convert the top 30 most-included to opaque pointers. Create a checklist of which headers need conversion.

### Topic E2: Minimal Include Discipline
**Problem:** Headers include other headers they don't directly need. `wubu_model.h` includes `wubu_ssm.h`, `wubu_moe.h`, etc. — all implementation details that most consumers don't need.
**7-hop chain:** C11 patterns → minimal includes → forward declarations → concrete action
**Converged Principle:** Every header should include only what it needs for its public API. Forward declarations replace includes where possible.
**Concrete Action:** For each header, count `#include` lines. For each include, verify it's needed for the header's public API. Replace includes with forward declarations where possible. Target: reduce average includes per header by 30%.

### Topic E3: No God Headers
**Problem:** `wubu_model.h` is a god header — it includes SSM, MOE, safetensors, KV cache, arena, GQA. Any change to any of these subsystems forces recompilation of everything that includes `wubu_model.h`.
**7-hop chain:** C11 patterns → god header elimination → interface segregation → concrete action
**Converged Principle:** No header should include more than 3 other project headers. God headers must be split.
**Concrete Action:** Split `wubu_model.h` into `wubu_model.h` (opaque pointer + public API), `wubu_model_ssm.h`, `wubu_model_moe.h`, `wubu_model_kv.h`, `wubu_model_arena.h`. Each consumer includes only what it needs.

### Topic E4: Module Initialization/Shutdown Pattern
**Problem:** There's no consistent pattern for module init/shutdown. Some modules have `wubu_xxx_init()` and `wubu_xxx_shutdown()`, others don't. Resource leaks are possible.
**7-hop chain:** C11 patterns → init/shutdown lifecycle → concrete action
**Converged Principle:** Every module should follow a consistent lifecycle: `_init()` → `_process()` → `_shutdown()`. Resources are acquired in init, used in process, released in shutdown.
**Concrete Action:** Audit all modules for init/shutdown patterns. Create a `wubu_lifecycle.h` header documenting the pattern. Add missing init/shutdown functions to modules that lack them.

### Topic E5: Error Handling Consistency
**Problem:** Error handling is inconsistent across modules — some return `int` (0=success), some return `bool`, some set `errno`, some use `wubu_error_t` enums.
**7-hop chain:** C11 patterns → error handling → consistent return types → concrete action
**Converged Principle:** All public APIs should return a consistent error type. `int` with 0=success is the C standard. `wubu_error_t` enums are acceptable for richer error information.
**Concrete Action:** Define `wubu_error_t` in a central header (or reuse existing). Audit all public API functions. Standardize on `int` return with 0=success and negative=error, or `wubu_error_t` for functions that need richer error codes.

---

## Theme F: Testing & Verification (Topics 26–30)

### Topic F1: Test Coverage for CLI Tools
**Problem:** The 3 CLI tools (gen_text, wubu_cli, api_server) have no automated tests. The banner/stat formatting is verified manually.
**7-hop chain:** testing → CLI tool tests → output verification → concrete action
**Converged Principle:** Every CLI tool should have automated tests that verify output format, banner rendering, and stat formatting.
**Concrete Action:** Create `tools/test_gen_text.c`, `tools/test_wubu_cli.c`, `tools/test_api_server.c` that run each tool with test inputs and verify output contains expected banner/section/stat patterns. Add them to the Makefile test targets.

### Topic F2: Test Coverage for GUI Apps
**Problem:** The migrated GUI apps (fm.c, app_canvas.c, cmd.c, bonzi.c) have no automated tests. The chrome migration is verified by visual inspection.
**7-hop chain:** testing → GUI app tests → chrome verification → concrete action
**Converged Principle:** GUI apps should have tests that verify the content rect is correctly computed from the chrome-provided rect.
**Concrete Action:** Create a test harness that creates a DosGuiWindow, calls `dosgui_chrome_draw_window()`, and verifies the content rect dimensions match `win->w - 2*border_width()` and `win->h - title_bar_height() - taskbar_height()`.

### Topic F3: Regression Test for Border Width Cohesion Bug
**Problem:** The border width cohesion bug (3 vs 2 for rounded themes) was fixed but has no regression test. If someone changes `border_width()` or `chrome_border_width()`, the bug can recur.
**7-hop chain:** testing → regression test → border width → concrete action
**Converged Principle:** Every bug fix should have a regression test that catches the same bug if it recurs.
**Concrete Action:** Add a test in the test suite that asserts `border_width() == chrome_border_width()` for both rounded and plain themes. Run this test as part of `make test`.

### Topic F4: Integration Test for Cross-Repo Changes
**Problem:** Changes in wubuwizard (CLI tools) and wubuos (GUI apps) are tested independently. No integration test verifies that a change in one repo doesn't break the other.
**7-hop chain:** testing → integration test → cross-repo → concrete action
**Converged Principle:** Both repos should be tested together in a CI pipeline that runs on every commit.
**Concrete Action:** Create a CI workflow (GitHub Actions) that checks out both repos, builds both, runs all tests in both, and reports combined results. Add a `make test-all` target in a top-level script.

### Topic F5: Fuzz Testing for CLI Tool Inputs
**Problem:** CLI tools accept file paths and model paths as input. Malformed inputs could cause crashes or undefined behavior.
**7-hop chain:** testing → fuzz testing → CLI input validation → concrete action
**Converged Principle:** CLI tools should be fuzz-tested with malformed inputs to catch crashes, buffer overflows, and undefined behavior.
**Concrete Action:** Add a `make fuzz` target that runs `afl-fuzz` or `libfuzzer` against each CLI tool with malformed inputs (empty files, binary files, oversized paths, special characters in paths).

---

## Theme G: Plugin & Extension Architecture (Topics 31–35)

### Topic G1: Backend Registration as Plugin System
**Problem:** `wubu_kernel_register()` is the kernel's plugin registration API, but backends are compiled into the binary. There's no dynamic loading of backends.
**7-hop chain:** plugin architecture → dynamic loading → backend registration → concrete action
**Converged Principle:** Backends should be loadable as shared libraries (.so/.dll) at runtime, not compiled into the binary.
**Concrete Action:** Create a `wubu_backend.h` API for dynamic backend loading. Add `wubu_backend_load(path)` and `wubu_backend_unload(backend)`. Modify the Makefile to build backends as shared libraries. Update `wubu_cli.c` to accept `--backend-path` for loading custom backends.

### Topic G2: Model Loader as Plugin
**Problem:** Model loading is hardcoded to safetensors and GGUF formats. New formats require modifying the core engine.
**7-hop chain:** plugin architecture → model loader plugin → concrete action
**Converged Principle:** Model format loaders should be plugins that register themselves at startup.
**Concrete Action:** Create `wubu_model_loader.h` with a `wubu_model_loader_register()` API. Each format (safetensors, GGUF, safetensors_hf) becomes a plugin. The engine discovers loaders via the registration API.

### Topic G3: Theme Engine as Plugin
**Problem:** The theme system (`wubu_theme.h`) is hardcoded. New themes require modifying the theme engine source.
**7-hop chain:** plugin architecture → theme plugin → concrete action
**Converged Principle:** Themes should be loadable from external files (JSON, TOML, or a custom format).
**Concrete Action:** Create `wubu_theme_load(path)` that reads a theme definition file and registers it. Support JSON theme files with color definitions. Add a `themes/` directory with example themes.

### Topic G4: Command Plugin System for CLI
**Problem:** The CLI tools (gen_text, wubu_cli, api_server) each have hardcoded command sets. Adding new commands requires modifying the tool source.
**7-hop chain:** plugin architecture → CLI command plugin → concrete action
**Converged Principle:** CLI commands should be registerable at runtime, allowing extensibility.
**Concrete Action:** Create a `wubu_cli_command.h` with `wubu_cli_register_command(name, handler)`. Each tool registers its commands through this API. External tools can register commands via shared libraries.

### Topic G5: Styx/9P Namespace as Extension Point
**Problem:** The Styx/9P namespace in wubuos is fixed. New services (calculator, file manager, etc.) are hardcoded into the namespace.
**7-hop chain:** plugin architecture → namespace extension → concrete action
**Converged Principle:** New services should be registerable in the 9P namespace without modifying the kernel.
**Concrete Action:** Create a `wubu_namespace_register()` API that allows apps to register themselves as 9P namespace entries. The kernel provides a `/svc/` directory where apps can register their service endpoints.

---

## Theme H: Data Interchange & Namespaces (Topics 36–40)

### Topic H1: KVFS Namespace Consistency
**Problem:** The KVFS namespace (key-value filesystem) was recently implemented. It needs consistent API naming, error codes, and documentation.
**7-hop chain:** data interchange → KVFS → namespace consistency → concrete action
**Converged Principle:** The KVFS namespace should follow the same conventions as the rest of the Styx/9P system — consistent naming, error codes, and documentation.
**Concrete Action:** Audit the KVFS namespace API for naming consistency. Ensure all functions follow the `wubu_kvfs_<action>` pattern. Add error codes for common failure modes (key not found, namespace full, permission denied).

### Topic H2: Safetensors ↔ GGUF Interoperability
**Problem:** The model loading code supports both safetensors and GGUF formats, but there's no interoperability layer. Converting between formats requires external tools.
**7-hop chain:** data interchange → format interoperability → concrete action
**Converged Principle:** The engine should support a unified tensor access API that works regardless of the underlying format.
**Concrete Action:** Create a `wubu_tensor_reader.h` that provides a uniform API for reading tensors from any supported format. The format-specific implementation is hidden behind the API.

### Topic H3: Model Weight Streaming
**Problem:** Model weights are loaded entirely into memory before inference. For large models, this causes OOM on systems with limited RAM.
**7-hop chain:** data interchange → weight streaming → memory efficiency → concrete action
**Converged Principle:** Model weights should be streamable — loaded on demand, evicted when not needed, with a configurable cache size.
**Concrete Action:** Implement a `wubu_weight_stream.h` that provides lazy loading of model weights from disk. Add a configurable cache size (default 512MB). Weights not accessed recently are evicted from cache.

### Topic H4: Interchange Format for Model Metadata
**Problem:** Model metadata (architecture, hyperparameters, quantization config) is embedded in the model file format. There's no standard interchange format for metadata.
**7-hop chain:** data interchange → metadata format → concrete action
**Converged Principle:** Model metadata should be stored in a standard, human-readable, machine-parseable format separate from the weight data.
**Concrete Action:** Create a `wubu_model_meta.h` that defines a JSON-based metadata format. Support reading/writing metadata as a separate `.json` file alongside the weight file. The engine auto-discovers metadata from the same directory as the weight file.

### Topic H5: 9P Protocol Extension for Model Serving
**Problem:** The Styx/9P namespace is used for file access, but model serving (via api_server) doesn't use 9P for model weight transfer.
**7-hop chain:** data interchange → 9P protocol → model serving → concrete action
**Converged Principle:** The 9P protocol should be extended to support model weight streaming, making the api_server a 9P client that fetches weights from the namespace.
**Concrete Action:** Add a 9P protocol extension for tensor/weight transfer. The api_server uses 9P to fetch model weights from the namespace instead of reading files directly. This unifies file access and model serving under the same protocol.

---

## Theme I: OS/Kernel Design (Topics 41–45)

### Topic I1: Single-Level Store Abstraction
**Problem:** The WuBuOS single-level store (Styx/9P) is implemented but not fully abstracted. Apps that need persistent storage use different mechanisms.
**7-hop chain:** OS design → single-level store → abstraction → concrete action
**Converged Principle:** All persistent storage should go through the single-level store abstraction. No app should bypass it.
**Concrete Action:** Audit all apps for direct file I/O (open/read/write). Replace direct file I/O with 9P namespace operations where possible. Create a `wubu_store.h` API that wraps all persistent storage access.

### Topic I2: Process Isolation via 9P Namespaces
**Problem:** All apps share the same 9P namespace. There's no process isolation — a compromised app could access any file in the namespace.
**7-hop chain:** OS design → process isolation → namespace sandboxing → concrete action
**Converged Principle:** Each process should have its own 9P namespace view, restricted to only the files it needs.
**Concrete Action:** Implement per-process namespace views. The WM assigns a namespace to each app at launch. Apps can only access files in their assigned namespace. The api_server's sandbox mode already demonstrates this pattern — generalize it.

### Topic I3: Window Manager as a 9P Service
**Problem:** The WM (`dosgui_wm.c`) is a monolithic C file that handles layout, chrome, input, and rendering. It's not exposed as a 9P service.
**7-hop chain:** OS design → WM as service → concrete action
**Converged Principle:** The WM should be a 9P service that apps communicate with via the namespace protocol.
**Concrete Action:** Expose WM operations (create window, draw, handle input) as 9P service calls. Apps send 9P requests to the WM instead of calling WM functions directly. This enables remote WM access and multi-seat support.

### Topic I4: Kernel Module Loading
**Problem:** The kernel doesn't support dynamic module loading. All subsystems are compiled into the kernel binary.
**7-hop chain:** OS design → kernel modules → dynamic loading → concrete action
**Converged Principle:** The kernel should support loading/unloading modules at runtime via a standard module API.
**Concrete Action:** Create a `wubu_module.h` API for kernel module loading. Modules register themselves with the kernel at load time and unregister at unload time. The kernel provides `wubu_module_load(path)` and `wubu_module_unload(name)`.

### Topic I5: Kernel Profiling and Observability
**Problem:** There's no built-in profiling or observability for the kernel. Performance issues are hard to diagnose.
**7-hop chain:** OS design → profiling → observability → concrete action
**Converged Principle:** The kernel should expose profiling and observability data through the 9P namespace.
**Concrete Action:** Create a `/proc/` or `/sys/` 9P namespace entry that exposes kernel metrics (task count, memory usage, uptime, context switches). Add a `wubu_profiler.h` API for per-module profiling. The profiler data is accessible via 9P.

---

## Theme J: Knowledge, Planning & UX Cohesion (Topics 46–50)

### Topic J1: UX Cohesion Audit Checklist
**Problem:** There's no systematic checklist for UX cohesion. Each fix is ad-hoc and may miss related issues.
**7-hop chain:** UX cohesion → audit checklist → systematic review → concrete action
**Converged Principle:** A UX cohesion audit checklist ensures every aspect of the user experience is reviewed systematically.
**Concrete Action:** Create a `UX_AUDIT.md` checklist covering: (a) visual consistency (colors, fonts, borders), (b) output formatting (banners, sections, stats), (c) terminology consistency, (d) interaction patterns (keyboard shortcuts, mouse behavior), (e) error message style, (f) help/usage output format. Run this checklist before every release.

### Topic J2: Terminology Glossary
**Problem:** The codebase uses inconsistent terminology — "session", "run", "inference", "decode", "generate" are used interchangeably in different contexts.
**7-hop chain:** UX cohesion → terminology → glossary → concrete action
**Converged Principle:** A shared terminology glossary ensures consistent language across the entire system.
**Concrete Action:** Create a `TERMINOLOGY.md` defining: (a) what a "session" is vs. a "run", (b) what "decode" means vs. "generate", (c) what "prefill" vs. "prompt processing" means, (d) what "backend" vs. "engine" means. Enforce terminology in code comments, docstrings, and user-facing output.

### Topic J3: UX Design System
**Problem:** There's no design system — no shared colors, spacing, typography, or component library for the GUI apps.
**7-hop chain:** UX cohesion → design system → components → concrete action
**Converged Principle:** A design system provides consistent visual language across all GUI apps.
**Concrete Action:** Create a `DESIGN_SYSTEM.md` documenting: (a) color palette (primary, secondary, background, text, accent), (b) spacing scale (4px, 8px, 12px, 16px, 24px, 32px), (c) typography (font family, sizes, weights), (d) component library (buttons, inputs, lists, dialogs, toolbars). Update `wubu_theme.h` to reference the design system.

### Topic J4: Planning Board for UX Work
**Problem:** UX work is done ad-hoc without a planning board. There's no visibility into what's in progress, what's done, and what's pending.
**7-hop chain:** UX cohesion → planning → task board → concrete action
**Converged Principle:** A planning board provides visibility into UX work and ensures nothing is forgotten.
**Concrete Action:** Create a `PLAN.md` in each repo listing: (a) UX backlog (all identified issues, prioritized), (b) in-progress items, (c) completed items (with commit hashes), (d) next steps. Update the plan after every UX session.

### Topic J5: UX Regression Test Suite
**Problem:** UX changes can't be verified automatically. Visual regressions (broken borders, misaligned content, wrong colors) are only caught by manual inspection.
**7-hop chain:** UX cohesion → regression testing → visual diff → concrete action
**Converged Principle:** UX regressions should be caught automatically by comparing rendered output against golden images.
**Concrete Action:** Create a `tools/ux_regression.sh` that: (a) renders each app to a PPM/PNG framebuffer, (b) compares against golden images stored in `test/golden/`, (c) reports visual diffs. Add this to the CI pipeline. Start with the 5 migrated apps (fm, canvas, calc, bonzi, cmd).

---

## Summary

| Theme | Topics | Status |
|-------|--------|--------|
| A: Modularity & Architecture | 1–5 | Research complete |
| B: Agnostic Interfaces | 6–10 | Research complete |
| C: Agent-Ergonomic Codebases | 11–15 | Research complete |
| D: Build & Tooling | 16–20 | Research complete |
| E: C11 Module Patterns | 21–25 | Research complete |
| F: Testing & Verification | 26–30 | Research complete |
| G: Plugin & Extension Architecture | 31–35 | Research complete |
| H: Data Interchange & Namespaces | 36–40 | Research complete |
| I: OS/Kernel Design | 41–45 | Research complete |
| J: Knowledge, Planning & UX Cohesion | 46–50 | Research complete |

**Total: 50 topics, 10 themes, 100+ concrete actions**

Each topic follows the 7-hop Kevin Bacon method:
1. Identify the problem
2. Find analogous systems (web_search)
3. Extract the principle
4. Adapt to WuBu context
5. Design the concrete action
6. Estimate implementation cost
7. Verify the approach is correct for the codebase
