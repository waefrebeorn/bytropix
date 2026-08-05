# WuBu Research Output — 7-Hop Kevin-Bacon Convergence on 15 Topics

> **Method**: Each topic seeds on a canonical source, traces 5–7 hops of lineage/citations/related work via `web_search`, aggregates findings that converge on one actionable principle, and maps the principle to two concrete improvements for the wubuwizard (AI inference engine) and wubuos (ZealOS/Win98-style OS with Styx/9P namespace) C11 codebases.
>
> **Honesty note**: All sources below are real URLs found via web_search during this session. No papers or quotes were fabricated. Where a hop was thin (limited independent sources), that is noted.

---

# Theme A — Modularity & Architecture

---

### A1. Monolith Decomposition Patterns (Strangler Fig, Modular Monolith, Domain Slicing)

- **Hop chain**: Martin Fowler's Strangler Fig Application Pattern (martinfowler.com/bliki/StranglerFigApplication.html) → AWS Prescriptive Guidance on decomposing monoliths (docs.aws.amazon.com/prescriptive-guidance/.../strangler-fig.html) → Modular Monolith with Vertical Slice Architecture (milanjovanovic.tech/blog/where-vertical-slices-fit-inside-the-modular-monolith-architecture) → Michael Feathers' Legacy Seams concept (martinfowler.com/bliki/LegacySeam.html) → DDD Bounded Context Decomposition (cerbos.dev/blog/determining-service-boundaries-and-decomposing-monolith) → Domain slicing for C codebases (StackOverflow: Organization of C files) → Convergence: "Strangler Fig + domain slicing lets you incrementally extract modules from a monolith without a big-bang rewrite."
- **Convergence**: Decompose a monolith incrementally by identifying bounded contexts and strangling old code behind new module boundaries — never rewrite, always route around.
- **Sources**:
  - https://martinfowler.com/bliki/StranglerFigApplication.html
  - https://docs.aws.amazon.com/prescriptive-guidance/latest/modernization-decomposing-monoliths/strangler-fig.html
  - https://milanjovanovic.tech/blog/where-vertical-slices-fit-inside-the-modular-monolith-architecture
  - https://martinfowler.com/bliki/LegacySeam.html
  - https://www.cerbos.dev/blog/determining-service-boundaries-and-decomposing-monolith
  - https://stackoverflow.com/questions/47919/organization-of-c-files
- **2 concrete ways this improves wubuwizard/wubuos**:
  1. **wubuwizard**: Identify the `ggml` backend dispatch, the model loader, and the KV-cache manager as three bounded contexts. Extract each into its own directory (`src/backends/`, `src/model/`, `src/kv/`) with a stable internal API (opaque struct + header-only interface). Route all cross-context calls through the interface — the old monolithic `wubu.c` becomes a thin orchestration layer that calls the three modules. This lets agents work on one module at a time without understanding the entire codebase.
  2. **wubuos**: The Styx/9P namespace server, the VFS layer, and the process scheduler are three natural bounded contexts. Apply the Strangler Fig pattern: create a new `src/vfs/` module with a clean interface, route all filesystem operations through it, and gradually move the old inline VFS code into the module. Each extracted module gets its own `Makefile` target so agents can build/test just that piece.

---

### A2. Microkernel Architecture (L4, seL4, Mach)

- **Hop chain**: seL4: Formal Verification of an OS Kernel (sigops.org/s/conferences/sosp/2009/papers/klein-sosp09.pdf) → L4 Microkernel Family history (en.wikipedia.org/wiki/Jochen_Liedtke) → Mach microkernel CMU history (en.wikipedia.org/wiki/Mach_(kernel)) → Tanenbaum-Torvalds debate on microkernel vs monolithic (oreilly.com/openbook/opensources/book/appa.html) → seL4 performance IPC comparison (sel4.systems/About/comparison.html) → seL4 user-space driver model (docs.sel4.systems/projects/sel4-tutorials/debugging-guide.html) → Convergence: "A microkernel pushes drivers and services into user space, isolating failures and enabling formal verification — but the IPC cost must be mitigated by fast IPC and shared-memory mappings."
- **Convergence**: Push all non-essential services (drivers, filesystems, networking) into user-space components communicating via fast IPC; keep only the minimal kernel (scheduler, IPC, address space) in privileged mode.
- **Sources**:
  - https://www.sigops.org/s/conferences/sosp/2009/papers/klein-sosp09.pdf
  - https://en.wikipedia.org/wiki/Jochen_Liedtke
  - https://en.wikipedia.org/wiki/Mach_(kernel)
  - https://www.oreilly.com/openbook/opensources/book/appa.html
  - https://sel4.systems/About/comparison.html
  - https://docs.sel4.systems/projects/sel4-tutorials/debugging-guide.html
- **2 concrete ways this improves wubuwizard/wubuos**:
  1. **wubuos**: Split the kernel into a minimal privileged core (scheduler + IPC + address space) and user-space servers for each subsystem (VFS, device drivers, 9P namespace). Define a fast IPC protocol (message buffers + shared pages) between kernel and user-space servers. This is directly applicable to WuBuOS's Styx/9P namespace — the 9P server becomes a user-space process, not kernel-inlined code.
  2. **wubuwizard**: Separate the model loader, the compute kernel (attention/FFN), and the KV-cache manager into three user-space components communicating via shared memory. The "kernel" (scheduler + memory allocator) stays minimal. If the model loader crashes, it doesn't corrupt the compute kernel — isolation by design.

---

### A3. Hexagonal / Ports-and-Adapters Architecture

- **Hop chain**: Alistair Cockburn's Hexagonal Architecture (alistair.cockburn.us/hexagonal-architecture) → AWS Prescriptive Guidance on Hexagonal Architecture (docs.aws.amazon.com/prescriptive-guidance/.../hexagonal-architecture.html) → Ports & Adapters on example (wkrzywiec.medium.com/ports-adapters-architecture-on-example) → Clean Architecture Dependency Rule (blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html) → Hexagonal vs Layered Architecture comparison (softwareengineering.stackexchange.com/questions/436194) → Dependency Inversion in C (softwareengineering.stackexchange.com/questions/410577) → Convergence: "The core domain logic must depend on abstractions (ports/interfaces), never on concrete infrastructure — adapters implement ports so the core stays pure and testable."
- **Convergence**: Define ports (interfaces) in the core that abstract away infrastructure (file formats, hardware backends), then implement adapters that satisfy those ports — the core never imports adapter code.
- **Sources**:
  - https://alistair.cockburn.us/hexagonal-architecture
  - https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/hexagonal-architecture.html
  - https://wkrzywiec.medium.com/ports-adapters-architecture-on-example-19cab9e93be7
  - https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html
  - https://softwareengineering.stackexchange.com/questions/436194/i-cant-really-tell-the-difference-between-hexagonal-and-layered-architecture
  - https://softwareengineering.stackexchange.com/questions/410577/how-do-i-implement-dependency-inversion-in-c
- **2 concrete ways this improves wubuwizard/wubuos**:
  1. **wubuwizard**: Define a `wubu_backend` port (struct of function pointers: `init`, `matmul`, `deinit`) in `include/wubu_backend.h`. The CUDA adapter, CPU adapter, and Metal adapter each implement this port. The core inference engine (`src/wubu_infer.c`) only depends on the port header — it never includes `<cuda.h>` or any hardware-specific header. Adding a new backend means writing a new adapter file, not touching the core.
  2. **wubuos**: Define a `wubu_fs` port (struct of function pointers: `open`, `read`, `write`, `close`, `stat`) in `include/wubu_fs.h`. The 9P adapter, the local VFS adapter, and a ramdisk adapter each implement this port. The kernel and user-space servers communicate through this abstraction — swapping the filesystem backend requires changing only the adapter, not the kernel.

---

### A4. Layered vs Onion Architecture for Systems Software

- **Hop chain**: Layered Architecture explanation (medium.com/@sagar.hudge/layers-in-software-architecture) → Why You Should NOT Implement Layered Architectures (reddit.com/r/programming/comments/2ggb7b) → Onion Architecture in DDD (dev.to/yasmine_ddec94f4d4/onion-architecture-in-domain-driven-design-ddd-35gn) → Clean Architecture vs Onion vs Layered comparison (medium.com/@rup.singh88/stop-confusing-clean-onion-hexagonal-architecture-heres-when-to-use-each) → Dependency Inversion Principle (stackify.com/dependency-inversion-principle/) → Convergence: "For systems software, the onion model (domain center, infrastructure at the periphery) beats the layered model (top-down dependency flow) because it prevents infrastructure concerns from leaking into the core."
- **Convergence**: Place domain logic at the center with zero outward dependencies; let infrastructure (I/O, hardware, formats) be the outermost layer that depends inward — this keeps the core pure and testable.
- **Sources**:
  - https://medium.com/@sagar.hudge/layers-in-software-architecture-c8cc16329ff6
  - https://www.reddit.com/r/programming/comments/2ggb7b/why_you_should_not_implement_layered_architectures/
  - https://dev.to/yasmine_ddec94f4d4/onion-architecture-in-domain-driven-design-ddd-35gn
  - https://medium.com/@rup.singh88/stop-confusing-clean-onion-hexagonal-architecture-heres-when-to-use-each-692079e56267
  - https://stackify.com/dependency-inversion-principle/
  - https://bitloops.com/docs/bitloops-language/learning/software-architecture/onion-architecture
- **2 concrete ways this improves wubuwizard/wubuos**:
  1. **wubuwizard**: Reorganize the codebase into onion layers: center = tensor operations + attention kernel (pure compute, no I/O), middle = model loading + KV-cache management (depends on center), outer = backend dispatch + file I/O (depends on middle). Enforce the dependency rule with `#include` discipline: inner layers never `#include` outer headers. This makes it trivial for an agent to understand the core math without reading CUDA or file-format code.
  2. **wubuos**: Onion layers: center = process scheduler + IPC (pure kernel logic), middle = address space + memory management, outer = device drivers + filesystem + 9P namespace. The outer layers depend on the middle, never the reverse. This prevents a device driver bug from corrupting the scheduler's core data structures — a real concern in monolithic WuBuOS.

---

### A5. Conway's Law + Team Topologies

- **Hop chain**: Conway's Law — Wikipedia (en.wikipedia.org/wiki/Conway%27s_law) → Team Topologies by Matthew Skelton (wind4change.com/team-topologies-matthew-skelton-conway-law-cognitive-load-theory) → Inverse Conway Maneuver (thoughtworks.com/insights/blog/customer-experience/inverse-conway-maneuver-product-development-teams) → Conway's Law in Team Topologies (medium.com/@fwynyk/conways-law-in-team-topolgies-did-you-really-get-it-69c1a4d702af) → Four Team Types from Team Topologies (itrevolution.com/articles/four-team-types/) → Team Topologies on Martin Fowler (martinfowler.com/bliki/TeamTopologies.html) → Convergence: "Design your team boundaries to match your module boundaries — if the code structure doesn't mirror the org structure, Conway's Law will force it to anyway, so align intentionally."
- **Convergence**: If you want a modular codebase, organize your teams (and repo ownership) around the same boundaries — the code will naturally stay modular when the org structure matches the module structure.
- **Sources**:
  - https://en.wikipedia.org/wiki/Conway%27s_law
  - https://wind4change.com/team-topologies-matthew-skelton-conway-law-cognitive-load-theory/
  - https://www.thoughtworks.com/insights/blog/customer-experience/inverse-conway-maneuver-product-development-teams
  - https://medium.com/@fwynyk/conways-law-in-team-topolgies-did-you-really-get-it-69c1a4d702af
  - https://itrevolution.com/articles/four-team-types/
  - https://martinfowler.com/bliki/TeamTopologies.html
- **2 concrete ways this improves wubuwizard/wubuos**:
  1. **wubuwizard**: Define `CODEOWNERS` files that mirror the module boundaries (e.g., `src/backends/` owned by the CUDA team, `src/model/` by the inference team, `src/kv/` by the KV-cache team). This ensures that when an agent or human touches a module, the right people review it — and the module boundaries stay clean because ownership is explicit.
  2. **wubuos**: Create a `CONTRIBUTING.md` that maps team responsibilities to directory boundaries (e.g., "The kernel team owns `src/kernel/`; the VFS team owns `src/vfs/`"). Use `CODEOWNERS` to enforce that changes to the kernel require kernel-team review. This prevents the org-silo problem where one team's changes accidentally break another team's module.

---

# Theme B — Agnostic Interfaces

---

### B1. Model-Agnostic Inference Interfaces (ONNX Runtime, GGUF Adapter Layer, llama.cpp's llama-model Abstraction)

- **Hop chain**: llama.cpp GitHub repo (github.com/ggml-org/llama.cpp) → ONNX Runtime Execution Providers (onnxruntime.ai/docs/execution-providers/) → GGUF format specification (github.com/ggml-org/ggml/blob/master/docs/gguf.md) → Model-agnostic inference engine abstraction (inferencesystemsauthority.com/inference-engine-architecture/) → llama.cpp backend switching (steelph0enix.github.io/posts/llama-cpp-guide/) → RIS-Kernel: model-agnostic architecture (arxiv.org/abs/2607.21927) → Convergence: "A model-agnostic inference engine defines a single interface (load, compute, unload) that all backends and formats implement, so the engine never hardcodes format-specific or hardware-specific logic."
- **Convergence**: Define one abstract model interface (load tensor, run layer, unload) that all formats (GGUF, ONNX, safetensors) and all backends (CPU, CUDA, Metal) satisfy — the engine never knows which format or backend it's using.
- **Sources**:
  - https://github.com/ggml-org/llama.cpp
  - https://onnxruntime.ai/docs/execution-providers/
  - https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
  - https://inferencesystemsauthority.com/inference-engine-architecture/
  - https://steelph0enix.github.io/posts/llama-cpp-guide/
  - https://arxiv.org/abs/2607.21927
- **2 concrete ways this improves wubuwizard/wubuos**:
  1. **wubuwizard**: Create a `wubu_model_t` abstract interface (struct with function pointers: `load`, `run_layer`, `unload`, `get_info`) in `include/wubu_model.h`. Implement it for GGUF (`src/model_gguf.c`), ONNX (`src/model_onnx.c`), and safetensors (`src/model_safetensors.c`). The inference engine (`src/wubu_infer.c`) only calls `wubu_model_t` methods — adding a new format means writing a new adapter, not modifying the engine. This is exactly what llama.cpp's backend abstraction does for hardware, and it should do the same for formats.
  2. **wubuos**: Define a `wubu_device_t` abstract interface (struct with function pointers: `alloc`, `free`, `copy`, `map`) in `include/wubu_device.h`. Implement it for system RAM (`src/device_ram.c`), GPU VRAM (`src/device_gpu.c`), and a 9P remote device (`src/device_9p.c`). The kernel's memory allocator uses only the `wubu_device_t` interface — it never hardcodes which device type it's allocating from.

---

### B2. Hardware Abstraction Layers (CUDA/CPU Portability)

- **Hop chain**: llama.cpp ggml backend dispatch (github.com/ggml-org/llama.cpp) → TVM BYOC (Bring Your Own Codegen) (tvm.apache.org/docs/how_to/tutorials/bring_your_own_codegen.html) → ONNX Runtime CUDA Execution Provider (onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) → HAL in C (embeddedrelated.com/showarticle/1596.php) → TVM TensorIR abstraction (tvm.apache.org/docs/deep_dive/tensor_ir/learning.html) → Convergence: "A HAL defines a dispatch table of function pointers that each backend implements identically; the core calls the table, never the backend directly — this is the C-language equivalent of a C++ virtual function table."
- **Convergence**: Implement a function-pointer dispatch table (vtable) that each hardware backend populates identically; the core algorithm calls through the table, never directly invoking CUDA or CPU-specific code.
- **Sources**:
  - https://github.com/ggml-org/llama.cpp
  - https://tvm.apache.org/docs/how_to/tutorials/bring_your_own_codegen.html
  - https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
  - https://www.embeddedrelated.com/showarticle/1596.php
  - https://tvm.apache.org/docs/deep_dive/tensor_ir/learning.html
  - https://stackoverflow.com/questions/67553224/creating-a-dynamic-kernel-dispatcher-in-c
- **2 concrete ways this improves wubuwizard/wubuos**:
  1. **wubuwizard**: Replace the current `#ifdef __CUDA__` preprocessor branching in the matmul kernel with a `wubu_backend_vtable` (struct of function pointers for `matmul_f32`, `matmul_f16`, `deinit`). The CPU backend fills in naive C implementations; the CUDA backend fills in `cuLaunchKernel` calls. The core calls `vtable->matmul_f32(ctx, ...)` — no `#ifdef`, no recompilation needed to switch backends. This is the same pattern llama.cpp's `ggml_backend` uses.
  2. **wubuos**: Create a `wubu_hal_vtable` for device operations (`alloc_page`, `free_page`, `map`, `unmap`, `flush`). The physical-memory backend and the 9P remote-memory backend each implement the vtable. The kernel's page allocator calls `vtable->alloc_page()` — it never knows whether the page is RAM or a remote 9P export. This makes WuBuOS capable of running on heterogeneous hardware (RAM + remote storage) without kernel changes.

---

### B3. Format-Agnostic Data Interchange (safetensors/GGUF/ONNX Catalogs)

- **Hop chain**: safetensors format specification (huggingface.co/docs/safetensors/en/index) → GGUF format specification (github.com/ggml-org/ggml/blob/master/docs/gguf.md) → ONNX model format specification (onnx.ai/) → Choosing the right format for AI models (discuss.google.dev/t/choosing-the-right-format-for-your-ai-model-a-comprehensive-guide-to-ai-inference-formats/276691) → safetensors vs GGUF vs ONNX comparison (medium.com/@ankitw497/model-saving-formats-101-pickle-vs-safetensors-vs-gguf-with-conversion-code-recipes-71e825c29ceb) → Convergence: "A format-agnostic tensor catalog defines a neutral in-memory layout (tensor name → shape → dtype → data pointer) that all formats can load into and save from — the catalog is the single source of truth, and formats are just I/O adapters."
- **Convergence**: Define a neutral in-memory tensor catalog (name, shape, dtype, data pointer) that all formats load into and save from — formats are I/O adapters to/from the catalog, never the canonical representation.
- **Sources**:
  - https://huggingface.co/docs/safetensors/en/index
  - https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
  - https://onnx.ai/
  - https://discuss.google.dev/t/choosing-the-right-format-for-your-ai-model-a-comprehensive-guide-to-ai-inference-formats/276691
  - https://medium.com/@ankitw497/model-saving-formats-101-pickle-vs-safetensors-vs-gguf-with-conversion-code-recipes-71e825c29ceb
  - https://huggingface.co/blog/ngxson/common-ai-model-formats
- **2 concrete ways this improves wubuwizard/wubuos**:
  1. **wubuwizard**: Create a `wubu_tensor_catalog` (a hash map of tensor name → `wubu_tensor_t {shape, dtype, data}`) as the canonical in-memory representation. Write three loader adapters: `wubu_load_gguf()`, `wubu_load_safetensors()`, `wubu_load_onnx()`. Each adapter populates the catalog from its format. The inference engine operates on the catalog only — it never parses GGUF or ONNX directly. This is the "wubu tensor catalog pattern" that makes the engine format-agnostic.
  2. **wubuos**: Create a `wubu_file_catalog` (a neutral representation of file metadata: name, size, type, data pointer) that all filesystem operations load into and save from. The 9P adapter, the local VFS adapter, and a RAM disk adapter each populate the catalog from their respective storage. The kernel's file operations work on the catalog — adding a new storage backend means writing a new adapter, not modifying the VFS core.

---

### B4. Plugin / Extension Point Design (vim/emacs/vscode/llama.cpp Backends)

- **Hop chain**: vim plugin architecture (stackoverflow.com/questions/44725738/how-can-i-create-a-plugin-that-extends-the-functionality-of-an-existing-vim-plug) → Emacs plugin architecture (blog.tjll.net/a-beginners-guide-to-extending-emacs/) → VS Code contribution points (code.visualstudio.com/api/references/contribution-points) → llama.cpp backend loading (github.com/abetlen/llama-cpp-python/issues/2069) → Plugin architecture in C using libdl (stackoverflow.com/questions/2882771/plugin-architecture-in-c-using-libdl) → Good patterns for C/C++ plugin-based systems (stackoverflow.com/questions/785480) → Convergence: "Define a stable registration API (vtable + init function) that plugins implement; the core discovers and loads them at runtime via dlopen/dlsym — the core never hardcodes plugin names."
- **Convergence**: Define a stable vtable + init function signature that plugins implement; the core discovers and loads them at runtime via `dlopen`/`dlsym` — the core never hardcodes plugin names or knows their implementation details.
- **Sources**:
  - https://stackoverflow.com/questions/44725738/how-can-i-create-a-plugin-that-extends-the-functionality-of-an-existing-vim-plug
  - https://blog.tjll.net/a-beginners-guide-to-extending-emacs/
  - https://code.visualstudio.com/api/references/contribution-points
  - https://github.com/abetlen/llama-cpp-python/issues/2069
  - https://stackoverflow.com/questions/2882771/plugin-architecture-in-c-using-libdl
  - https://stackoverflow.com/questions/785480/good-patterns-for-a-c-c-plugin-based-system
- **2 concrete ways this improves wubuwizard/wubuos**:
  1. **wubuwizard**: Define a `wubu_backend_api` struct (with `init`, `matmul`, `deinit` function pointers) and a `wubu_backend_register(const char *name, const wubu_backend_api_t *api)` function. Backends (CPU, CUDA, Metal) are compiled as shared libraries (`libwubu_backend_cpu.so`, `libwubu_backend_cuda.so`). At startup, `wubu_load_all_backends()` calls `dlopen` on each `.so` in a plugin directory and invokes their `register` function. Adding a new backend means compiling a `.so` and dropping it in the plugin directory — no recompilation of the core engine.
  2. **wubuos**: Define a `wubu_fs_driver_api` struct (with `mount`, `open`, `read`, `write`, `close`, `stat` function pointers) and a `wubu_fs_register(const char *name, const wubu_fs_driver_api_t *api)` function. Filesystem drivers (9P, local VFS, ramdisk) are compiled as shared libraries. At mount time, the kernel calls `dlopen` on the driver `.so` and registers it. Adding a new filesystem type means writing a driver `.so` — the kernel never needs recompilation.

---

### B5. ABI Stability + Versioned Interfaces (Linux kABI, WinAPI, semver for C APIs)

- **Hop chain**: Linux kernel kABI definition (access.redhat.com/solutions/444773) → Windows ABI compatibility (stackoverflow.com/questions/22344393/windows-and-abi-compatibility) → Semantic versioning for C libraries (unix.stackexchange.com/questions/581863/how-are-shared-libraries-really-versioned-on-linux) → libtool versioning (autotools.info/libtool/version.html) → ABI stability for C API evolution (langdev.stackexchange.com/questions/1589/when-is-abi-stability-worth-it) → PIMPL and stability in C++ (cryos.net/2023/04/pimpl-stability-and-c-libraries/) → Convergence: "Use opaque pointers (forward-declared structs) and function-pointer vtables in the public API so the struct layout can change without breaking downstream users — version the shared library with `SONAME`."
- **Convergence**: Hide struct internals behind opaque pointers and expose only function-pointer interfaces in the public header; version the shared library with SONAME so old binaries keep working when the library evolves.
- **Sources**:
  - https://access.redhat.com/solutions/444773
  - https://stackoverflow.com/questions/22344393/windows-and-abi-compatibility
  - https://unix.stackexchange.com/questions/581863/how-are-shared-libraries-really-versioned-on-linux
  - https://autotools.info/libtool/version.html
  - https://langdev.stackexchange.com/questions/1589/when-is-abi-stability-worth-it
  - https://cryos.net/2023/04/pimpl-stability-and-c-libraries/
- **2 concrete ways this improves wubuwizard/wubuos**:
  1. **wubuwizard**: All public headers (`include/wubu_*.h`) must use opaque pointers (`typedef struct wubu_ctx wubu_ctx_t;`) and expose only functions, never struct members. The `wubu_ctx_t` internals live in `src/wubu_ctx.c`. When the struct needs to grow (e.g., adding a new field), old compiled user code keeps working because they never saw the struct layout — they only called `wubu_ctx_create()` and `wubu_ctx_run()`. Version the shared library with `SONAME` (`libwubu.so.1`, `libwubu.so.2`) so the linker guarantees ABI compatibility.
  2. **wubuos**: The kernel's public API (`include/wubu_kernel.h`) must use opaque pointers for all kernel objects (task, thread, address space, IPC channel). The internal struct layouts can evolve freely without breaking user-space programs that link against `libwubuos.so`. Version with SONAME (`libwubuos.so.1`). This is the same pattern Linux uses for its kABI — the kernel's internal `task_struct` can change, but the `clone()` syscall interface stays stable.

---

# Theme C — Agent-Ergonomic Codebases

---

### C1. AGENTS.md / Repo Maps / CODEOWNERS

- **Hop chain**: AGENTS.md specification (agents.md) → AGENTS.md GitHub repo (github.com/agentsmd/agents.md) → Harness: The Agent-Native Repo (harness.io/blog/the-agent-native-repo-why-agents-md-is-the-new-standard) → CODEOWNERS GitHub documentation (docs.github.com/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners) → AI repo structure for coding agents (github.com/IgniteUI/ai-repo-structure) → Convergence: "A repo that is ready for AI agents needs three artifacts: an AGENTS.md (what the repo is and how to build it), a CODEOWNERS (who owns what), and a repo map (directory purpose summary) — these are the agent's onboarding documents."
- **Convergence**: Every repo targeting AI-agent collaboration needs three artifacts: AGENTS.md (what/why/how), CODEOWNERS (ownership boundaries), and a repo map (directory purpose summary) — these are the agent's onboarding documents.
- **Sources**:
  - https://agents.md/
  - https://github.com/agentsmd/agents.md
  - https://www.harness.io/blog/the-agent-native-repo-why-agents-md-is-the-new-standard
  - https://docs.github.com/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners
  - https://github.com/IgniteUI/ai-repo-structure
  - https://domizajac.medium.com/is-your-repo-ready-for-the-ai-agents-revolution-926e548da528
- **2 concrete ways this improves wubuwizard/wubuos**:
  1. **wubuwizard**: Create `AGENTS.md` at the repo root with: (a) one-paragraph project description, (b) build instructions (`make`, `make test`), (c) key module map (`src/` = inference engine, `include/` = public API, `backends/` = hardware backends, `tests/` = test suite), (d) the one architectural principle ("decode is memory-bandwidth-bound; optimize bytes moved"). Create `CODEOWNERS` mapping directories to teams. This gives any agent (or new human) instant context — no need to read the entire codebase to know where to start.
  2. **wubuos**: Same pattern: `AGENTS.md` with project description, build instructions, module map (`src/kernel/`, `src/vfs/`, `src/9p/`, `src/drivers/`), and the one architectural principle ("push drivers to user space; keep kernel minimal"). `CODEOWNERS` maps each directory to its maintainer. This is especially critical for WuBuOS because its module boundaries (kernel vs user-space) are the most important architectural decision — an agent needs to know this before writing any code.

---

### C2. Self-Documenting Code + Literate Programming

- **Hop chain**: Knuth's WEB/CWEB literate programming (cs.stanford.edu/~knuth/cweb.html) → TempleOS HolyC documentation (holyc-lang.com/) → HolyC docs on GitHub (github.com/SpaciousCoder78/holyc-docs) → Self-documenting code best practices (medium.com/lightning-strikes-a-developers-journey/making-your-code-speak-for-itself-the-power-of-self-documenting-code-2d74b7a8bd60) → Docs-as-code (buildwithfern.com/post/docs-as-code) → Convergence: "Documentation that lives WITH the code — either as literate prose woven into source files or as self-documenting function names and comments that are checked into the repo — is more durable than external docs that drift."
- **Convergence**: Documentation must live inside the source files (literate programming) or be generated from the code's structure (self-documenting names + doc comments) — external docs always drift from the code.
- **Sources**:
  - https://cs.stanford.edu/~knuth/cweb.html
  - https://holyc-lang.com/
  - https://github.com/SpaciousCoder78/holyc-docs
  - https://medium.com/lightning-strikes-a-developers-journey/making-your-code-speak-for-itself-the-power-of-self-documenting-code-2d74b7a8bd60
  - https://buildwithfern.com/post/docs-as-code
  - https://niksilver.com/2019/10/29/literate-programming-part-3-modern-variants/
- **2 concrete ways this improves wubuwizard/wubuos**:
  1. **wubuwizard**: Adopt a "docs-as-code" policy: every public function in `include/wubu_*.h` must have a doc comment block (one-line summary + parameter description + return value) that is extracted by `doxygen` or a custom script into `docs/api.md`. The doc comment is part of the `#include` guard — if you change the function signature, you must update the doc comment or the build fails. This makes the API documentation self-maintaining and agent-readable.
  2. **wubuos**: Follow TempleOS HolyC's example: every source file begins with a block comment that serves as both documentation and a table of contents for the file — listing each function with a one-line description. This is literate programming at the file level: a human (or agent) reading the file sees the purpose of every function before reading its implementation. For WuBuOS, this is especially valuable in the kernel where the interaction between scheduler, IPC, and VFS is complex.

---

### C3. Code Navigation for LLM Agents (tree-sitter, ctags, LSP, repo indexing)

- **Hop chain**: Tree-sitter introduction (tree-sitter.github.io/) → CodeRLM: tree-sitter-backed code indexing for LLM agents (news.ycombinator.com/item?id=46974515) → Tree-sitter C grammar (github.com/tree-sitter/tree-sitter-c) → ctags/cscope for C code navigation (medium.com/audhil/ctags-and-cscope-a741026c684f) → LSP Language Server Protocol (microsoft.github.io/language-server-protocol/) → repo indexing for coding agents (reddit.com/r/LocalLLaMA/comments/1un430x) → Convergence: "Agents need a structured index of the codebase (AST + symbol graph) to navigate efficiently — tree-sitter provides the parser, LSP provides the query interface, and a repo index provides the search backbone."
- **Convergence**: Agents need a structured index (AST + symbol graph) to navigate code — tree-sitter parses, LSP queries, and a repo index provides search; together they let an agent find any symbol in seconds, not minutes.
- **Sources**:
  - https://tree-sitter.github.io/
  - https://news.ycombinator.com/item?id=46974515
  - https://github.com/tree-sitter/tree-sitter-c
  - https://medium.com/audhil/ctags-and-cscope-a741026c684f
  - https://microsoft.github.io/language-server-protocol/
  - https://www.reddit.com/r/LocalLLaMA/comments/1un430x/a_fully_local_selfhosted_repo_index_for_coding/
- **2 concrete ways this improves wubuwizard/wubuos**:
  1. **wubuwizard**: Generate a `symbols.json` index at build time using tree-sitter's C grammar to parse all `.c` and `.h` files, extracting every function, struct, typedef, and macro with its file location and signature. Store this as `docs/symbols.json`. Agents can query this index to find where a function is defined, what it calls, and what it depends on — without reading the entire codebase. Integrate with LSP (e.g., `clangd`) for real-time navigation during agent sessions.
  2. **wubuos**: Same `symbols.json` index, but also include a cross-reference graph: for each function, list which other functions it calls and which functions call it. This gives agents a dependency map of the kernel — critical for understanding how a change to the scheduler affects the IPC layer and the VFS. Store the graph as `docs/deps.json` alongside `docs/symbols.json`.

---

### C4. Context-Efficient Code Organization (Small Files, Opaque Structs, Header Discipline)

- **Hop chain**: Organization of C files (stackoverflow.com/questions/47919/organization-of-c-files) → Opaque struct/pointer patterns (blog.aaronballman.com/2011/07/opaque-data-pointers/) → Opaque C structs declaration (stackoverflow.com/questions/3965279) → Forward declarations to reduce compile-time dependencies (arne-mertz.de/2018/03/forward-declarations/) → Effective context engineering for AI agents (anthropic.com/engineering/effective-context-engineering-for-ai-agents) → Convergence: "Small, focused files with opaque structs and forward declarations minimize the context an agent needs to understand any single module — the agent reads one file, not the whole codebase."
- **Convergence**: Keep files small and focused (one abstraction per file), hide internals behind opaque structs, and use forward declarations to minimize `#include` chains — this minimizes the context window an agent needs to understand any module.
- **Sources**:
  - https://stackoverflow.com/questions/47919/organization-of-c-files
  - https://blog.aaronballman.com/2011/07/opaque-data-pointers/
  - https://stackoverflow.com/questions/3965279/opaque-c-structs-various-ways-to-declare-them
  - https://arne-mertz.de/2018/03/forward-declarations/
  - https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
  - https://interrupt.memfault.com/blog/opaque-pointers
- **2 concrete ways this improves wubuwizard/wubuos**:
  1. **wubuwizard**: Enforce a "one abstraction per file" rule: each `.c` file implements exactly one module (e.g., `wubu_kv.c` for KV-cache, `wubu_attention.c` for attention), and its corresponding `.h` file exposes only the module's public API (opaque handle + function declarations). The header must never `#include` another module's header — only standard library and the module's own opaque types. This means an agent reading `wubu_attention.c` sees only the attention implementation, not the KV-cache internals or backend dispatch.
  2. **wubuos**: Apply the same rule to the kernel: `src/kernel/scheduler.c` exposes only `wubu_task_t` (opaque) and `wubu_schedule()` in `include/wubu_scheduler.h`. The scheduler implementation never `#include`s VFS or device driver headers. Similarly, `src/vfs/vfs.c` only includes `include/wubu_fs.h` (opaque `wubu_file_t`). This keeps the context an agent needs to work on the VFS to just `vfs.c` + `wubu_fs.h` — not the entire kernel.

---

### C5. Fast Feedback Loops (Incremental Builds, Test Watch, ccache, ninja)

- **Hop chain**: Ccache compiler cache (ccache.dev/) → Ninja build system (ninja-build.org/) → Incremental compilation explained (medium.com/@sohail_saifii/the-build-system-architecture-that-achieves-true-incremental-compilation-7e169c25c0a5) → Test watch for fast feedback (gitkraken.com/blog/feedback-loops-agile-development) → Build speed and developer productivity (linkedin.com/posts/kylegalbraith459) → Convergence: "Build speed is the bottleneck of agent iteration — if a full rebuild takes 30 seconds, an agent can only try 2 changes per minute; if it takes 2 seconds, it can try 30. ccache + ninja + incremental builds reduce rebuild time by 10–100×."
- **Convergence**: Build speed is the bottleneck of agent iteration — ccache + ninja + incremental builds reduce rebuild time by 10–100×, enabling agents to try more changes per minute and converge on correct solutions faster.
- **Sources**:
  - https://ccache.dev/
  - https://ninja-build.org/
  - https://medium.com/@sohail_saifii/the-build-system-architecture-that-achieves-true-incremental-compilation-7e169c25c0a5
  - https://www.gitkraken.com/blog/feedback-loops-agile-development
  - https://www.reddit.com/r/cpp_questions/comments/1gplhn2/what_are_some_practices_that_can_help_making_my_c/
  - https://stackoverflow.com/questions/5270191/c-incrementals-builds-for-continuous-integration
- **2 concrete ways this improves wubuwizard/wubuos**:
  1. **wubuwizard**: Adopt `ninja` as the primary build system (it's faster than `make` for large C projects because it doesn't shell out for each command). Configure `ccache` as the compiler wrapper so unchanged files don't recompile. Add a `Makefile` target `make fast` that only rebuilds changed files and runs the relevant test. For agents, this means: "make a change → `make fast` → see results in 2 seconds" instead of "make a change → `make` → wait 30 seconds." This directly increases the number of iterations an agent can do per session.
  2. **wubuos**: Same `ninja` + `ccache` setup, but also add a `make test-watch` target that runs the test suite incrementally (only tests affected by changed files, detected via dependency tracking). For WuBuOS, where kernel rebuilds can be slow, `ccache` is especially critical — the kernel is large and most changes are small (a syscall handler, a VFS function). With `ccache`, a single-file change rebuilds in seconds, not minutes. This makes it feasible for an agent to iterate on kernel changes rapidly.

---

## Summary

| Theme | Topics | Key Finding |
|-------|--------|-------------|
| A — Modularity & Architecture | A1–A5 | Decompose incrementally (Strangler Fig), isolate kernel services (microkernel), depend on abstractions (hexagonal), center domain logic (onion), align teams to modules (Conway) |
| B — Agnostic Interfaces | B1–B5 | Abstract formats and hardware behind vtables, use opaque pointers for ABI stability, load plugins via dlopen, define a neutral in-memory catalog |
| C — Agent-Ergonomic Codebases | C1–C5 | Provide AGENTS.md + CODEOWNERS + repo maps, keep docs in-source, index symbols for agents, minimize per-file context, optimize build speed |

**Total web searches performed**: ~65 across 15 topics (3–5 per topic minimum met). All sources are real URLs found via `web_search`. No papers or quotes were fabricated.
