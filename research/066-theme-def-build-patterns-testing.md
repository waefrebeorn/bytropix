# Theme D — Build & Tooling

### D1. Incremental build systems (Ninja vs Make vs CMake — build graph, dependency tracking, header deps) for huge C projects
- **Hop chain**: seed: Ninja design chapter (Evan Martin, aosabook POSA) → hop1: Ninja vs Make benchmark (Röthlisberger) → hop2: "The Success and Failure of Ninja" (Martin 2020 retrospective) → hop3: CMake Ninja generator + Ninja dependency-scanning issues (CMake discourse / Kitware issue #25912 / LLVM discourse) → hop4: gcc `-MMD -MP` automatic header dependency generation (StackOverflow, Late Developer) → hop5: ccache for large C/C++ builds (Bits'n'Bites) → hop6: "Build Systems à la Carte" (Mokhov, Mitchell, Peyton Jones) → hop7: ninja-rs educational re-implementation of the same graph semantics
- **Convergence**: A generator (CMake or a make-front) emits an explicit build graph (targets, edges, header deps), a fast graph engine (Ninja) walks it in parallel, and the compiler itself reports its real deps (`-MMD -MP`) — humans never hand-maintain a 1000-file DAG.
- **Sources**:
  - https://aosabook.org/en/posa/ninja.html
  - https://neugierig.org/software/blog/2020/05/ninja.html
  - https://david.rothlis.net/ninja-benchmark/
  - https://www.microsoft.com/en-us/research/wp-content/uploads/2018/03/build-systems.pdf
  - https://make.mad-scientist.net/papers/advanced-auto-dependency-generation/
  - https://www.bitsnbites.eu/faster-c-builds/
- **Improvements for wubuwizard/wubuos**:
  - wubuwizard: add a `make compile_commands` target (bear) plus a CMake/Ninja generator profile for the 331-file `src/` tree, so incremental rebuilds and agent tooling (clangd, static analysis) all see one dependency graph instead of re-deriving includes by hand.
  - wubuos: keep the Makefile front but add a Ninja-generated variant + ccache for kernel builds, and a forced "touch one header → assert exactly the right object set rebuilds" regression test that proves the existing header-dependency tracking actually fires.

### D2. Unity builds vs modular builds — when each wins, how to support both
- **Hop chain**: seed: Unity build (Wikipedia) → hop1: "A guide to unity builds" (onqtam: up to 90%+ faster, LTO-like cross-TU visibility) → hop2: CMake `UNITY_BUILD` property (3.16; BATCH/GROUP modes, batch size, C/CXX/CUDA) → hop3: Qt "Precompiled Headers and Unity (Jumbo) Builds" (Qt blog) → hop4: SQLite amalgamation (sqlite.org: one big file, chosen deliberately) → hop5: duplicate-symbol/ODR failures when translation units collide (StackOverflow) → hop6: chunked unity builds preserve incremental rebuilds (r/cpp comparison thread) → hop7: C++ Weekly ep. 413 (unity builds in practice)
- **Convergence**: Keep modular per-file objects as the default for incremental development and put unity/jumbo "buckets" (CMake BATCH mode, batch-size-capped `#include` wrappers) behind a single build flag used for clean/CI builds — both, supported from one build description.
- **Sources**:
  - https://cmake.org/cmake/help/latest/prop_tgt/UNITY_BUILD.html
  - https://onqtam.github.io/programming/2018-07-07-unity-builds/
  - https://en.wikipedia.org/wiki/Unity_build
  - https://sqlite.org/amalgamation.html
  - https://www.qt.io/blog/2019/08/01/precompiled-headers-and-unity-jumbo-builds-in-upcoming-cmake
- **Improvements for wubuwizard/wubuos**:
  - wubuwizard: add a `make unity` / `-DUNITY_BUILD=ON` profile that buckets the 331 sources (BATCH mode), fixing any cross-file `static` name collisions once; CI clean builds use unity, dev builds stay modular.
  - wubuos: batch the monolithic kernel C files into a few unity TUs plus a PCH for the heavy shared includes, cutting full-build wall time while keeping a modular debug target for bisecting.

### D3. Code generation / self-hosting (TempleOS MAlloc-a-file, JIT, meta-programming) — generating code from tables
- **Hop chain**: seed: TempleOS (Wikipedia: ring-0, integrated compiler/OS) → hop1: "A Constructive Look At TempleOS" (codersnotes: one integrated system, compiler lives in the OS) → hop2: HolyC (holyc-lang.com; Jamesbarford/holyc-lang: compiles to executable memory and runs it — TempleOS's compile-and-run loop) → hop3: x86 JIT from scratch (spencertipping/jit-tutorial) → hop4: LLVM TableGen (llvm.org: tables → generated include files) → hop5: X-macros for enums/tables as single source of truth (Daniel's Blog; r/embedded) → hop6: metacompilers / Forth meta-compilation (esolangs; ultratechnology) → hop7: self-hosting compilers (Wikipedia; Robert Heaton)
- **Convergence**: Make one data table the single source of truth and generate headers, switch bodies, docs, and test vectors from it as a build step — then (TempleOS-style) let the compiler execute generated code directly so the generator and the running system can never drift.
- **Sources**:
  - http://www.codersnotes.com/notes/a-constructive-look-at-templeos/
  - https://holyc-lang.com/
  - https://github.com/Jamesbarford/holyc-lang
  - https://llvm.org/docs/TableGen/
  - https://danilafe.com/blog/chapel_x_macros/
  - https://en.wikipedia.org/wiki/Self-hosting_(compilers)
- **Improvements for wubuwizard/wubuos**:
  - wubuwizard: convert the quant-grid tables (e.g. the `iq2xxs_grid` dequant tables) and the kernel dispatch table to X-macro single-source, generating the dequant switch, extern declarations, and the `WUBU_KERNEL_SCHEMA` guard header from the one table, with a `make regenerate` + git-diff drift check.
  - wubuos: make the VSL/INT syscall tables (`[FN] = handler,` lines) the single source that generates the header, stubs, and test vectors in the build (the existing `gen_ancient_corpus.py` already parses them), and give the HolyC compiler a TempleOS-style compile-to-memory-and-run JIT path.

### D4. Dependency management without package managers (vendoring, submodules, hermetic builds)
- **Hop chain**: seed: git submodule vs subtree (adam-p: subtree = forget external code, submodule = track updates) → hop1: git-vendor (thejoshwolfe: vendoring intent, not submodule semantics) → hop2: vendor-branch workflow for a C codebase (Röthlisberger: vendoring curl into `third_party/`) → hop3: hermetic builds (Kusari; Lawless: isolated/hermetic/reproducible) → hop4: Bazel hermeticity docs + Bazel vendor mode → hop5: CMake FetchContent (coderefinery; CMake discourse best practices) → hop6: vcpkg (vcpkg.io, microsoft/vcpkg) → hop7: hermetic-vs-deterministic discussion (HN)
- **Convergence**: Vendor pinned, minimal third-party trees into the repo itself (subtree or hash-pinned FetchContent), so builds are hermetic, deterministic, and fully readable by AI agents without network access.
- **Sources**:
  - https://adam-p.ca/blog/2022/02/git-submodule-subtree/
  - https://david.rothlis.net/vendor-branch/
  - https://bazel.build/external/vendor
  - https://bazel.build/basics/hermeticity
  - https://coderefinery.github.io/cmake-workshop/fetch-content/
  - https://vcpkg.io/
- **Improvements for wubuwizard/wubuos**:
  - wubuwizard: create a `third_party/` vendored tree (subtree) for the few non-self dependencies with hash-pinned FetchContent for anything fetched at configure time, plus a `make vendor-check` that fails any build step touching the network — hermetic agent builds.
  - wubuos: replace submodules with subtree/vendor-branch for external tools (Proton subsystems, bundled utilities) so a fresh clone is fully self-contained, and pin the toolchain (compiler + binutils versions) in the build for deterministic rebuilds.

### D5. Makefile patterns for 1000+ file C projects (pattern rules, automatic header deps, parallel safety)
- **Hop chain**: seed: GNU make features (MIT manual: automatic variables, pattern rules) → hop1: "Recursive Make Considered Harmful" (Miller 1997 AUUG / ACCU Overload 2006) → hop2: "Advanced Auto-Dependency Generation" (Mad-Scientist/Tromey method: `-MMD -MP` + `include` + re-exec) → hop3: non-recursive make layouts (mischasan; StackOverflow experience) → hop4: "Non-Recursive Make Considered Harmful" (Hadrian, Mokhov/Microsoft Research: make hits scale limits) → hop5: bear → `compile_commands.json` (rizsotto/Bear; clang JSON Compilation Database spec) → hop6: Ninja as the graph engine under a make-compatible front (bridges D1) → hop7: generic makefiles with GCC (`-M` family; repnz/auto-makefile)
- **Convergence**: One non-recursive make graph with pattern rules, per-object `.d` files (`-MMD -MP`) included-and-remade, order-only prerequisites for `-j` parallel safety, and a bear-generated `compile_commands.json` so editors and agents see exactly the same build.
- **Sources**:
  - https://accu.org/journals/overload/14/71/miller_2004/
  - https://make.mad-scientist.net/papers/advanced-auto-dependency-generation/
  - https://mischasan.wordpress.com/2013/03/30/non-recursive-make-gmake-part-1-the-basic-gnu-makefile-layouts/
  - https://www.microsoft.com/en-us/research/wp-content/uploads/2016/03/hadrian.pdf
  - https://github.com/rizsotto/bear
  - https://clang.llvm.org/docs/JSONCompilationDatabase.html
- **Improvements for wubuwizard/wubuos**:
  - wubuwizard: collapse to a single non-recursive Makefile with one pattern rule over `src/*.c` → `build/*.o`, per-object `.d` include-and-remake, and order-only dir deps; verify with `make -j$(nproc)` run twice (second run must be a no-op).
  - wubuos: apply the same pattern to the kernel build and add `bear -- make` to emit `compile_commands.json`, plus a header-touch test asserting only the correct object set rebuilds — parallel-safe and agent-navigable.

# Theme E — C11 Module Patterns

### E1. Opaque struct pattern in C (the wubu style) — when it helps vs hurts
- **Hop chain**: seed: opaque pointer (Wikipedia) → hop1: Memfault "Opaque Pointers and Objects in C" (handle-not-pointer, lifecycle, when to use) → hop2: StackOverflow opaque-struct declaration styles → hop3: "C Interfaces and Implementations" (Hanson — the canonical C interface book, lcc author) → hop4: PIMPL / compiler-firewall idiom (GotW "The Joy of Pimpls") → hop5: OpenSSL EVP opaque-struct migration (getters, SO) → hop6: encapsulation & information hiding in C (embeddedartistry) → hop7: stack-allocatable opaque types / heap-vs-static debate (SO, r/C_Programming)
- **Convergence**: Opaque structs buy ABI stability and a compile firewall only at API boundaries — use them where a struct crosses a module seam (or the agent/AI boundary), always pair them with create/destroy/size functions, and never force heap allocation for hot inner loops.
- **Sources**:
  - https://interrupt.memfault.com/blog/opaque-pointers
  - https://en.wikipedia.org/wiki/Opaque_pointer
  - http://www.gotw.ca/publications/mill05.htm
  - https://embeddedartistry.com/fieldatlas/encapsulation-and-information-hiding-in-c/
  - http://www.r-5.org/files/books/computers/languages/c/mod/David_R_Hanson-C_Interfaces_and_Implementations-EN.pdf
  - https://eli.thegreenplace.net/2009/10/07/book-review-c-interfaces-and-implementations-by-david-r-hanson
- **Improvements for wubuwizard/wubuos**:
  - wubuwizard: audit the 329 headers — at cross-module seams (kernel dispatch, model adapter, loader) expose only opaque handles + accessors and move private fields into the owning `.c`; add a compile-firewall test (changing one private field must recompile exactly one TU).
  - wubuos: give the NT-bridge/VSL personality modules opaque handles at their boundary to the kernel core so each personality's internal structs stay private and can evolve (or be split out of the monolith) without touching the others.

### E2. Registry/dispatch tables (function pointer tables, plugin registration) — how the wubu_kernel dispatch table should generalize
- **Hop chain**: seed: Linux syscall table (filippo.io; Chromium OS docs) → hop1: Linux Inside — syscall table initialization from macros → hop2: Linux VFS `file_operations` (kernel docs; kernel-internals.org: dispatch through `f_op` vtable) → hop3: Linux driver model registration (docs.kernel.org driver-api) → hop4: function dispatch tables in C (StackOverflow; Barr jump tables; HN design critique) → hop5: ELF `.init_array` constructor registration (maskray; StackOverflow) → hop6: COM vtable/IUnknown layout (Old New Thing; timdbg) → hop7: QEMU QOM `type_init`/`TypeInfo` registration with realize/unrealize
- **Convergence**: Make every dispatch table a data-driven registry — static const entries plus a `type_init`-style registration macro that appends to a linker-collected or init-time list — and give it a self-test that walks the table asserting no NULL slots, unique names, and doc-comment sync.
- **Sources**:
  - https://filippo.io/linux-syscall-table/
  - https://docs.kernel.org/filesystems/vfs.html
  - https://0xax.gitbooks.io/linux-insides/content/SysCall/linux-syscall-2.html
  - https://maskray.me/blog/2021-11-07-init-ctors-init-array
  - https://qemu-project.gitlab.io/qemu/devel/qom.html
  - https://devblogs.microsoft.com/oldnewthing/20040205-00/?p=40733
- **Improvements for wubuwizard/wubuos**:
  - wubuwizard: generalize the wubu_kernel dispatch table into a macro-registered registry (`WUBU_KERNEL_ENTRY(name, fn)`), generating the extern-decl header from the table and adding a table-walk test to `make test_kernel_dispatch` (NULL-slot, duplicate-name, schema cross-check).
  - wubuos: switch VSL/INT personality handlers and devices to self-registration via `.init_array`-style lists next to their implementations, QOM-style with parent/type info, so adding a personality never requires editing one giant central table.

### E3. Error handling in C (error codes, errno, result types, exceptions-as-values) — best modern practice
- **Hop chain**: seed: errno thread-safety (StackOverflow) → hop1: errno and return codes in C (Obregón) → hop2: Linux `ERR_PTR`/`IS_ERR` typed error pointers (torvalds/linux err.h; WUSTL course notes; staticthinking) → hop3: valid `goto` cleanup pattern (StackOverflow; itnext "goto hell") → hop4: Rust `Result<T,E>` (The Rust Book) → hop5: Zig error unions (zig.guide; Zig language reference) → hop6: SEI CERT C coding standard structure (CMU SEI) → hop7: "From error-handling to structured concurrency" (Nelhage)
- **Convergence**: Return errors as values — an int/enum status or a tagged result, never reliance on `errno` — with one `goto`-cleanup exit path per function and explicit propagation, which is thread-safe, greppable, and trivially machine-checkable by agents.
- **Sources**:
  - https://stackoverflow.com/questions/1694164/is-errno-thread-safe
  - https://github.com/torvalds/linux/blob/master/include/linux/err.h
  - https://staticthinking.wordpress.com/2022/08/01/mixing-error-pointers-and-null/
  - https://doc.rust-lang.org/book/ch09-00-error-handling.html
  - https://zig.guide/language-basics/errors/
  - https://blog.nelhage.com/post/concurrent-error-handling/
- **Improvements for wubuwizard/wubuos**:
  - wubuwizard: standardize a `wubu_status` enum + single-exit cleanup convention across the engine; add a lint/test asserting no control flow depends on `errno` and that every allocation/load path checks and propagates status.
  - wubuos: kernel handlers adopt `ERR_PTR`/`IS_ERR`-style typed error pointers and one central named error-code registry (grep one table for "what does -EINVAL mean here") instead of scattered raw errno guesses.

### E4. Memory ownership in C (arenas, region allocators, RAII-in-C, Rust ownership lessons for C)
- **Hop chain**: seed: region-based memory management (Wikipedia) → hop1: Tofte & Talpin region-inference paper (1997; UCLA copy) → hop2: Cyclone regions (Grossman et al., static region typing) → hop3: bump allocators and header-only arena libraries in C (gooderfreed/arena_c; Rax-x; r/C_Programming "arenas in C") → hop4: region/bump usage guidance (polished_allocators docs; r/rust_gamedev ELI5) → hop5: Rust ownership (woodruff.dev; OpenTitan "Rust for Embedded C Programmers") → hop6: memory-management series part 4 — Rust vs C/C++ lifetimes (educatedguesswork) → hop7: RAII and single-ownership (Wikipedia; verdagon "Next Steps for Single Ownership and RAII")
- **Convergence**: Encode ownership in code structure instead of runtime discipline: per-request arenas/bump allocators with explicit reset points replace most malloc/free pairs, and Rust-style single-owner + borrow-not-copy conventions applied to arena-backed handles eliminate the double-free/use-after-free classes.
- **Sources**:
  - https://en.wikipedia.org/wiki/Region-based_memory_management
  - https://www.cs.umd.edu/projects/cyclone/papers/cyclone-regions.pdf
  - https://www.cs.ucla.edu/~palsberg/tba/papers/tofte-talpin-iandc97.pdf
  - https://github.com/gooderfreed/arena_c
  - https://educatedguesswork.org/posts/memory-management-4/
  - https://opentitan.org/book/doc/rust_for_c_devs.html
- **Improvements for wubuwizard/wubuos**:
  - wubuwizard: introduce a per-generation decode arena (bump allocator reset at the end of each generate call) replacing per-call malloc/free in the KV-cache/SSM-workspace paths — matches the existing bandwidth-bound workspace pattern — and add owner-tag fields to handle structs in debug builds.
  - wubuos: give the kernel a small bump-allocator frame for transient syscall buffers, plus an allocation-tracking layer in debug builds that catches leaks and free-mismatches inside the 8086/VSL emulator.

### E5. Component communication (event bus, message passing, callbacks vs polling) for loosely-coupled modules
- **Hop chain**: seed: observer vs publish-subscribe (embeddedartistry) → hop1: Linux uevent/netlink kernel→userspace notification (StackOverflow; sid-project) → hop2: message passing vs shared memory IPC (StackOverflow; GeeksforGeeks) → hop3: Erlang actor model / message passing (dist-prog-book) → hop4: QEMU QOM tree + signals (QOM docs) → hop5: event-bus design for decoupled modules (gamedev StackExchange) → hop6: callback vs polling decoupling (bitsquid "Managing Decoupling Part 2"; StackOverflow event-driven) → hop7: the event loop (Node.js docs)
- **Convergence**: Modules communicate through a small typed message/event bus with decoupled queues drained by a central loop (poll), reserving callbacks for latency-critical paths — never direct cross-module function calls at the seams.
- **Sources**:
  - https://embeddedartistry.com/fieldatlas/differentiating-observer-and-publish-subscribe-patterns/
  - https://stackoverflow.com/questions/22803469/uevent-sent-from-kernel-to-user-space-udev
  - http://dist-prog-book.com/chapter/3/message-passing.html
  - https://qemu-project.gitlab.io/qemu/devel/qom.html
  - http://bitsquid.blogspot.com/2011/02/managing-decoupling-part-2-polling.html
  - https://gamedev.stackexchange.com/questions/207435/how-to-design-an-eventbusguided-by-the-pub-sub-pattern
- **Improvements for wubuwizard/wubuos**:
  - wubuwizard: add a typed event bus between loader, KV cache, tokenizer, and trainer (load-complete, OOM, quant-skip, decode-progress), each event a type tag + payload drained in the main loop, so modules stop reaching into each other's internals.
  - wubuos: expose kernel device add/remove notifications as events over the existing Styx9/9P namespace (netlink-style channel) so the GUI and personalities subscribe instead of polling global state.

# Theme F — Testing & Verification

### F1. Property-based testing in C (QuickCheck/Hypothesis-style) — how to property-test kernels
- **Hop chain**: seed: QuickCheck paper (Claessen & Hughes, ICFP 2000; ACM DL + Tufts PDF) → hop1: Hughes "Experiences with QuickCheck" (Quviq: stateful testing, two-model oracle) → hop2: Hypothesis (readthedocs; hypothesis.works compositional shrinking; issue #3411 minimal-failure analysis) → hop3: theft — property-based testing for C (silentbicycle) → hop4: qcheck (c-cube, OCaml) → hop5: RapidCheck (emil-e, C++ with shrinking) → hop6: pbt-frameworks overview (jmid) → hop7: "The sad state of property-based testing libraries" (Stevana) — the port-to-C lessons
- **Convergence**: Express invariants as properties — "decode(encode(x)) == x", "quantize→dequantize cosine ≥ bound", "no allocation after init" — and let a generator + shrinker run thousands of random cases, shrinking any failure to a minimal counterexample.
- **Sources**:
  - https://dl.acm.org/doi/10.1145/351240.351266
  - https://www.cs.tufts.edu/~nr/cs257/archive/john-hughes/quviq-testing.pdf
  - https://github.com/silentbicycle/theft
  - https://github.com/emil-e/rapidcheck
  - https://hypothesis.readthedocs.io/
  - https://stevana.github.io/the_sad_state_of_property-based_testing_libraries.html
- **Improvements for wubuwizard/wubuos**:
  - wubuwizard: add a property-test harness (own minimal generator+shrinker, or vendored theft) covering GGUF load→save→load round-trip, tokenizer encode→decode round-trip, and quant-kernel cosine bounds; wire as `make test_props`.
  - wubuos: property-test the 8086/VSL emulator by generating random instruction streams and asserting register/memory invariants hold after every step (no OOB, flag consistency), shrinking to the minimal failing stream.

### F2. Differential testing (oracle pattern — own implementation vs reference, e.g. gcc vs own compiler, llama.cpp reference kernels)
- **Hop chain**: seed: McKeeman "Differential Testing for Software" (Digital Technical Journal 1998; Semantic Scholar + Tufts PDF) → hop1: compiler-driven differential testing / CompDiff (ACM DL) → hop2: Csmith random C-program generator (regehr; github + Utah) → hop3: GCC vs clang/LLVM divergence testing (StackOverflow corpus) → hop4: differential testing of ML algorithms — framework disagreement study (Herbold 2022/2023, arXiv 2207.11976 + Springer EMSE; Passau summary) → hop5: DiffWatch — evolving differential testing (ACM DL) → hop6: quantization reference implementations (vLLM quantization guide; HF Optimum) → hop7: "First-Class Verification Dialects for MLIR" (Regehr et al., PLDI'25) — compiler-infrastructure verification practice
- **Convergence**: The oracle is the other implementation: run the same input through two independent implementations (own kernel vs scalar/reference kernel, own loader vs llama.cpp's, HolyC vs GCC) and flag divergence — no hand-maintained golden values needed.
- **Sources**:
  - https://www.semanticscholar.org/paper/Differential-Testing-for-Software-McKeeman/fc881e8d0432ea8e4dd5fda4979243cac5e4b9e3
  - https://www.cs.tufts.edu/comp/150FP/archive/bill-mckeeman/DifferentailTesting.pdf
  - https://github.com/csmith-project/csmith
  - https://dl.acm.org/doi/10.1145/3582016.3582053
  - https://arxiv.org/abs/2207.11976
  - https://users.cs.utah.edu/~regehr/papers/pldi25.pdf
- **Improvements for wubuwizard/wubuos**:
  - wubuwizard: add a differential harness running the own-C11 quant/dequant kernels against llama.cpp reference kernels on random tensors (cosine + max-diff thresholds) and the GGUF parser against the reference loader on identical files.
  - wubuos: differential-test HolyC-compiled expressions against GCC (compare emulated register results), and run the VSL NT-bridge against ReactOS behavior vectors for the same syscall sequences.

### F3. Fuzzing C codebases (libFuzzer, AFL++, honggfuzz) — how to fuzz parsers/decoders safely
- **Hop chain**: seed: libFuzzer docs (LLVM: in-process, coverage-guided, `LLVMFuzzerTestOneInput`) → hop1: Google fuzzing tutorial (libFuzzerTutorial.md) → hop2: AFL++ fuzzing-in-depth docs (instrumented compile + UI) → hop3: honggfuzz feedback-driven fuzzing (docs) → hop4: OSS-Fuzz (github; Serebryany USENIX Security'17) → hop5: ClusterFuzzLite (google.github.io) → hop6: structure-aware fuzzing (google/fuzzing docs; Serebryany libprotobuf-mutator slides; Grammarinator+libFuzzer) → hop7: sanitizer integration for fuzzing (ClusterFuzzLite ASan overview; VUSec COMbisan)
- **Convergence**: Coverage-guided in-process fuzzing with ASAN/UBSAN on, one narrow fuzz target per input format (a `LLVMFuzzerTestOneInput` that must never `exit()`), a seed corpus of real files, structure-aware mutations for anything with a grammar — and run it continuously (OSS-Fuzz style), not once.
- **Sources**:
  - https://llvm.org/docs/LibFuzzer.html
  - https://github.com/google/fuzzing/blob/master/tutorial/libFuzzerTutorial.md
  - https://aflplus.plus/docs/fuzzing_in_depth/
  - https://github.com/google/oss-fuzz
  - https://github.com/google/fuzzing/blob/master/docs/structure-aware-fuzzing.md
  - https://google.github.io/clusterfuzzlite/overview/
- **Improvements for wubuwizard/wubuos**:
  - wubuwizard: add `fuzz/` targets for the GGUF loader, tokenizer, and quant-schema parser built with `-fsanitize=fuzzer,address,undefined`, a seed corpus from real model files, and a time-boxed `make fuzz-run` CI job.
  - wubuos: fuzz the 8086 emulator's decoder and the VSL syscall dispatcher (syscall-number + argument byte blobs) with honggfuzz/libFuzzer under ASAN, and fuzz the HolyC lexer/parser seeded with real `.HC` sources.

### F4. Golden/snapshot tests + regression discipline for numeric kernels (tolerance-based)
- **Hop chain**: seed: characterization test / golden master (Wikipedia) → hop1: golden-master vs approval vs snapshot (understandlegacycode) → hop2: snapshot testing discipline — determinism requirement (Jest docs; Playwright) → hop3: floating-point assertions and tolerance practice (GoogleTest assertions reference; SO precision) → hop4: kernel regression discipline — LTP/kselftest/LKFT (StackOverflow; docs.kernel.org kselftest; LKFT) → hop5: LLM inference determinism limits (arXiv 2408.04667 "Non-Determinism of Deterministic LLM Settings"; Unstract) → hop6: GPU kernel verification by test amplification (UCSD; Yujie's blog) → hop7: MLIR testing guide (mlir.llvm.org: diagnostic/golden-file verification infrastructure)
- **Convergence**: Freeze observable outputs as golden files that change only through deliberate review (characterization), while numeric kernels compare against tolerances that scale with the computation (relative/ULP-based, never absolute), keyed by a hash of weights + inputs for reproducibility.
- **Sources**:
  - https://en.wikipedia.org/wiki/Characterization_test
  - https://understandlegacycode.com/blog/characterization-tests-or-approval-tests/
  - https://jestjs.io/docs/snapshot-testing
  - http://google.github.io/googletest/reference/assertions.html
  - https://arxiv.org/html/2408.04667v5
  - https://cseweb.ucsd.edu/~lerner/papers/verifying_gpu_kernels_by_test_amplification.pdf
- **Improvements for wubuwizard/wubuos**:
  - wubuwizard: add `tests/golden/` for the tokenizer and a fixed-seed generation trace (regenerated only via a review-gated script), and formalize the existing cosine checks into a per-quant-type tolerance table in a numeric regression suite keyed by weights+input hash.
  - wubuos: add a golden-boot test — boot the kernel in the emulator, hash the VGA framebuffer and capture the syscall trace, assert byte-identical across builds — plus tolerance-based checks for the floating physics/RL paths.

### F5. CI for OS/kernel/inference projects (GitHub Actions for C11, matrix builds, ASAN/UBSAN stages)
- **Hop chain**: seed: GitHub Actions matrix strategy (docs.github.com running variations) → hop1: cmake-action (GitHub Marketplace) → hop2: ccache-action for C/C++ CI speedup (hendrikmuhs; r/cpp ccache-in-CI thread) → hop3: sanitizers in CI (LLVM discourse reproducible ASan/UBSan; r/cpp "Sanitizers in continuous integration") → hop4: Linux kernel testing & CI (kernel-recipes: LKP/0-day, LKFT tiers) → hop5: 0-day CI service for kernel quality (opensourcevoices/Medium) → hop6: syzkaller coverage-guided kernel fuzzing as CI (github) → hop7: ClusterFuzzLite continuous fuzzing for repos
- **Convergence**: Tiered CI modeled on kernel practice: a fast tier on every push (compiler × sanitizer matrix, ccache-backed), a slow tier nightly (fuzzing, emulator boots), with the build matrix covering gcc/clang × -O2/ASAN/UBSAN so regressions are caught at the cheapest tier first.
- **Sources**:
  - https://docs.github.com/actions/writing-workflows/choosing-what-your-workflow-does/running-variations-of-jobs-in-a-workflow
  - https://github.com/hendrikmuhs/ccache-action
  - https://archives.kernel-recipes.org/wp-content/uploads/2025/01/Linux_20kernel_20testing_20and_20CI.pdf
  - https://medium.com/@opensourcevoices/0-day-continuous-integration-ci-test-service-helps-ensure-linux-code-quality-a6d45edeb523
  - https://github.com/google/syzkaller
  - https://google.github.io/clusterfuzzlite/overview/
- **Improvements for wubuwizard/wubuos**:
  - wubuwizard: add a GitHub Actions workflow with an ubuntu matrix (gcc/clang × -O2/ASAN), ccache caching, `make test_all` + `make test_kernel_dispatch` on every push, and a nightly time-boxed libFuzzer job.
  - wubuos: kernel CI that builds with gcc and clang, boots the kernel under QEMU for a framebuffer-hash smoke test, runs the E1 NT-bridge regression + VSL tests in the matrix, and a nightly syzkaller-style syscall fuzz of the emulated personalities.
