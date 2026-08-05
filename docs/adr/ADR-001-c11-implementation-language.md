# ADR-001: C11 as the Implementation Language

- **Status:** accepted
- **Date:** 2026-08-05

## Context

The engine must run where the Body runs — including bare-metal WuBuOS,
hosted Linux/WSL, and future GPU/Metal paths. It must be self-contained
(no runtime dependencies), agent-workable (small modules, greppable),
and verifiable (sanitizers, static analysis, formal oracles).

## Decision

All implementation is strict C11 (`-std=c11`), no C++, no exceptions, no
RTTI. Only the CUDA kernel path compiles as C++ (nvcc requirement).

## Consequences

- **Positive:** single toolchain, ABI-stable opaque APIs, ASan/UBSan
  compatible, trivially embeddable in the OS kernel and the hosted binary.
- **Negative:** no language-level ownership (manual arenas/RAII-in-C
  discipline), no generics (X-macros and codegen fill the gap).
- **Alternatives rejected:** C++ (ABI + exception + RTTI weight, two
  toolchains), Rust (toolchain weight, not writable in-kernel here).

## Verification

`-std=c11` in both Makefiles; engine builds with GCC and Clang.
