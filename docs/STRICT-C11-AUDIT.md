# Strict-C11 audit (STACK #773) — 2026-10-18

Goal #773 asks for `gcc -std=c11 -pedantic -fsyntax-only` over every `CORE_OBJ`
source to produce an **empty diagnostic file**. It does not. This page records
the real measurement, the fixes applied, and what is left.

Run it:

```
bash tools/audit_c11.sh            # summary, exit 1 while not clean
bash tools/audit_c11.sh --list-errors
```

The script maps `.o → .c`, skips `.cu`/`.S` (not C11 translation units), and
counts diagnostics per file.

## Measurement

| | Before | After |
|---|---:|---:|
| CORE_OBJ C sources | 197 | 197 |
| Diagnostic lines | 455 | **105** |
| `error:` lines | 10 | **0** |
| Files with diagnostics | 38 | 6 |

Second pass (same day): 386 → 162 by adding the missing standard headers
(`stdio.h`, `stdlib.h`, `strings.h`, `unistd.h`, `alloca.h`) and the
feature-test macros the POSIX calls need (`_POSIX_C_SOURCE` for `posix_memalign`
/ `popen` / `strdup` / `ftruncate`, `_GNU_SOURCE` for `memmem`). Those were
correctness bugs: an implicit `snprintf` is assumed to return `int`.

Third pass: 162 → 105 by guarding the duplicate `GQA_KV_DIM` definition in
`wubu_model.h` (the runtime alias in `wubu_dims.h` is the owner), which removed
the redefinition warnings and the knock-on format diagnostics.

## What was fixed

All 10 errors were the same class: the sources relied on GNU extensions that
the default `gnu17` build exposes and strict `-std=c11` hides. Eight files got
an explicit feature-test guard:

```c
#if defined(__STRICT_ANSI__)
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE 1
#endif
```

- `CLOCK_MONOTONIC` — wubu_model.c, wubu_codeexec.c, wubu_gamebud.c, wubu_scale.c
- `M_PI` — wubu_acq.c, wubu_audio.c (×2), wubu_kvvq.c
- `MADV_WILLNEED` / `MADV_HUGEPAGE` — wubu_gguf_header.c (×2)

These were real portability bugs: any build with a strict standard (or a
different libc) would have failed, while the default build hid them.

## What remains (386 lines, 32 files)

| Warning | Count | What it means |
|---|---:|---|
| ISO C forbids braced-groups within expressions | 10 | GCC statement-expressions `({ ... })` — not C11 |
| `GQA_KV_DIM` redefined | 6 | macro defined twice with different values |
| const-qualifier discards in tokenizer calls | 2 | `vocab_find` / `apply_bpe` take `char *` |
| `wubu_plugin_api` declared inside a parameter list | 1 | struct scope leak |
| incompatible pointer type on `api->cleanup` | 1 | plugin vtable signature mismatch |

The implicit-declaration family (17 lines in the first pass) is gone.

Next: the 10 braced-groups need rewriting as ordinary C (or a helper function),
and `GQA_KV_DIM` needs one definition in one header.

## Status

**#773 stays OPEN.** Zero errors is progress; an empty diagnostic file is the
criterion and 105 lines remain. The audit target is deliberately *not* in
`test_all`: a gate that fails on a known 386-line debt would be turned off
within a day. It becomes a gate when the file is empty, and that is the point
of the goal.
