# THE ISA LADDER — the driver space for EVERY chip (7-hop, 2026-08-04)

Method: Kevin-Bacon 7-hop across the ENTIRE history of instruction-set
architectures. Purpose (user directive, 2026-08-04): "as an AGI we need
to run on everything even if it's on a Motorola 68,000 — we have
researched all of the past and we know where we are by knowing where we
aren't; we make proper modules for our system so we can have all of the
programming languages and all of the chip sets." Saved to the corpus for
AGI training. The compiler frontend emits ONE mid-level IR (wubu_mir),
and every ISA is a DRIVER (wubu_isa_driver.h) that consumes it — the
driver space. Adding a chipset = adding a driver, never touching the
frontend.

Triple-DA: (1) correctness — each driver differential-tested against gcc
on the 33-expression battery; (2) privacy/safety — no third-party code,
all self-contained C11; (3) robustness — interpreters bound memory,
encodings self-consistent (emitter+decoder agree).

## The 7-hop ladder (every family that mattered, in lineage order)

| Hop | ISA | Year | Lineage / key fact | The driver's essence |
|-----|-----|------|--------------------|----------------------|
| 1 | **Motorola 6800 → 6502** | 1975 | 6502 = ex-Motorola team (Peddle) simplifying the 6800: one accumulator, 8-bit stack, 56 instrs, "the $20 CPU" that won Apple II/C64/Atari/NES. "As close as you can get to a true 8-bit CPU." | 8-bit acc + X/Y index, 16-bit address bus, zero-page, little-endian. |
| 2 | **Zilog Z80** | 1976 | 8080 designers left Intel (Faggin), made a MORE capable chip that stayed 8080-compatible. 8-bit data, 16-bit address, dual register banks, index regs IX/IY. CP/M + ZX Spectrum/TRS-80/MSX. | 1-4 byte variable instrs, accumulator-centric, BCD. |
| 3 | **Intel 8086 → x86-64** | 1978 | The "quick hack" 16-bit follow-on to the 8080 that became the world. 80386 (1985) made 32-bit; AMD's x86-64 (1999-2003) won 64-bit over Intel's own Itanium. Backward compatibility by MODES (real/protected/long). The duopoly that only refinement could defend. | CISC variable-length, 8/16 GPRs, segments → flat long mode, the encoding we JIT today. |
| 4 | **Motorola 68000** | 1979 | THE clean CISC: 8 data (D0-D7) + 8 address (A0-A7, A7=SP) 32-bit regs, 56 orthogonal instrs, flat unsegmented 24-bit address space, big-endian, variable 16-bit+ words, condition codes N/Z/V/C/X. Mac/Amiga/Atari ST/arcade. "A breath of fresh air after 6502/Z80/8086." | The USER'S FLAGSHIP: our driver emits RV-style MIR into real 68k opcodes, executed by our own interpreter. |
| 5 | **ARM (Acorn RISC Machine)** | 1985 | Acorn (BBC Micro, a 6502 house) needed 10x perf at BBC price; Berkeley RISC papers + a WDC visit convinced them to DESIGN their own chip. ARM1 1985, ARMv8/AArch64 2011. Hardwired, no microcode, like the 6502. 230B+ chips: the most widely used ISA ever. | 32-bit fixed-width RISC, 16 GPRs, cond codes baked into every instr, barrel shifter. AArch64: 31×64-bit GPRs. |
| 6 | **MIPS → RISC-V** | 1981→2014 | Patterson's Berkeley RISC-I/II (1981) → SOAR "RISC-III" → SPUR "RISC-IV" → RISC-V (2010, Asanović, "the 5th generation"). The only truly open ISA: royalty-free, 4500+ members, from 10-cent MCUs (CH32V003) to Alibaba server CPUs. M extension = MUL/DIV/REM (what our driver emits). | Clean load-store RISC, 32 × 64-bit regs (x0 hardwired 0), I/R/S/B/U/J formats, the ISA our driver space already proves. |
| 7 | **The dead giants (VAX, Alpha, SPARC, PA-RISC, POWER, Itanium)** | 1977-2001 | The graveyard that teaches the lesson: VAX CISC complexity collapsed under RISC; Alpha/SPARC/PA-RISC died to consolidation; **Itanium (EPIC/VLIW) died because the COMPILER wasn't clever enough** — "no one was clever enough to write that compiler" — plus AMD64 evolutionary upgrade won over revolutionary IA-64. "Baking microarchitectural decisions into the ISA" was the cardinal error. | OUR EDGE: the best compiler is the unlock. The driver space + MIR is exactly the retargetable low-level IR Itanium lacked — and our differential harness (we know where we are by knowing where we aren't) is the correctness backbone. |

## The convergence principle

1. **Every ISA is a driver.** The frontend (HolyC/any language) → MIR →
   driver. x86-64 (native JIT), RISC-V (RV64I + interpreter), Motorola
   68000 (the "even if it's a 68,000" proof), PTX (GPU), ARM64,
   6502/Z80 (8-bit, the type-set extremes). One frontend, N backends.
2. **We know where we are by knowing where we aren't.** The 33-expression
   differential battery runs EVERY driver against gcc. A driver that
   disagrees is a FINDING. This is how a hand-written 68k encoder earns
   trust without silicon.
3. **Type sets are the width ladder.** MIR ops are typed by the ISA's
   natural width: 8/16/32/64-bit drivers (6502=8, 68k=32, RISC-V/x86-64
   =64). The IR is width-agnostic; the driver widens/truncates. This is
   the "all chip sets AND all type sets" half of the directive.
4. **Itanium's lesson is our mandate.** VLIW failed because compilers
   couldn't exploit it. OUR compiler exists precisely to be clever
   enough — the driver space is the infrastructure that makes "any
   hardware" a compiler problem, not a rewrite problem.

## The driver space (built 2026-08-04, wubuos)

```
wubu_isa_driver.h   the contract: name/family/exec, compile(MIR)->bytes, run(bytes)->result
wubu_isa_x86_64.c   native JIT driver (this machine) — the same MIR
wubu_isa_riscv.c    RV64I encoder + wubu_riscv_interp.c (executes the bytes)
wubu_isa_m68k.c     Motorola 68000 encoder + wubu_m68k_interp.c (the flagship)
holyc_ptx.c         the existing NVIDIA GPU backend (the driver space swallows it)
tools/mir_driver_test.c  the differential battery across ALL drivers vs gcc
```

## Sources (the hops)

- Motorola 68000: Wikipedia (1979, 16/32-bit CISC, 8+8 regs, 24-bit addr),
  iljitsch.com "6502, Z80, 8086, 68000: the four CPUs that dominated the 1980s",
  the M68000 Programmer's Reference Manual (NXP PDF).
- 6502: Wikipedia (Chuck Peddle, ex-6800 team, $20, Apple II/C64/NES), the
  iljitsch 4-CPU retrospective.
- Z80: Wikipedia + the iljitsch retrospective (Faggin, 8080-compatible).
- x86: Wikipedia (8086 1978 → 80386 1985 → AMD64), handmade.network
  "x86: the history of naming" (modes = the compatibility mechanism).
- ARM: Wikipedia (Acorn 1985, Sophie Wilson/Steve Furber, Berkeley RISC
  influence, WDC visit, ARMv8 2011, 230B chips, Fugaku #1 supercomputer).
- RISC-V: Wikipedia (Asanović 2010, 5th Berkeley gen, open ISA, RV32/64/128,
  M extension, CH32V003 10-cent MCU, Alibaba server CPUs).
- Dead giants: HN "AMD Killed the Itanium" thread (VLIW/compiler lesson,
  Alpha/SPARC/PA-RISC consolidation), SE "Why was Itanium difficult to
  write a compiler for" (speculative loads, template NOP-padding, PGO),
  Quora (evolutionary AMD64 vs revolutionary IA-64).
