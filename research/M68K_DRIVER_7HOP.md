# THE M68K DRIVER — oracle-verified encodings (2026-08-04)

Directive: "as an AGI we need to run on everything even if it's on a
Motorola 68,000." The 68000 driver is the third in the driver space
(x86-64 JIT, RISC-V interp planned, m68k interp DONE). The rule that
made it correct: **no guessed opcodes — every encoding verified
byte-for-byte against GNU binutils objdump (m68k:68000) via
tools/verify_isa.sh.**

## The oracle caught a real bug in minutes

Hand-derived MOVE encodings (from the classic "0x2000 | n<<9 | m<<3 | 4"
pattern) were WRONG. objdump showed:
- `0x2000` = `movel %d0,%d0`  (so MOVE.L D0,D0 is 0x2000, NOT 0x2004)
- `0x2004` = `movel %d4,%d0`  (bit 2 = SRC REGISTER, not size!)
- `0x2008` = `movel %a0,%d0`  (bit 3 = src mode An)
- `0x2010` = `movel %a0@,%d0` (bit 4 = src mode (An))
- `0x2080` = `movel %d0,%a0@` (bit 7 = DEST mode (An))
- `0x2200` = `movel %d0,%d1`  (bit 9 = DEST register)

## The REAL MOVE format (verified)

```
00 ssss ddd mmm eee gg 0   (16-bit word, big-endian)
  15-14 = 00
  13-12 = size  (01=B, 10=L, 11=W)
  11-9  = DEST register
  8-6   = DEST mode
  5-3   = SRC mode
  2-0   = SRC reg
Modes: 000=Dn 001=An 010=(An) 011=(An)+ 100=-(An) 101=(d16,An) 111=imm
```

So: `MOVE.L Dn,Dm = 0x2000 | (m<<9) | n` (DEST in 11-9, SRC in 2-0).

## Verified instruction table (the driver's subset)

| instr | encoding | objdump |
|---|---|---|
| MOVE.L Dn,Dm | 0x2000\|(m<<9)\|n | movel %d1,%d0 |
| MOVE.L #imm32,Dm | 0x203C\|(m<<9) + imm32 | movel #5,%d0 |
| MOVE.L Dn,(d16,A6) | 0x2D40\|n + d16 | movel %d0,%fp@(-4) |
| MOVE.L (d16,A6),Dm | 0x202E\|(m<<9) + d16 | movel %fp@(-4),%d0 |
| ADD.L Dn,Dm | 0xD080\|(m<<9)\|n | addl %d1,%d0 |
| SUB.L Dn,Dm | 0x9080\|(m<<9)\|n | subl %d1,%d0 |
| AND.L Dn,Dm | 0xC080\|(m<<9)\|n | andl %d1,%d0 |
| OR.L Dn,Dm | 0x8080\|(m<<9)\|n | orl %d1,%d0 |
| CMP.L Dn,Dm | 0xB080\|(m<<9)\|n | cmpl %d1,%d0 |
| EOR.L Dn,Dm | 0xB180\|(n<<9)\|m | eorl %d1,%d1 |
| MULS.W Dn,Dm | 0xC1C0\|(m<<9)\|n | mulsw %d1,%d0 |
| DIVS.W Dn,Dm | 0x81C0\|(m<<9)\|n | divsw %d1,%d0 |
| NEG.L Dn | 0x4480\|n | negl %d1 |
| NOT.L Dn | 0x4680\|n | notl %d1 |
| TST.L Dn | 0x4A80\|n | tstl %d1 |
| MOVEQ #imm8,Dn | 0x7000\|(n<<9)\|imm | moveq #1,%d0 |
| SUBQ.L #1,Dn | 0x5380\|n | subql #1,%d1 |
| LSL.L #1,D0 | 0xE388 | lsll #1,%d0 |
| LSR.L #1,D0 | 0xE288 | lsrl #1,%d0 |
| LINK A6,#-N | 0x4E56 + s16 | linkw %fp,#-80 |
| UNLK A6 | 0x4E5E | unlk %fp |
| RTS | 0x4E75 | rts |
| BRA.s d | 0x6000\|d | bras |
| BEQ/BNE/BGT/BGE/BLT/BLE.s | 0x6700/0x6600/0x6E00/0x6C00/0x6D00/0x6F00\|d | beqs/bnes/... |

## The ALU pattern (the non-obvious bit)

`ADD.L D0,D0 = 0xD080` — the size (10=long) lives in bits 7-6, and the
source Dn mode lives in bits 5-3 with the src register in 2-0. The
"0xD000 | n<<9 | m<<3 | 4" folk pattern was wrong on THREE fields.
EOR is reversed (src at 11-9, dest at 2-0) because it's `EOR Dn,<ea>`.

## The frame strategy (verified at runtime)

- `LINK A6,#-N` allocates N bytes BELOW A6 (the fp). Slots must be at
  NEGATIVE displacements: `(A6 - (vr+1)*4)` = `movel %fp@(-4),%d0`.
  Positive offsets point ABOVE the frame into the caller's stack — the
  same bug the x86-64 driver had (slots at [rbp+off] instead of
  [rbp-off]) → SIGSEGV on return. THE lesson: a frame pointer means
  NEGATIVE offsets, always.
- Result in D0 on RTS (the 68000's return convention).

## The second-order lesson: BATTERY finds what oracle can't

The oracle verifies EACH OPCODE; the differential battery (33
expressions × every driver, vs gcc-known values) verifies the WHOLE
program. It found two bugs the oracle couldn't:
1. `5^3` = 0 everywhere: `bitxor_ops[]` used `HC_AST_XOR` — a name that
   DOESN'T EXIST in the enum (it's `HC_AST_BITXOR`), so the parser
   silently emitted garbage kind 58. C: an enum misspelling is NOT a
   compile error — it's a silent wrong constant.
2. `1<2` = 0 on m68k only: the SUB/CMP flag math reused the ADD
   overflow formula with ~a+1, computing V=1 for `1-2` (no overflow!)
   → LT broke. Subtraction needs its own V: `(dst^src)&(dst^res)&sign`.

## Tools made permanent (not in /tmp)

- `tools/verify_isa.sh` — the encoding oracle: packs hex words,
  disassembles with the multiarch objdump (found at
  ~/opt/binutils-multiarch, /opt, then /tmp fallback; WUBU_BINUTILS
  override). Use BEFORE shipping any new driver encoding.
- `make test_drivers` — the differential battery (x86-64 + m68k,
  33/33 green). Extend the expr list when adding ISA support.
- Binutils extraction (no root needed):
  `cd /tmp && apt-get download binutils-multiarch && dpkg -x *.deb ~/opt/binutils-multiarch`
