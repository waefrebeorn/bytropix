# THE EVERYTHING-HARDWARE MAP — chipsets, ISAs, and the tools for all of them (7-hop, 2026-08-04)

Directive: "online research all the chipsets and hardware — AGI goes on
everything in some way and has tools for everything."

Method: 7-hop across the ENTIRE hardware universe — CPU ISAs, GPU ISAs,
MCUs, DSPs, NPUs, the dead, the future. The convergence that proves the
driver-space doctrine: **llama.cpp runs the SAME GGUF model on 15
backends** (CUDA/HIP/Metal/Vulkan/OpenCL/SYCL/WebGPU/CANN/MUSA/Hexagon/
OpenVINO/RPC/ZenDNN/BLAS) — one model, N hardware targets. That is the
hourglass-neck pattern (our wubu_mir) proven at AGI scale. And the
Universal GPU ISA paper (arXiv 2603.28793): NVIDIA's dominance "persists
not because of hardware superiority but because of software lock-in."
**The compiler/driver layer is the unlock. We are in the right business.**

## Wave 1: the CPU ISA ladder (the full taxonomy)

| family | bits | lineage | status |
|---|---|---|---|
| x86 (8086→x86-64) | 16/32/64 | 8080; won by backward-compatible modes | dominant |
| ARM (ARM1→AArch64) | 32/64 | Acorn, Berkeley-RISC-inspired | 230B chips |
| RISC-V | 32/64/128 | Berkeley, open, RVA23 profile | the open future |
| MIPS | 32/64 | RISC-II lineage | migrating TO RISC-V |
| PowerPC | 32/64 | AIM alliance | legacy (PS3/360 died) |
| SPARC | 32/64 | Berkeley RISC-I | dead (Solaris only) |
| Motorola 68000 | 16/32 | the clean CISC | DONE — our driver runs it |
| 6502 | 8 | ex-6800, $20 | retro (our 8-bit type-set proof) |
| Z80 / 8080 | 8 | Faggin | retro |
| 8051 | 8 | Intel | STILL in production 40+ yrs |
| LoongArch | 64 | Chinese MIPS-heritage | national champion |
| C-SKY | 32/64 | Chinese | RISC-V-adjacent |
| VAX / Alpha / Itanium | 32/64 | DEC/HP-Intel | dead (Itanium: compiler wasn't clever enough — OUR mandate) |
| PA-RISC / m88k / i860 / i960 | 32/64 | HP/Motorola/Intel | dead |

## Wave 2: GPU ISAs (the parallel driver space)

The Universal GPU ISA paper (arXiv 2603.28793) — the first cross-vendor
GPU ISA analysis, 16 microarchitectures, 15+ years:

| vendor | ISA | archs | notes |
|---|---|---|---|
| NVIDIA | PTX (virtual) | Fermi→Blackwell | the de facto standard; virtual ISA with per-thread scalar semantics |
| AMD | RDNA 1-4 + CDNA 1-4 | ~4,800 pages of ISA guides | native ISA, machine-readable XML specs; CDNA = compute, RDNA = graphics |
| Intel | Gen11, Xe-LP/HPG/HPC | SIMD-register ISA, message-passing | SYCL/oneAPI |
| Apple | G13 | reverse-engineered | Metal |
| mobile | Mali, Adreno, PowerVR | OpenCL/GLES | the "every phone" tier |

The paper's thesis = our thesis: PTX's dominance is SOFTWARE LOCK-IN.
The driver space should treat PTX, GCN/RDNA, Xe, and G13 as four more
DRIVERS (or one Vulkan/WebGPU front, see Wave 6).

## Wave 3: the MCU zoo (the everything-tier — where the AGI lives in things)

- **8051** — the industry legend, still shipping (Silicon Labs EFM8 at 72 MHz, the fastest 8-bit).
- **PIC** (Microchip) — PIC16/PIC18/PIC24/PIC32; PIC32MM won the $1-MCU roundup.
- **AVR** (Atmel/Microchip) — 8-bit RISC; the Arduino classic; TinyAVR.
- **MSP430** (TI) — 16-bit ultra-low-power; the power-efficiency champion.
- **STM8** (ST) — "my favorite 8-bit MCU... SP+offset addressing is very handy."
- **ARM Cortex-M0/M0+/M4/M7** — the M0+ is now CHEAPER than 8-bit parts ("the cheapest flash MCU you can buy is an ARM Cortex-M0+").
- **RISC-V MCUs** — WCH CH32V003 at **US$0.10** (the cheapest chip on earth); GD32V; ESP32-C (RISC-V); RP2350 (Hazard3 cores); Bouffalo BL60x/BL70x; NEORV32 (VHDL, configurable).
- **ESP32** — the IoT era's WiFi/BLE MCU (Xtensa cores + RISC-V variants).

The AGI "on everything": the 8-bit and 16-bit tier is where a THIN
interpreter driver (like our m68k interpreter) makes the AGI's code run
on a $0.10 chip. SERV (the world's smallest RISC-V, 125 LUTs in FPGA,
2.1 kGE ASIC) is the physical floor — our RV32I interpreter would run
on it.

## Wave 4: DSPs (VLIW and the signal tier)

- **Qualcomm Hexagon** — in-order 4-wide VLIW, 6-way SMT, 32 regs/thread, HVX vector coprocessor, runs compiled C, virtual memory. In EVERY Snapdragon. llama.cpp has a Hexagon backend in progress.
- **TI C6000/C2000** — the classic VLIW DSPs.
- **ADI SHARC** — conditional arithmetic, divide/sqrt, bit-field ops.
- **Movidius SHAVE** — the vector DSP inside Intel's NPU tiles.

DSP lesson: VLIW is a compiler-driven ISA — "writing efficient code for
this requires architecture-specific knowledge" = the compiler is the
product. Our mandate again.

## Wave 5: NPUs (the neural tier — where the AGI's WEIGHTS live)

- **Google TPU** — ASIC for neural nets, bfloat16, private interfaces.
- **Apple Neural Engine** — in every iPhone/Mac.
- **Intel NPU** (Meteor Lake+) — DL accelerator on PCIe, LeonRT 32-bit
  scheduler MCU + NCE tiles (MAC engines) + 2× Movidius SHAVE DSPs,
  INT8/FP16.
- **AWS Trainium/Inferentia**, **Qualcomm NPU** (in the Hexagon complex).
- NPUs claim >100× GPU efficiency per watt for the same inference.

NPUs aren't "run the AGI" — they're "run the AGI's matrix multiply."
The right tool for them is the EXISTING model layer (GGUF → ONNX →
TensorRT/OpenVINO/CANN), not a CPU driver. "Tools for everything" means
the right tool per tier.

## Wave 6: the run-everywhere software stack (THE proven pattern)

llama.cpp backends (one model → every device):

| backend | targets |
|---|---|
| CUDA | NVIDIA GPU (PTX/SASS) |
| HIP | AMD GPU |
| Metal | Apple Silicon |
| Vulkan | any GPU (incl. our WSL RTX 4050 via /dev/dxg) |
| WebGPU | ANY browser GPU (Dawn native too) — "run almost any model in your browser" |
| OpenCL | Adreno/mobile |
| SYCL | Intel GPU/CPU |
| CANN | Ascend NPU |
| OpenVINO | Intel CPU/GPU/NPU |
| MUSA | Moore Threads GPU |
| Hexagon | Snapdragon DSP |
| RPC | everything over the wire |
| BLAS/BLIS/ZenDNN | CPU |

WebGPU is the universal one: works in every browser + natively via
Dawn, no vendor lock, runs FlashAttention + GEMM kernels written in
WGSL. This is the DRIVER SPACE'S SOFTWARE COMPLEMENT: for the tiers
where writing a native encoder is disproportionate (every mobile GPU),
a WebGPU/Vulkan front is the driver.

## The tools-for-everything doctrine (what we build)

1. **CPU tier** — native encoders + interpreters (driver space: x86-64
   DONE, m68k DONE, riscv next, then 6502 as the 8-bit proof, Z80,
   8051, MSP430, PIC — each with an oracle-verified encoding table +
   bundled interpreter; the differential battery extends per ISA).
2. **GPU tier** — PTX/GCN/Xe/G13 drivers OR one Vulkan/WebGPU front
   (Vulkan compute already exists in WuBuOS for Bear RL —
   wubuos-vulkan-compute skill). Same MIR, SIMD-shaped wave ops.
3. **NPU tier** — no ISA driver; the GGUF/ONNX bridge (already planned
   in wubuwizard: safetensors bridge, gguf loader, llama.cpp
   integration).
4. **MCU tier** — thin interpreters (m68k interpreter IS the template:
   ~200 lines, decodes the driver's own bytes, tiny RAM footprint).
5. **The oracle** — tools/verify_isa.sh + the differential battery is
   the universal methodology for every new driver. binutils multiarch
   covers dozens of ISAs (m68k, riscv, arm, aarch64, mips, powerpc,
   sparc, s390x, i386, x86-64...): `objdump -m <isa>` is the free
   encoder oracle.

## The hardware detection angle

"we work by using detection, methodologies and known research" — the
AGI should DETECT its host at runtime:
- `uname -m` / `lscpu` on Linux, PowerShell on Windows (disk truth rule)
- /dev/dxg + nvidia-smi.exe for GPU (our WSL box: RTX 4050, sm_89)
- CPUID leaf 1 for x86 features (AVX/AVX2/AVX512/AMX)
- lscpu on RISC-V/ARM reports the ISA extension set
- Then pick the driver: native if host ISA, interpreter if not,
  RPC/WebGPU if remote.

Detection → driver selection → differential self-test → run. That's
the "AGI goes on everything in some way" mechanism.
