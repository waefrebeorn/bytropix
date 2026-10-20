# Portfolio reality check — 2026-10-18 (cycle 8)

Before cycle 8 could close anything in KERNEL / FS / RUNTIME / WORLDSIM, each
candidate goal was checked against the code it names. Most do not match. This
page records the evidence so no future cycle closes them on the strength of a
filename.

Triage currently calls a goal **active** when its module path exists. That is
too weak a test: a module can exist and the *capability* be absent. The rows
below are exactly that case.

| Goal | Claimed capability | What the code actually is | Evidence |
|---|---|---|---|
| KERNEL #29 | 4-level page-table walker, 2 MiB leaves, PML4/PDPT/PD/PT bit checks | **implemented in cycle 8** as `src/kernel/wubu_pt.c`; `test_wubu_pt` hand-walks every level (34/34) | this row is now closed; the original bitmap allocator is untouched and still separate |
| KERNEL #30/#31 | buddy orders 0..10 + per-CPU magazines; bump→buddy handoff | no `buddy`/`bump`/`magazine` symbols in `memory.c` | grep over `src/kernel/memory.c` |
| KERNEL #32 | context switch asm: 1e6 switches, callee-saved GPRs + RFLAGS identical | cooperative scheduler test with **no r12–r15/RFLAGS assertions** | `src/kernel/test/legacy/tasking_test.c` |
| KERNEL #984 | ACPI table checksum verification + XSDT/RSDT selection | XSDT preference exists; **no checksum function** | grep `checksum` in `wubu_acpi.c` returns comments only |
| FS #808 | resumable readdir cookie over FAT32 | `fat32_dir.c` has **no readdir/cookie** | grep returns nothing |
| FS #812 | FSInfo free-cluster count kept authoritative | FSInfo is **read once at format**; no mirror on alloc/free | `fat32_format.c:48-56` |
| FS #813 | FAT12/16 mounted through the same driver as FAT32 | FAT table ops and type detection only | `test_fat2.c` B01–B04 |
| FS #815 | volume label/serial read + boot-sector self-repair on mount | label is **written** at format only | `fat32_format.c:166-172` |
| RUNTIME #721 | exec into a container, namespace join, exit-code round-trip | no `test_ct_exec` target exists | `mk/tests.mk` |
| RUNTIME #722 | posix_spawn file actions | no `test_posix_spawn` target exists | `mk/tests.mk` |
| RUNTIME #728 | NT job limits/accounting | no `test_vsl_nt_job_ext` target exists | `mk/tests.mk` |
| RUNTIME #729 | nested suspend counts, STATUS_SUSPEND_COUNT_EXCEEDED | only `vsl_nt_suspend_thread` (SIGSTOP); **no resume handler, no count, no status constant** | `vsl_nt_thread.c` |
| WORLDSIM #564 | PCG-XSH-RR world RNG with statistical battery | **xorshift64** | `test_worldsim.c:16-19` |
| DRIVERS #960 | PCI capability-list walk (MSI-X/PME/PCIe caps) | no capability-walk symbols | grep `wubu_pci.c` |

## Why this matters

The registry currently reports 216 "active" goals. A meaningful fraction of
them are active only in the sense that the file they name exists. Closing one
of these on a green test would be the exact behaviour the whole portfolio
exercise exists to prevent: **a passing test that proves something else.**

## Two honest options

1. **Implement them for real.** Each row above is a multi-hour feature, not a
   gap to be relabelled. KERNEL #29 alone means writing a page-table walker and
   a mapping API; FS #808 means a resumable cluster iterator with a fixture of
   thousands of entries.
2. **Re-scope the registry.** Introduce a fifth triage class — `nominal`:
   module exists, capability absent. Move those goals there with the evidence
   above, so `active` means "the code is there and the work is integration",
   not "the filename resolved".

Either is defensible. What is not defensible is continuing to count these as
active and then closing them with a test that asserts something adjacent.

## Proposed triage change (option 2)

- `active` — module exists **and** the capability's entry point is present.
- `nominal` — module exists, capability absent (this page is the evidence).
- `blocked` — module absent.
- `deferred` — real, but scheduled.

Under that definition KERNEL's 33 active goals would split sharply, and the
next cycle's target would be chosen from a list that can actually be finished.

## Status

Cycle 8 is **paused at this decision**. No goal was closed from these domains;
no test was written to fit a goal after the fact. The body gate is unchanged
and green from cycle 7 (45 suites, 1506 checks, 14 XFAIL).

## Cycle 12 (2026-10-20) — verified rows, measured with QEMU ground truth

| Goal | Claimed capability | What the code actually is now | Evidence |
|---|---|---|---|
| KERNEL #474 | LAPIC + IOAPIC init and the EOI path | **verified + fixed this cycle** (wubuos 4aaba3a6): apic_init enables the LAPIC via MSR + masks LVTs (`src/kernel/interrupt_apic.c:14-73`); `lapic_eoi` writes LAPIC_EOI (`interrupt_apic.c:147-153`); `interrupt_eoi` covers LAPIC-owned vectors, not just I/O-APIC-routed ones (`src/kernel/interrupt.c:222-241`); `pit_handler` EOIs FIRST so an abandoned frame can never poison the priority class (`interrupt.c:290-311`); `wubu_apic_enable` reads back the LVT and panics loudly on a dead MMIO window (`src/kernel/wubu_apic.c:100-117`). QEMU HMP `info lapic` during the freeze showed `ISR 32` stuck + `IRR 32` pending + `PPR 0x20` (abandoned-EOI signature) and after the fix `ISR (none)`, timer armed, tick chain live | two-tier `tools/boot_gate.sh` PASS 2/2 (firmware milestones + full chain: kernel_main entered, attestation VALID, bonzi heartbeat); `src/firmware/run-agi.sh` RESULT: PASS |
| KERNEL #560 | Idle task with HLT and a per-CPU jiffy wakeup | **verified + fixed this cycle**: `task_idle` hlt + `task_yield` per wake so a woken task actually schedules (`src/kernel/tasking.c:405-417`); the trampoline stis so a first-run task takes interrupts (`tasking.c:433-438`); a resumed task always runs IF=1 (`tasking.c:483-495`). The heartbeat arriving at ~1s of supervisor uptime is the observable: the timer wakes the hlt, the rotation schedules the sleeping bonzi task, and `bonzi: heartbeat regions=34 attest=VALID` appears on the serial log | `run-agi.sh` RESULT: PASS (heartbeat check); boot gate 2/2 |
