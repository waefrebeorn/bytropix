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

| NAMESPACE #591 | Tstat/Twstat Dir encode/decode round-trip fidelity | **fixed + verified this cycle**: Twstat now answers Rwstat (was Rclunk -- the test asserted the bug, 0intro:296-306) and the dispatch parses the full Dir; five missing client builders added (`styx_build_tauth/tflush/tcreate/tremove/twstat`, src/runtime/styx_enc.c) | `make test_styx_serve` ALL PASSED -- the wstat callback receives dev=9/qid.path=777/mode=0600/atime/mtime/length/name/uid/gid/muid field-exactly as encoded; Tstat Rstat decodes back with the same fidelity |
| NAMESPACE #592 | Tversion msize negotiation floor, ceiling, Rversion echo | **fixed + verified this cycle**: the reply now echoes the request tag (was hardcoded 0; clients send NOTAG), enforces a 64-byte msize floor below which Rversion cannot fit, clamps to srv->max_msize, and refuses oversized messages after negotiation | `make test_styx_serve` ALL PASSED -- NOTAG 0xFFFF echoed, msize 32 -> Rerror, ask 8192 vs ceiling 1024 -> echoed 1024, post-negotiation 8192-byte message -> Rerror |
| RUNTIME #716 | styx_enc has no test target | **closed this cycle**: `test_styx_enc` wired into mk/tests.mk + the test_medium_other aggregate; asserts the wire layout of all 13 T-message builders (8 legacy + 5 new) plus Rerror | `make test_styx_enc` ALL PASSED |
| RUNTIME #723 | styx_serve has no test target | **closed this cycle**: `test_styx_serve` wired into mk/tests.mk + the test_medium_other aggregate; the 9P2000 conformance suite for the dispatch | `make test_styx_serve` ALL PASSED |
| RUNTIME #717/#718 | styx_flush (test) has no test target | **verified, not written**: target existed and passes | `make test_styx_flush`: 6 passed, 0 failed |
| RUNTIME #724/#725 | styx_socket (test) has no test target | **verified, not written**: target existed and passes | `make test_styx_socket`: 4 passed, 0 failed |
| RUNTIME #727 | styx_tags has no test target | **verified, not written**: target existed and passes | `make test_styx_tags`: 5 passed, 0 failed |
| RUNTIME #728 | styx_test has no test target | **verified, not written**: test_styx runs it | `make test_styx`: 32/32 passed, 0 failed |
| RUNTIME #729/#730 | styxfs_callbacks (test) has no test target | **verified, not written**: test_styxfs_callbacks asserts the module | `make test_styxfs_callbacks`: 23 passed, 0 failed |
| KERNEL #569 | Tcreate/Tremove/Tclunk fid lifecycle with clunk-on-task-exit | **advanced, still OPEN**: the Tcreate wire arm now exists in the dispatch (was unreachable -- "Unknown message type"), re-roots the fid (open(5):18-30), and the Tclunk-frees-fid leg is asserted; the Tremove leg and clunk-on-task-exit remain unproven | `make test_styx_serve` ALL PASSED (create/re-root/clunk lifecycle); Tremove + task-exit legs: no assertions yet |
| KERNEL #570 | Tflush racing an in-flight Tread | **advanced, still OPEN**: Tflush now answers Rflush for any oldtag (was Unknown-message) and the flushtable begin/cancel/is_cancelled/end cycle is asserted; the actual flush-vs-in-flight-read race through the socket loop remains unproven | `make test_styx_serve` ALL PASSED (flush arm + flushtable); race leg: no assertions yet |
| BUILD-INTEGRITY | test recipes link pre-built .o files without make prerequisites | **fixed this cycle**: adding fields to styx_server_t made a stale styxfs_host.o (built 02:12, pre-change) ABI-incompatible; consumers failed with ENOENT cascades (test_ns_9p 15/3, test_styxfs_callbacks 16/7) while stash-proofs showed both rc=0 pre-change. A continuation-aware audit added every command-line $(RT)/*.o to its recipe's prerequisite list -- 39 targets hardened | rebuilt objects -> `make test_ns_9p` 18/0, `make test_styxfs_callbacks` 23/0; `make -n test` parses clean |
| BUILD-INTEGRITY | tests.mk:534 literal double-backslash passed to ld | **fixed this cycle**: `$(RT)/styx_locks.c \\` (two backslashes) at the end of test_ns_brain's link line made ld read a stray `\` argument -- `cannot find \`; pre-existing (HEAD-stash proof: same rc=2). One character collapsed | `make test_ns_brain`: 10 checks, 0 failures; `make test_ns_resolve`: 8 checks, 0 failures |
