# B2 remainder — why three goals cannot close in the hosted build

Closed in B2: 45 (EEVDF), 46 (RT class), 47 (timer wheel), 48 (lifecycle),
925 (cap ceiling). All five are wired into `task_schedule_next` /
`task_timer_tick` and measured by `make test_tasking`.

Open in B2: 66, 67, 68. Each one needs a task to execute on its own kernel
stack, and the hosted build cannot provide that.

## The blocker

`tasking.c` states it in its own header: hosted mode switches contexts with
`setjmp`/`longjmp`. Those run on the **host** thread's stack. `CTask.stack_base`
is allocated and canary-checked, but no hosted task ever executes on it.

Consequences, goal by goal:

- **66 stack high-water watermark.** The criterion is "a task that recurses to
  depth N reports a stack high-water within one cacheline of N frames". There is
  no exit path: in hosted mode the recursion happens on the host stack, so a
  measurement of `CTask.stack_base` would read the untouched fill pattern and
  report zero no matter what the task did. Implementing the scan and asserting
  success would be a fabricated result.
- **67 TLS (FS/GS base).** The criterion requires two tasks reading different
  `tls_var` values at the same address *and* `make test_arch` asserting the
  bases are programmed. Setting FS/GS base is a real CPU operation; hosted code
  has no task-local FS/GS to program.
- **68 idle task with HLT.** The criterion asks for a QEMU boot log showing the
  BSP parked in idle with the tick advancing. That needs a boot, not a library
  test.

## What would unblock them

A hosted execution path that switches to `CTask.stack_base` with a real stack
pointer (assembly stub, `-DWUBU_HOSTED` off) or a boot under QEMU. Until then
these three stay open with this reason recorded, rather than being closed on a
pattern-scan test that cannot see the stack they describe.

## Next batch

B4 (STYX / 9P: `styx_serve` 8, `styxfs_callbacks` 8, `styxfs_server` 5) is
hosted-testable and does not need a boot. That is the next cluster.
