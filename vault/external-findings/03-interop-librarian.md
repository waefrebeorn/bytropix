# Vault: The Interop Librarian — self-growing cross-system translation

> 2026-10-11. User directive: "make a librarian that helps make em all
> friendly too so they can find each other and apps can call other systems
> and be seeing what they want — a 'building as it goes' translation
> library." This is what we built and the external work it sits on.

## What exists now (verified, gate-green)

Four modules in `wubuos/src/runtime/`, all with state-change tests:

| Module | Job | Test |
|---|---|---|
| `wubu_abi_path.{c,h}` | foreign PATHS → `/n/foreign/<abi>/...` (win C:\, unix /usr, dos drives; backslash→slash, drive lowercased + preserved so drives don't collide) | `test_abi_path` 7/7 |
| `wubu_syscall_shim.{c,h}` | foreign SYSCALLS → WuBu ops (file→9P on translated path, graphics→VBE/WM, net→9P net, unmapped→REFUSE fail-closed) | `test_syscall_shim` 8/8 |
| `wubu_librarian.{c,h}` | the LIBRARIAN: registry (name/ABI/root/caps), capability resolver (fail-closed), **growing translation table** | `test_librarian` 13/13 |
| `wubu_librarian_serve.{c,h}` | live front door; self-registers default WuBu participants (shell/kv/models/bonzi/experts/hive); FILE ops resolve against the target's registered root | `test_librarian_serve` 10/10 |

`wubu_exec` registers every foreign binary as a participant under its
detected ABI at dispatch — a Windows .exe / DOS .com / Mach-O app becomes a
reachable, findable participant the instant it's exec'd.

## The "building as it goes" property

The translation table is the key idea. The FIRST time a call crosses an ABI
boundary, the librarian computes the mapping (via the shim) and RECORDS it
(from_abi, to_abi, op, how). Every later crossing of the same kind is a
direct lookup. The library accumulates what it has actually had to translate
— no giant hardcoded table, it learns its own dictionary from real traffic.
Verified: re-routing the same crossing does not re-grow the table.

## External work this sits on (Kevin-Bacon, verified against primary sources)

- **Plan 9 / Styx 9P** (Pike; inferno-os Styx paper): "present resources as
  files in a hierarchical name space… represent stored data, but also
  devices, dynamic information sources, interfaces to services, and control
  points." The translator model: a foreign OS root is a file server you
  attach. Our librarian is the attachment + routing layer above that.
- **Plan 9 user-level translators** (NFS/disk served by user processes):
  the precedent for serving a foreign ABI's expectations as files rather
  than kernel special-cases. Our shim is the per-ABI translator.
- **Capability security via namespace** (Styx private malleable namespaces):
  a process's attach set is its capability set. Our AD01 (`wubu_9p_cap_allowed`)
  + the librarian's fail-closed resolver are the same principle — an unknown
  participant or un-offered capability resolves to not-found, never a guess.

## Design decisions worth recording

1. **Fail-closed everywhere.** Unmapped syscall → REFUSE; unknown
   participant → not found; un-offered capability → not found. Never a
   silent guess. (The capability-security principle applied to interop.)
2. **Grow, don't pre-build.** The translation dictionary is learned from
   real crossings, not hardcoded. Scales to ABIs we haven't enumerated.
3. **Resolve against the target's registered root.** A FILE op joins the
   target's namespace root + the cleaned path — the target is a WuBu-native
   9P participant, so its root is where the file lives. (Fixed an early
   version that used a generic foreign translation and produced doubled
   path segments.)
4. **Dependency-light modules.** Each layer links only the one below it;
   the GUI/exec link the chain via weak symbols where needed so nothing
   hard-requires the whole stack.
