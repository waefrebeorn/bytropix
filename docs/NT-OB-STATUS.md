# NT object namespace (`vsl_nt_ob`) — status

Library: `wubuos/src/runtime/vsl/vsl_nt_ob.c` + `.h`
Test: `wubuos/src/runtime/vsl/test_nt_ob.c` — `make test_nt_ob`, 28 checks, 0 failures.

## Closed against the criterion

| goal | what the test shows |
|---|---|
| 416 path canonicalisation | `\??\C:\Brain\Kv\slot0`, `/n/brain/kv/slot0` and `/N/BRAIN/KV/SLOT0` all produce `/n/brain/kv/slot0`; `..` escaping the root returns `OBJECT_PATH_INVALID`; dot segments and doubled slashes collapse |
| 415 access-mask table | `GENERIC_READ`→OREAD, `READ|WRITE`→ORDWR, `WRITE`→OWRITE, `DELETE`→ORCLOSE, `SYNCHRONIZE`→OREAD; an unmapped bit returns `ACCESS_DENIED` and reports the offending bit |

## Advanced but NOT closed

These four need the NT entry points or the cap binding, and marking them done
would be a claim the code cannot support:

- **399 `ObOpenByKeyName` over cap-backed nodes.** `vsl_nt_ob_open` checks node
  rights and returns a handle, but the node store is an in-memory tree, not
  `wubu_cap_get(STYX_NODE)`. The rights gate is real; the cap binding is not.
- **425 `NtCreateDirectoryObject`.** `vsl_nt_ob_mkdir` creates a node and
  `vsl_nt_ob_query_dir` lists it, but there is no NT entry point and no
  cap-backed directory object.
- **427 `NtQueryDirectoryFile` paging.** Cursor paging, no-repeat and
  restart-identical behaviour are tested. The NT syscall wrapper is not written.
- **428 reparse symlinks.** Links resolve, open follows them, cycles are bounded
  at 8 hops. `Tstat` does not yet report `DMSYMLINK`; the two views a criterion
  would compare do not both exist.

## Next

1. Bind `ob_node_t` to `wubu_cap` (kind `STYX_NODE`), so open returns a real cap
   handle and 399/425 close.
2. Add `NtCreateDirectoryObject` / `NtQueryDirectoryObject` / `NtQueryDirectoryFile`
   wrappers over this library, which closes 425/427.
3. Add `Tstat` symlink reporting and compare it with the reparse tag, closing 428.
