# NT object namespace (`vsl_nt_ob`) — status

Library: `wubuos/src/runtime/vsl/vsl_nt_ob.c` + `.h`
Test: `wubuos/src/runtime/vsl/test_nt_ob.c` — `make test_nt_ob`, 28 checks, 0 failures.

## Closed against the criterion

| goal | what the test shows |
|---|---|
| 416 path canonicalisation | `\??\C:\Brain\Kv\slot0`, `/n/brain/kv/slot0` and `/N/BRAIN/KV/SLOT0` all produce `/n/brain/kv/slot0`; `..` escaping the root returns `OBJECT_PATH_INVALID`; dot segments and doubled slashes collapse |
| 399 `ObOpenByKeyName` | `vsl_nt_ob_bind_cap` gives a node a real `STYX_NODE` capability; `ObOpenByKeyName` resolves through it, returns rights equal to the cap's (`WUBU_RIGHT_READ`), denies a pid outside the audience, denies write without write rights, and denies everyone after `wubu_cap_revoke` |
| 415 access-mask table | `GENERIC_READ`→OREAD, `READ|WRITE`→ORDWR, `WRITE`→OWRITE, `DELETE`→ORCLOSE, `SYNCHRONIZE`→OREAD; an unmapped bit returns `ACCESS_DENIED` and reports the offending bit |

## Advanced but NOT closed

These four need the NT entry points or the cap binding, and marking them done
would be a claim the code cannot support:

- **425 `NtCreateDirectoryObject`.** `vsl_nt_ob_mkdir` creates a node and
  `vsl_nt_ob_query_dir` lists it, but there is no NT entry point and no
  cap-backed directory object.
- **427 `NtQueryDirectoryFile` paging.** Cursor paging, no-repeat and
  restart-identical behaviour are tested. The NT syscall wrapper is not written.
- **428 reparse symlinks.** Links resolve, open follows them, cycles are bounded
  at 8 hops. `Tstat` does not yet report `DMSYMLINK`; the two views a criterion
  would compare do not both exist.

## Next

1. Add `NtCreateDirectoryObject` / `NtQueryDirectoryObject` / `NtQueryDirectoryFile`
   wrappers over this library, which closes 425/427.
2. Add `Tstat` symlink reporting and compare it with the reparse tag, closing 428.
