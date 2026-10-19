# Defect: directory Rread paging mixes two entry sources (goal 410)

Found while writing the test for goal 410 (directory read batching). The test is
not committed, because it fails for a real reason.

## What was measured

12 synthetic files `e00..e11` in one directory, read through
`styxfs_read_cb` at entry-index offsets 0, 4 and 8 with different `count`
values:

| count | entries per page | result over 3 reads |
|------:|-----------------:|---------------------|
| 200 | 3 | 9 of 12 seen, 0 duplicates |
| 400 | >4 | 8 seen, **4 duplicates** |
| 312 | 4 | 10 seen, **2 duplicates** |
| 156 | 2 | 6 seen, 0 duplicates |

No `count` gives 12 unique entries in three reads.

## Cause

`styxfs_read_cb` enumerates directory entries from **two sources** in one pass:
the server's mount list, then the host directory. Paging treats `offset` as an
entry index (`if (entry_count < dir_offset) { entry_count++; continue; }`), but
the counter increments across *both* sources while only one of them contributes
the names the caller sees. Skipping N entries therefore does not skip N visible
names, so pages either overlap or drop entries depending on how many mount
entries precede the host entries.

There is a second issue worth naming: real 9P `offset` in a directory read is a
**byte offset into the directory stream**, not an entry index. Clients advance
by the number of bytes returned. The current code cannot satisfy that contract
at all, and a client that follows 9P conventions would read garbage.

## What a fix requires

1. Build the full entry list for the directory (mounts + host), not two
   interleaved counters.
2. Page by byte offset: encode records in order, find the record that starts at
   or after `offset`, and return whole records up to `count`.
3. Return a count that ends on a record boundary, so a client that advances by
   `nread` lands on the next record.

Goal 410 stays open until that is done. The three reads in the goal's criterion
are only meaningful once offsets are bytes.

## Unaffected

The other four B4 goals closed on the callback harness (404, 405, 406, 419) do
not touch directory reads.