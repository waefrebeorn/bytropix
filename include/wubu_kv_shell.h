/* wubu_kv_shell.h — shell command routing to KV/FS (Phase 11)
 *
 * Exposes the KV filesystem as a shell-accessible namespace.
 * Commands like `ls /kv/in/` and `cat /kv/in/file.txt` route through
 * the KV embedding layer so the shell sees the same context the
 * model sees.
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 11 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_KV_SHELL_H
#define WUBU_KV_SHELL_H

#include "wubu_kv_embedding.h"
#include "wubu_fs_dataset.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_kv_shell wubu_kv_shell_t;

/* Create the KV shell. The shell references (does not own) the
 * kv embedding layer and fs dataset. kv_base is the flat KV tensor
 * pointer (needed for cat/stat data reads); may be NULL if cat
 * is not used. */
wubu_kv_shell_t *wubu_kv_shell_create(wubu_kv_embedding_t *kv,
                                       wubu_fs_dataset_t *dataset,
                                       float *kv_base);

/* Execute a shell-like command against the KV namespace.
 *
 * Supported commands (mimic Unix shell):
 *   ls  /kv/in/<dir>     → list files in directory (one per line)
 *   cat /kv/in/<file>    → print file content as decoded bytes
 *   stat /kv/in/<file>   → print KV metadata (path, blocks, tokens)
 *
 * Returns 0 on success, -1 on error (unknown command, path not found, etc).
 * Command output is written to out (caller-allocated buffer of out_cap).
 * For `cat`, the decoded bytes are written; this requires kv_base to
 * be non-NULL at creation time. */
int wubu_kv_shell_exec(wubu_kv_shell_t *shell,
                        const char *command, const char *path,
                        char *out, size_t out_cap);

/* Free the KV shell. Does NOT free kv, dataset, or kv_base. */
void wubu_kv_shell_free(wubu_kv_shell_t *shell);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_KV_SHELL_H */
