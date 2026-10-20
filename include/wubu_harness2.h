/* wubu_harness2.h -- Harness^2 loop in C11.
 *
 * Two-level recursion with fixed model weights, after
 * Harness^2: Recursive Agent Harnessing for an Open World (arXiv 2610.14224):
 *
 *   task level      propose harness edits -> probe each on the same task ->
 *                   compose from the behavioural contrast
 *   harnessing level persist adopted harnesses and evidence-backed lessons
 *
 * This module owns the parts that must be exact: the editable rule set, the
 * rendered harness text, probe scoring by a CHECK COMMAND's exit status,
 * composition, and an append-only JSONL ledger. It does not need a model to be
 * tested: a probe is scored by running a command, so the loop's decisions are
 * decided by bytes on disk and exit codes.
 *
 * The model call itself is an argv builder plus an optional fork/exec, so the
 * command line can be asserted without spending a turn.
 */
#ifndef WUBU_HARNESS2_H
#define WUBU_HARNESS2_H

#include <stddef.h>

#define WUBU_H2_MAX_RULES     32
#define WUBU_H2_RULE_LEN      192
#define WUBU_H2_MAX_PROBES    16
#define WUBU_H2_ARGV_MAX      16
#define WUBU_H2_PATH_LEN      512

typedef struct wubu_h2_harness wubu_h2_harness_t;

/* One probe's evidence. `check_rc` is the exit status of the task's check
 * command; pass is defined as check_rc == 0, never as the executor's claim. */
typedef struct {
    int  probe_index;
    int  rules_count;               /* rules the probe ran with */
    int  check_rc;
    long elapsed_ms;
    char check_tail[256];           /* last bytes the check printed */
    const wubu_h2_harness_t *harness; /* the harness this probe ran (not owned) */
} wubu_h2_probe_t;

/* Contrast verdict, after the paper's edit attribution. Our setting DOES have a
 * check command, so a pass/fail flip is real evidence; anything else is
 * confounded rather than credited. */
typedef enum {
    WUBU_H2_HELPED = 0,      /* reference failed, candidate passed */
    WUBU_H2_HURT,            /* reference passed, candidate failed */
    WUBU_H2_NO_EFFECT,       /* same result both ways */
    WUBU_H2_CONFOUNDED       /* several edits at once, or missing reference */
} wubu_h2_verdict_t;

/* -- harness ---------------------------------------------------------- */
wubu_h2_harness_t *wubu_h2_harness_new(void);
void wubu_h2_harness_free(wubu_h2_harness_t *h);

/* Append a rule. Duplicates and a full table are refused (-1), not silently
 * ignored: the caller is composing a harness and must know. */
int wubu_h2_harness_add_rule(wubu_h2_harness_t *h, const char *rule);
int wubu_h2_harness_rule_count(const wubu_h2_harness_t *h);
const char *wubu_h2_harness_rule(const wubu_h2_harness_t *h, int i);
int wubu_h2_harness_has_rule(const wubu_h2_harness_t *h, const char *rule);

/* Render the harness as AGENTS.md text. Returns the number of bytes written
 * (excluding NUL) or -1 when the buffer is too small. Deterministic: same rules
 * in the same order -> same bytes. */
int wubu_h2_harness_render(const wubu_h2_harness_t *h, char *out, size_t cap);

/* Write the rendered harness into `dir` as AGENTS.md, creating the directory
 * when needed. Returns 0 on success, -1 on I/O error. */
int wubu_h2_harness_stage(const wubu_h2_harness_t *h, const char *dir);

/* -- probes ------------------------------------------------------------ */
/* Run `check_cmd` (shell) with `timeout_s` seconds. 0 = pass. A timeout is
 * reported as 124 so it can never be mistaken for success. */
int wubu_h2_probe_run(wubu_h2_probe_t *out, int probe_index,
                      const wubu_h2_harness_t *h, const char *check_cmd,
                      int timeout_s);

/* -- composition ------------------------------------------------------- */
/* Index of the first passing probe, or -1 when none passed. The baseline
 * probe (index 0, no edits) participates like any other: a harness only adopts
 * an edit when an edit is what passed. */
int wubu_h2_compose(const wubu_h2_probe_t *probes, int n);

/* Classify a candidate against the reference probe (index 0 by convention). */
wubu_h2_verdict_t wubu_h2_verdict(const wubu_h2_probe_t *ref,
                                  const wubu_h2_probe_t *cand);

/* Copy the adopted harness's rule text (the harness probe `winner` ran with)
 * into `out`. Returns the count copied, -1 for a bad index. */
int wubu_h2_adopt(const wubu_h2_probe_t *probes, int n, int winner,
                  char out[][WUBU_H2_RULE_LEN], int cap);

/* -- lessons / ledger (harnessing-level recursion) --------------------- */
/* Append one JSON object per line to `path`. Creates the file. Refuses strings
 * that would need JSON escaping beyond quote/backslash/newline (returns -1),
 * so a ledger line always parses. */
int wubu_h2_ledger_append(const char *path, const char *task_id,
                          const wubu_h2_probe_t *probes, int n, int adopted_probe);

/* Number of well-formed records in `path` (0 when absent). */
int wubu_h2_ledger_count(const char *path);

/* -- executor argv ----------------------------------------------------- */
/* Build: hermes -z <prompt> --in <workspace> [-m <model>] --pass-session-id
 * into `argv` (NULL-terminated). Returns the argument count, or -1 on overflow.
 * The strings are pointers INTO the caller's storage; nothing is allocated. */
int wubu_h2_executor_argv(char *argv[], int cap,
                          const char *hermes_bin, const char *prompt,
                          const char *workspace, const char *model);

#endif /* WUBU_HARNESS2_H */
