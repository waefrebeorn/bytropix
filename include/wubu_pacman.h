#ifndef WUBU_PACMAN_H
#define WUBU_PACMAN_H
#include <stddef.h>

/* wubu_pacman.h -- OMARCHY #272 Arch base: pacman package manager driver.
 *
 * The driver layer between the AGI kernel and Arch's pacman: it parses
 * pacman-style command lines, classifies the operation (query -Q, sync -S,
 * remove -R, database -D, etc.), validates arguments, and dispatches to a
 * pluggable exec backend that actually runs the command (per the RUN POLICY:
 * third-party software runs ON the AGI kernel via the wubuos exec backends).
 * The driver owns the policy/parsing/validation; the exec backend owns the
 * execution. Fail-closed: an unknown/malformed command is refused before any
 * exec. Deterministic, no allocation, real work (no stubs). */

typedef enum {
    WUBU_PAC_QUERY=0,   /* -Q: query the local package database */
    WUBU_PAC_SYNC,      /* -S: sync/install from repos */
    WUBU_PAC_REMOVE,    /* -R: remove packages */
    WUBU_PAC_DATABASE,  /* -D: modify the package database */
    WUBU_PAC_FILES,     /* -F: query files owned by packages */
    WUBU_PAC_UPGRADE,   /* -U: install from a local file */
    WUBU_PAC_DEPTEST,   /* -T: check dependencies */
    WUBU_PAC_INVALID,   /* unrecognised */
} wubu_pac_op_t;

/* Common operation flags (subset pacman supports). */
#define WUBU_PAC_FLAG_INFO    1u<<0  /* -i / -Qi: package info */
#define WUBU_PAC_FLAG_LIST    1u<<1  /* -Ql: list files */
#define WUBU_PAC_FLAG_REFRESH 1u<<2  /* -Sy: refresh db */
#define WUBU_PAC_FLAG_SYSUP   1u<<3  /* -Syu: full system upgrade */
#define WUBU_PAC_FLAG_NOCONF  1u<<4  /* --noconfirm */
#define WUBU_PAC_FLAG_NEEDED  1u<<5  /* --needed: skip if installed */

#define WUBU_PAC_MAX_ARGS 32

typedef struct {
    wubu_pac_op_t op;
    unsigned flags;
    char targets[WUBU_PAC_MAX_ARGS][64];  /* package names / files */
    int n_targets;
} wubu_pac_cmd_t;

/* Parse a pacman-style command line (argv, argc). Returns 0 on a valid,
 * recognised command (fills cmd), -1 on invalid/malformed (fail-closed).
 * argv[0] is the program name ("pacman"); options start with '-'. */
int wubu_pacman_parse(const char *const *argv, int argc, wubu_pac_cmd_t *cmd);

/* Validate: a command needs the right operands (e.g. -S/-R/-U need targets;
 * -Q/-D may not). Returns 0 if valid, -1 if not. */
int wubu_pacman_validate(const wubu_pac_cmd_t *cmd);

/* Human-readable op name (for logging / exec-backend dispatch). */
const char *wubu_pac_op_name(wubu_pac_op_t op);

/* Should this command touch the network (sync/upgrade)? For egress gating. */
int wubu_pacman_needs_network(const wubu_pac_cmd_t *cmd);

#endif
