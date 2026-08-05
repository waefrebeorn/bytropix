/*
 * wubu_colonel.h — the everything-through-the-Colonel dispatcher.
 *
 * C11 port of WuBuOS/src/runtime/wubu_colonel.{c,h}.
 *
 * The Colonel (TempleOS/ZealOS lineage) is the OS core: EVERY command
 * — app launches, OS actions, AGI evals — dispatches through it.
 * This module routes a command string to a typed result. Self-contained,
 * opaque struct, minimal includes.
 *
 * SPDX-License-Identifier: WaefreBeorn-UMV3
 */
#ifndef WUBU_COLONEL_H
#define WUBU_COLONEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dispatch result codes. */
enum {
    WUBU_COLONEL_OK = 0,       /* parsed/dispatched; result valid */
    WUBU_COLONEL_EMPTY,        /* empty command (no-op) */
    WUBU_COLONEL_UNKNOWN,      /* unknown command class or app */
    WUBU_COLONEL_EVAL_ERR,     /* the eval callback failed */
    WUBU_COLONEL_BAD           /* bad args (NULL, etc.) */
};

/* Command classes (what the Colonel routes). */
enum {
    WUBU_COL_CMD_APP = 1,      /* launch an app: "run <name>" */
    WUBU_COL_CMD_EVAL,         /* evaluate expression: "eval <src>" */
    WUBU_COL_CMD_OS,           /* OS action: "os <verb>" */
    WUBU_COL_CMD_SYS,          /* syscall: "sys <verb>" */
    WUBU_COL_CMD_AGI,          /* AGI action: "agi <verb>" */
    WUBU_COL_CMD_LOAD          /* load payload: "load <fmt> <path>" */
};

/* Opaque parse result. */
typedef struct {
    int   class;       /* WUBU_COL_CMD_* */
    int   result;      /* WUBU_COLONEL_* */
    int64_t value;     /* the eval result */
    char  cmd[64];     /* the parsed command word */
    char  arg[256];    /* the parsed argument */
} wubu_colonel_t;

/* Parse a command string into a typed result.
 * Returns WUBU_COLONEL_OK on success, WUBU_COLONEL_EMPTY if the line
 * is whitespace-only, WUBU_COLONEL_BAD if args are NULL.
 * Sets *out_result to the parse result. */
int wubu_colonel_parse(const char *line, wubu_colonel_t *out_result);

/* Dispatch: parse + evaluate through the provided eval callback.
 * Returns the result enum (WUBU_COLONEL_OK, WUBU_COLONEL_BAD, etc.). */
int wubu_colonel_dispatch(const char *line, wubu_colonel_t *out_result,
                          int64_t (*eval_fn)(const char *));

/* Check if an app name is in the registry.
 * Returns 1 if known, 0 otherwise. */
int wubu_colonel_app_known(const char *name);

/* Accessors for the opaque struct's parsed fields. */
int      wubu_colonel_get_class(const wubu_colonel_t *c);
int      wubu_colonel_get_result(const wubu_colonel_t *c);
int64_t  wubu_colonel_get_value(const wubu_colonel_t *c);
const char *wubu_colonel_get_cmd(const wubu_colonel_t *c);
const char *wubu_colonel_get_arg(const wubu_colonel_t *c);

/* Free a colonel result (no-op — struct is stack-allocated). */
/* This is kept as a no-op stub for API symmetry. */

#ifdef __cplusplus
}
#endif

#endif /* WUBU_COLONEL_H */
