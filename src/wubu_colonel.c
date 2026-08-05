/*
 * wubu_colonel.c — the everything-through-the-Colonel dispatcher. C11.
 *
 * Port of WuBuOS/src/runtime/wubu_colonel.c. Made self-contained:
 *   - Opaque struct (internals in .c, not .h)
 *   - Minimal includes (string, ctype, stdlib)
 *   - No external dependencies
 *   - All buffers bounded
 *
 * SPDX-License-Identifier: WaefreBeorn-UMV3
 */
#include "wubu_colonel.h"
#include <string.h>
#include <ctype.h>
#include <stdlib.h>

/* The registered app names the Colonel routes for "run <name>". */
static const char *const g_apps[] = {
    "calc", "notepad", "paint", "explorer", "terminal",
    "holyc", "controlpanel", "taskmgr", "canvas", "freedoom",
    "bonzi", "comfy", "settings", "packagemanager", "containermanager",
    "sound", "music", "browser", "notes", "todo"
};
#define N_APPS (int)(sizeof(g_apps) / sizeof(g_apps[0]))

struct wubu_colonel {
    int   class;       /* WUBU_COL_CMD_* */
    int   result;      /* WUBU_COLONEL_* */
    int64_t value;     /* eval result */
    char  cmd[64];     /* parsed command word */
    char  arg[256];    /* parsed argument */
};

/* Skip leading whitespace. Returns 1 if non-whitespace remains. */
static int skip_ws(const char **s) {
    while (**s == ' ' || **s == '\t') (*s)++;
    return **s != '\0';
}

/* Extract one word. Returns chars written (excl NUL). */
static int word(const char **s, char *out, int cap) {
    int n = 0;
    while (**s && **s != ' ' && **s != '\t' && n < cap - 1)
        out[n++] = *(*s)++;
    out[n] = '\0';
    return n;
}

int wubu_colonel_parse(const char *line, wubu_colonel_t *c) {
    if (!line || !c) return WUBU_COLONEL_BAD;
    memset(c, 0, sizeof(*c));
    c->result = WUBU_COLONEL_OK;

    const char *s = line;
    if (!skip_ws(&s)) return WUBU_COLONEL_EMPTY;

    if (strncmp(s, "run ", 4) == 0) {
        s += 4;
        c->class = WUBU_COL_CMD_APP;
        word(&s, c->cmd, sizeof(c->cmd));
        return WUBU_COLONEL_OK;
    }
    if (strncmp(s, "eval ", 5) == 0) {
        s += 5;
        c->class = WUBU_COL_CMD_EVAL;
        strncpy(c->arg, s, sizeof(c->arg) - 1);
        c->arg[sizeof(c->arg) - 1] = '\0';
        return WUBU_COLONEL_OK;
    }
    if (strncmp(s, "os ", 3) == 0) {
        s += 3;
        c->class = WUBU_COL_CMD_OS;
        word(&s, c->cmd, sizeof(c->cmd));
        return WUBU_COLONEL_OK;
    }
    if (strncmp(s, "sys ", 4) == 0) {
        s += 4;
        c->class = WUBU_COL_CMD_SYS;
        word(&s, c->cmd, sizeof(c->cmd));
        return WUBU_COLONEL_OK;
    }
    if (strncmp(s, "agi ", 4) == 0) {
        s += 4;
        c->class = WUBU_COL_CMD_AGI;
        word(&s, c->cmd, sizeof(c->cmd));
        return WUBU_COLONEL_OK;
    }
    if (strncmp(s, "load ", 5) == 0) {
        s += 5;
        c->class = WUBU_COL_CMD_LOAD;
        word(&s, c->cmd, sizeof(c->cmd));
        skip_ws(&s);
        strncpy(c->arg, s, sizeof(c->arg) - 1);
        c->arg[sizeof(c->arg) - 1] = '\0';
        return WUBU_COLONEL_OK;
    }

    /* A bare token defaults to the APP class (run shorthand). */
    c->class = WUBU_COL_CMD_APP;
    word(&s, c->cmd, sizeof(c->cmd));
    return WUBU_COLONEL_OK;
}

int wubu_colonel_dispatch(const char *line, wubu_colonel_t *c,
                          int64_t (*eval_fn)(const char *)) {
    if (!line || !c) return WUBU_COLONEL_BAD;
    int r = wubu_colonel_parse(line, c);
    if (r != WUBU_COLONEL_OK) return r;

    switch (c->class) {
    case WUBU_COL_CMD_EVAL:
        if (!eval_fn) return WUBU_COLONEL_BAD;
        c->value = eval_fn(c->arg);
        return WUBU_COLONEL_OK;
    case WUBU_COL_CMD_APP:
        return wubu_colonel_app_known(c->cmd) ? WUBU_COLONEL_OK
                                              : WUBU_COLONEL_UNKNOWN;
    case WUBU_COL_CMD_OS:
    case WUBU_COL_CMD_SYS:
    case WUBU_COL_CMD_AGI:
    case WUBU_COL_CMD_LOAD:
        return c->cmd[0] ? WUBU_COLONEL_OK : WUBU_COLONEL_UNKNOWN;
    default:
        return WUBU_COLONEL_UNKNOWN;
    }
}

int wubu_colonel_app_known(const char *name) {
    if (!name || !name[0]) return 0;
    for (int i = 0; i < N_APPS; i++)
        if (strcmp(g_apps[i], name) == 0) return 1;
    return 0;
}

int wubu_colonel_get_class(const wubu_colonel_t *c) {
    return c ? c->class : 0;
}

int wubu_colonel_get_result(const wubu_colonel_t *c) {
    return c ? c->result : WUBU_COLONEL_BAD;
}

int64_t wubu_colonel_get_value(const wubu_colonel_t *c) {
    return c ? c->value : 0;
}

const char *wubu_colonel_get_cmd(const wubu_colonel_t *c) {
    return c ? c->cmd : NULL;
}

const char *wubu_colonel_get_arg(const wubu_colonel_t *c) {
    return c ? c->arg : NULL;
}

/* wubu_colonel_free is a no-op stub — struct is stack-allocated. */
/* Kept for API symmetry with potential future heap-alloc extensions. */
void wubu_colonel_free(wubu_colonel_t *c) {
    (void)c;
}
