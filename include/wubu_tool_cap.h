/*
 * wubu_tool_cap.h -- capability-gated tool dispatch (AX04 + AD01).
 *
 * Wraps wubu_tool_call with an AD01 9P capability check: a tool is bound
 * to a capability path, and a call is refused unless the caller's subtree
 * owns that path. The agent's namespace attach set IS its capability set.
 * Unbound tools are deny-by-default. See wubu_tool_cap.c.
 */
#ifndef WUBU_TOOL_CAP_H
#define WUBU_TOOL_CAP_H

#define WUBU_TOOL_CAP_MAX 64

/* the tool registry type (from wubu_tooluse.h) */
#include "wubu_tooluse.h"

typedef struct {
    char tool_name[64];   /* the registered tool name */
    char cap_path[256];   /* the 9P subtree that owns this tool */
} wubu_tool_cap_entry_t;

typedef struct {
    wubu_tool_cap_entry_t entries[WUBU_TOOL_CAP_MAX];
    int n;
} wubu_tool_cap_t;

/* Bind a registered tool to a capability path. Returns 0 on success, -1 on
 * a full registry. */
int wubu_tool_cap_bind(wubu_tool_cap_t *caps, const char *tool_name,
                       const char *cap_path);

/* Capability-gated dispatch. Returns the handler's result on an allowed
 * call; -1 on bad args/unknown tool; -2 when DENIED (caller's subtree does
 * not own the tool's cap path, OR the tool has no bound capability --
 * deny-by-default). The handler is never invoked on a denial. */
int wubu_tool_cap_call(const wubu_tool_cap_t *caps,
                       wubu_tool_registry_t *reg,
                       const char *caller_subtree, const char *name,
                       const char *args_json, char *result_buf, int buf_size);

#endif /* WUBU_TOOL_CAP_H */
